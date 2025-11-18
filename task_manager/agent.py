import logging
import asyncio
import uuid
from typing import List, Dict, Any, Optional

from core.interfaces import (
    TaskManagerInterface, TaskDefinition, Program, BaseAgent,
    PromptDesignerInterface, CodeGeneratorInterface, EvaluatorAgentInterface,
    DatabaseAgentInterface, SelectionControllerInterface
)
from core.fitness import selection_sort_key, ensure_program_fitness
from config import settings

from prompt_designer.agent import PromptDesignerAgent
from code_generator.agent import CodeGeneratorAgent
from evaluator_agent.agent import EvaluatorAgent
from database_agent.agent import InMemoryDatabaseAgent
from selection_controller.agent import SelectionControllerAgent

logger = logging.getLogger(__name__)

class TaskManagerAgent(TaskManagerInterface):
    def __init__(self, task_definition: TaskDefinition, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.task_definition = task_definition
        self.prompt_designer: PromptDesignerInterface = PromptDesignerAgent(task_definition=self.task_definition)
        self.code_generator: CodeGeneratorInterface = CodeGeneratorAgent()
        self.evaluator: EvaluatorAgentInterface = EvaluatorAgent(task_definition=self.task_definition)
        
        # Simplified database initialization:
        self.database: DatabaseAgentInterface = InMemoryDatabaseAgent()
        logger.info(f"Using {settings.DATABASE_TYPE} database (InMemoryDatabaseAgent).")
            
        self.selection_controller: SelectionControllerInterface = SelectionControllerAgent(task_definition=self.task_definition)

        self.population_size = settings.POPULATION_SIZE
        self.num_generations = settings.GENERATIONS
        self.num_parents_to_select = self.population_size // 2
        self.num_islands = settings.NUM_ISLANDS
        self.programs_per_island = self.population_size // self.num_islands

    async def initialize_population(self) -> List[Program]:
        logger.info(f"Initializing population for task: {self.task_definition.id}")
        initial_population = []
        
        initial_model = settings.LLM_SECONDARY_MODEL # Use secondary for broad initial generation
        logger.info(f"Using model '{initial_model}' for initial population generation.")

        # Generate initial programs
        tasks = []
        for i in range(self.population_size):
            initial_prompt = self.prompt_designer.design_initial_prompt()
            tasks.append(self.code_generator.generate_code(initial_prompt, model_name=initial_model, temperature=0.8))

        generated_codes = await asyncio.gather(*tasks)
        for i, generated_code in enumerate(generated_codes):
            program_id = f"{self.task_definition.id}_gen0_prog{i}"
            logger.debug(f"Generated initial program {i+1}/{self.population_size} with id {program_id}")
            program = Program(
                id=program_id,
                code=generated_code,
                generation=0,
                status="unevaluated",
                task_id=self.task_definition.id,
                prompt_id=getattr(self.prompt_designer, "last_prompt_metadata", {}).get("template_id") if hasattr(self.prompt_designer, "last_prompt_metadata") else None,
            )
            initial_population.append(program)
            await self.database.save_program(program)

        # Initialize islands with the initial population
        self.selection_controller.initialize_islands(initial_programs=initial_population)
        
        logger.info(f"Initialized population with {len(initial_population)} programs across {self.num_islands} islands.")
        return initial_population

    async def evaluate_population(self, population: List[Program]) -> List[Program]:
        logger.info(f"Evaluating population of {len(population)} programs.")
        evaluated_programs = []
        programs_to_evaluate = [prog for prog in population if prog.status != "evaluated"]
        evaluation_tasks = [self.evaluator.evaluate_program(prog, self.task_definition) for prog in programs_to_evaluate]
        
        results = await asyncio.gather(*evaluation_tasks, return_exceptions=True)
        
        for original_program, result in zip(programs_to_evaluate, results):
            if isinstance(result, Exception):
                logger.error(f"Error evaluating program {original_program.id}: {result}", exc_info=result)
                original_program.status = "failed_evaluation"
                original_program.errors.append(str(result))
                evaluated_programs.append(original_program)
            else:
                evaluated_programs.append(result)
            evaluated_programs[-1].task_id = self.task_definition.id
            await self.database.save_program(evaluated_programs[-1])

        # Preserve already evaluated programs
        evaluated_programs.extend([prog for prog in population if prog.status == "evaluated"])
            
        logger.info(f"Finished evaluating population. {len(evaluated_programs)} programs processed.")
        return evaluated_programs

    async def manage_evolutionary_cycle(self):
        logger.info(f"Starting evolutionary cycle for task: {self.task_definition.description[:50]}...")
        current_population = await self.initialize_population()
        current_population = await self.evaluate_population(current_population)

        for gen in range(1, self.num_generations + 1):
            logger.info(f"--- Generation {gen}/{self.num_generations} ---")

            # Select parents from islands
            parents = self.selection_controller.select_parents(current_population, self.num_parents_to_select)
            if not parents:
                logger.warning(f"Generation {gen}: No parents selected. Ending evolution early.")
                break
            logger.info(f"Generation {gen}: Selected {len(parents)} parents.")

            # Generate offspring
            offspring_population = []
            num_offspring_per_parent = (self.population_size + len(parents) - 1) // len(parents)
            
            generation_tasks = []
            for i, parent in enumerate(parents):
                for j in range(num_offspring_per_parent):
                    child_id = f"{self.task_definition.id}_gen{gen}_child{i}_{j}"
                    generation_tasks.append(self.generate_offspring(parent, gen, child_id))
            
            generated_offspring_results = await asyncio.gather(*generation_tasks, return_exceptions=True)

            for result in generated_offspring_results:
                if isinstance(result, Exception):
                    logger.error(f"Error generating offspring: {result}", exc_info=result)
                elif result:
                    offspring_population.append(result)
                    await self.database.save_program(result)

            logger.info(f"Generation {gen}: Generated {len(offspring_population)} offspring.")
            if not offspring_population:
                logger.warning(f"Generation {gen}: No offspring generated. May indicate issues with LLM or prompting.")
                if not parents:
                    break

            # Evaluate offspring
            offspring_population = await self.evaluate_population(offspring_population)

            # Select survivors using island model
            current_population = self.selection_controller.select_survivors(current_population, offspring_population, self.population_size)
            logger.info(f"Generation {gen}: New population size: {len(current_population)}.")

            # Log best program in this generation
            best_program_this_gen = sorted(
                current_population,
                key=lambda p: selection_sort_key(p, self.task_definition),
                reverse=True
            )
            if best_program_this_gen:
                best_program = best_program_this_gen[0]
                ensure_program_fitness(best_program, self.task_definition)
                logger.info(
                    f"Generation {gen}: Best program: ID={best_program.id}, "
                    f"Fitness={best_program.fitness:.4f}, Metrics={best_program.metrics or best_program.fitness_scores}"
                )
            else:
                logger.warning(f"Generation {gen}: No programs in current population after survival selection.")
                break

        logger.info("Evolutionary cycle completed.")
        final_best = await self.database.get_best_programs(task_id=self.task_definition.id, limit=1)
        if final_best:
            best_program = final_best[0]
            ensure_program_fitness(best_program, self.task_definition)
            logger.info(
                f"Overall Best Program: {best_program.id}, Code:\n{best_program.code}\n"
                f"Fitness: {best_program.fitness:.4f}, Metrics: {best_program.metrics or best_program.fitness_scores}"
            )
        else:
            logger.info("No best program found at the end of evolution.")
        return final_best

    async def generate_offspring(self, parent: Program, generation_num: int, child_id: str) -> Optional[Program]:
        logger.debug(f"Generating offspring from parent {parent.id} for generation {generation_num}")
        
        prompt_type = "mutation"
        # Default to primary for bug fixes or high-fitness mutations
        chosen_model = settings.LLM_PRIMARY_MODEL 
        parent_metrics = parent.metrics or parent.fitness_scores
        parent_fitness = ensure_program_fitness(parent, self.task_definition)

        if parent.errors and parent_metrics.get("correctness", 1.0) < settings.BUG_FIX_CORRECTNESS_THRESHOLD:
            primary_error = parent.errors[0]
            execution_details = None
            if len(parent.errors) > 1 and isinstance(parent.errors[1], str) and ("stdout" in parent.errors[1].lower() or "stderr" in parent.errors[1].lower()):
                execution_details = parent.errors[1]
            
            mutation_prompt = self.prompt_designer.design_bug_fix_prompt(
                program=parent,
                error_message=primary_error,
                execution_output=execution_details
            )
            logger.info(f"Attempting bug fix for parent {parent.id} using diff. Model: {chosen_model}. Error: {primary_error}")
            prompt_type = "bug_fix"
            # chosen_model remains LLM_PRIMARY_MODEL for bug fixing
        else:
            # Decide model for mutation based on parent fitness
            if parent_fitness >= settings.HIGH_FITNESS_THRESHOLD_FOR_PRIMARY_LLM:
                # chosen_model is already LLM_PRIMARY_MODEL
                logger.info(f"Parent {parent.id} has high fitness, using primary model {chosen_model} for mutation.")
            else:
                chosen_model = settings.LLM_SECONDARY_MODEL # Use secondary for lower-fitness mutations
                logger.info(f"Parent {parent.id} has lower fitness, using secondary model {chosen_model} for mutation.")

            feedback = {
                "errors": parent.errors,
                "metrics": parent_metrics,
                "scalar_fitness": parent_fitness,
            }
            feedback = {k: v for k, v in feedback.items() if v is not None}

            mutation_prompt = self.prompt_designer.design_mutation_prompt(program=parent, evaluation_feedback=feedback)
            logger.info(f"Attempting mutation for parent {parent.id} using diff. Model: {chosen_model}")
        
        generated_code = await self.code_generator.execute(
            prompt=mutation_prompt,
            model_name=chosen_model,
            temperature=0.75,
            output_format="diff",
            parent_code_for_diff=parent.code
        )

        if not generated_code.strip():
            logger.warning(f"Offspring generation for parent {parent.id} ({prompt_type}) resulted in empty code/diff. Skipping.")
            return None
        
        if generated_code == parent.code:
            logger.warning(f"Offspring generation for parent {parent.id} ({prompt_type}) using diff resulted in no change to the code. Skipping.")
            return None
        
        if "<<<<<<< SEARCH" in generated_code and "=======" in generated_code and ">>>>>>> REPLACE" in generated_code:
            logger.warning(f"Offspring generation for parent {parent.id} ({prompt_type}) seems to have returned raw diff. LLM or diff application may have failed. Skipping. Content:\n{generated_code[:500]}")
            return None
        
        if "# Error:" in generated_code[:100]:
            logger.warning(f"Failed to generate valid code for offspring of {parent.id} ({prompt_type}). LLM Output indicates error: {generated_code[:200]}")
            return None

        offspring = Program(
            id=child_id,
            code=generated_code,
            generation=generation_num,
            parent_id=parent.id,
            island_id=parent.island_id,  # Inherit island ID from parent
            status="unevaluated",
            task_id=self.task_definition.id,
            prompt_id=getattr(self.prompt_designer, "last_prompt_metadata", {}).get("template_id") if hasattr(self.prompt_designer, "last_prompt_metadata") else None,
        )
        logger.info(f"Successfully generated offspring {offspring.id} from parent {parent.id} ({prompt_type}).")
        return offspring

    async def execute(self) -> Any:
        return await self.manage_evolutionary_cycle()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    task_manager = TaskManagerAgent(task_definition=sample_task)

    sample_task = TaskDefinition(
        id="sum_list_task_001",
        description="Write a Python function called `solve(numbers)` that takes a list of integers `numbers` and returns their sum. The function should handle empty lists correctly by returning 0.",
        input_output_examples=[
            {"input": [1, 2, 3], "output": 6},
            {"input": [], "output": 0},
            {"input": [-1, 0, 1], "output": 0},
            {"input": [10, 20, 30, 40, 50], "output": 150}
        ],
        evaluation_criteria={"target_metric": "correctness", "goal": "maximize"},
        initial_code_prompt = "Please provide a Python function `solve(numbers)` that sums a list of integers. Handle empty lists by returning 0."
    )
    
    task_manager.num_generations = 3
    task_manager.population_size = 5
    task_manager.num_parents_to_select = 2

    async def run_task():
        try:
            best_programs = await task_manager.manage_evolutionary_cycle()
            if best_programs:
                print(f"\n*** Evolution Complete! Best program found: ***")
                print(f"ID: {best_programs[0].id}")
                print(f"Generation: {best_programs[0].generation}")
                print(f"Fitness: {best_programs[0].fitness_scores}")
                print(f"Code:\n{best_programs[0].code}")
            else:
                print("\n*** Evolution Complete! No suitable program was found. ***")
        except Exception as e:
            logger.error("An error occurred during the task management cycle.", exc_info=True)

    asyncio.run(run_task()) 