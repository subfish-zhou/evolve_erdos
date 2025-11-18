"""
Main entry point for the AlphaEvolve Pro application.
Orchestrates the different agents and manages the evolutionary loop.
"""
import asyncio
import logging
import sys
import os
import yaml
import argparse
from typing import Optional
                                               
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from task_manager.agent import TaskManagerAgent
from core.interfaces import TaskDefinition
from config import settings

                   
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(settings.LOG_FILE, mode="a")
    ]
)
logger = logging.getLogger(__name__)

def load_task_from_yaml(yaml_path: str) -> Optional[TaskDefinition]:
    """Load task configuration from a YAML file."""
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError("YAML root must be a mapping.")

        task_kwargs = {
            "id": data.get("task_id"),
            "description": data.get("task_description"),
            "function_name_to_evolve": data.get("function_name") or data.get("function_name_to_evolve"),
            "allowed_imports": data.get("allowed_imports"),
            "input_output_examples": data.get("input_output_examples"),
            "tests": data.get("tests"),
            "evaluation_criteria": data.get("evaluation_criteria"),
            "expert_knowledge": data.get("expert_knowledge"),
            "evaluation_mode": data.get("evaluation_mode", "tests"),
            "metrics_eval_module": data.get("metrics_eval_module"),
            "metrics_primary_key": data.get("metrics_primary_key"),
            "metrics_scalarization": data.get("metrics_scalarization"),
            "metrics_success_key": data.get("metrics_success_key", "success"),
            "metrics_config": data.get("metrics_config"),
        }

        missing = [k for k in ("id", "description", "function_name_to_evolve") if not task_kwargs.get(k)]
        if missing:
            raise ValueError(f"Missing required task fields: {', '.join(missing)}")

        return TaskDefinition(**task_kwargs)
    except Exception as e:
        logger.error(f"Error loading task from YAML: {e}")
        return None

async def main():
    parser = argparse.ArgumentParser(description="Run OpenAlpha_Evolve with a specified YAML configuration file.")
    parser.add_argument("yaml_path", type=str, help="Path to the YAML configuration file")
    args = parser.parse_args()
    yaml_path = args.yaml_path

    logger.info("Starting OpenAlpha_Evolve autonomous algorithmic evolution")
    logger.info(f"Configuration: Population Size={settings.POPULATION_SIZE}, Generations={settings.GENERATIONS}")

    task = load_task_from_yaml(yaml_path)
    
    if not task:
        logger.error("Failed to load task configuration from YAML file. Exiting.")
        return

    task_manager = TaskManagerAgent(
        task_definition=task
    )

    best_programs = await task_manager.execute()

    if best_programs:
        logger.info(f"Evolutionary process completed. Best program(s) found: {len(best_programs)}")
        for i, program in enumerate(best_programs):
            logger.info(f"Final Best Program {i+1} ID: {program.id}")
            logger.info(f"Final Best Program {i+1} Fitness: {program.fitness_scores}")
            logger.info(f"Final Best Program {i+1} Code:\n{program.code}")
    else:
        logger.info("Evolutionary process completed, but no suitable programs were found.")

    logger.info("OpenAlpha_Evolve run finished.")

if __name__ == "__main__":
    asyncio.run(main())
