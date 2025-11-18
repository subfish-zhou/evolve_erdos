import math
from typing import Dict, Optional

from core.interfaces import Program, TaskDefinition

NUMERIC_TYPES = (int, float)


def _numeric_metrics(metrics: Optional[Dict[str, float]]) -> Dict[str, float]:
    numeric: Dict[str, float] = {}
    if not metrics:
        return numeric
    for key, value in metrics.items():
        if isinstance(value, NUMERIC_TYPES):
            float_val = float(value)
            if math.isfinite(float_val):
                numeric[key] = float_val
    return numeric


def scalarize_metrics(metrics: Optional[Dict[str, float]], task: Optional[TaskDefinition]) -> float:
    numeric = _numeric_metrics(metrics)
    if not numeric:
        return 0.0

    if task and task.metrics_primary_key and task.metrics_primary_key in numeric:
        return float(numeric[task.metrics_primary_key])

    if task and task.metrics_scalarization:
        total = 0.0
        for metric_name, weight in task.metrics_scalarization.items():
            if not isinstance(weight, NUMERIC_TYPES):
                continue
            total += float(weight) * numeric.get(metric_name, 0.0)
        return total

    return float(sum(numeric.values()))


def derive_success_metric(metrics: Optional[Dict[str, float]], task: Optional[TaskDefinition]) -> float:
    numeric = _numeric_metrics(metrics)
    if not numeric:
        return 0.0

    if task and task.metrics_success_key:
        val = numeric.get(task.metrics_success_key)
        if val is not None:
            return float(val)

    for candidate_key in ("success", "is_success", "passed_tests"):
        if candidate_key in numeric:
            return float(numeric[candidate_key])
    return 0.0


def extract_runtime_metric(metrics: Optional[Dict[str, float]]) -> float:
    numeric = _numeric_metrics(metrics)
    for key in ("elapsed", "runtime_ms", "runtime", "avg_runtime_ms"):
        if key in numeric:
            return float(numeric[key])
    return float("inf")


def ensure_program_fitness(program: Program, task: Optional[TaskDefinition]) -> float:
    if program.fitness is not None:
        return program.fitness
    program.fitness = scalarize_metrics(program.metrics or program.fitness_scores, task)
    return program.fitness


def selection_sort_key(program: Program, task: Optional[TaskDefinition]):
    metrics = program.metrics or program.fitness_scores
    success = derive_success_metric(metrics, task)
    scalar = ensure_program_fitness(program, task)
    runtime = extract_runtime_metric(metrics)
    return (
        success,
        scalar,
        -runtime if math.isfinite(runtime) else float("-inf"),
        -program.generation,
        -program.created_at,
    )


