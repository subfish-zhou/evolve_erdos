#!/usr/bin/env python3
"""
基于 `azure_api.py` 和 `examples/erdos_993.yaml` 的快速连通性测试脚本。

运行方式：
    python scripts/test_azure_api.py
脚本会：
    1. 读取 `erdos_993.yaml` 中的任务描述；
    2. 调用 Azure OpenAI（与主程序相同的配置）；
    3. 输出模型返回的简短总结，以确认 API 仍可正常工作。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import yaml

def _ensure_utf8(text: str) -> str:
    if isinstance(text, bytes):
        return text.decode("utf-8")
    return str(text).encode("utf-8").decode("utf-8")

USER_PROMPT = """You are a systems-minded infrastructure builder who layers search, caching, and statistics to keep the heuristic maintainable.\\nEncapsulate configuration knobs, resource budgets, and logging utilities so that evaluation pipelines can reuse the components safely.\\n\\nEvaluation mode: continuous multi-metric scoring.\n- The evaluator module `examples.erdos_993_eval` calls `evaluate_candidate(module, task_definition)` to compute metrics.\n- Your program must behave like a search heuristic and emit rich telemetry (e.g., best_violation, samples_tried, success flags) so selection can reason about progress.\n- Primary optimization target: `score`.\n- Scalarization weights: score:1.0, runtime_penalty:0.2, valid_ratio:0.1\n- Evaluation configuration:\n{\n  \"interface\": \"Implement `search_erdos_993(seed: int, budget: int) -> dict`.\\nExpectations for each invocation:\\n  1. Use the provided `seed` to initialize any RNGs so the evaluator can reproduce behavior.\\n  2. Treat `budget` as the maximum number of internal iterations / samples you may draw; you may use\\n     it to control loop bounds, simulated annealing steps, backtracking depth, etc.\\n  3. Return a dictionary with at least the following telemetry fields:\\n       {\\n         \\\"candidate\\\": <adjacency list or dict>,\\n         \\\"best_violation\\\": <float>,\\n         \\\"samples_tried\\\": <int>,\\n         \\\"notes\\\": <optional string for debugging>\\n       }\\n     - `candidate` must describe an undirected forest. The evaluator enforces symmetry and absence\\n       of cycles; invalid graphs will be rejected.\\n     - `best_violation` is the highest L_inf deviation you observed during this run; it should be\\n       positive when a real counterexample is found, but can also be a non-positive “progress score”\\n       when the search only approximates unimodality.\\n     - `samples_tried` counts how many candidate forests you evaluated internally; this helps the\\n       evaluator normalize runtime-efficiency metrics.\\n     - `notes` is optional but encouraged (e.g., strategy name, last tweak applied, etc.).\\n  4. If you cannot produce a valid `candidate`, set it to `None` but still report `best_violation`\\n     so that evolution has a continuous signal to optimize.\\n\"\n}\n\nTask Description: Evolve a *search heuristic* (not a single-shot solver) that continuously generates undirected forests\nin pursuit of a counterexample to the Erdős–Moser conjecture: a forest whose independence-set counting\nsequence is **non-unimodal**. Each candidate forest is represented as an adjacency list (list of lists\nor dict of lists) describing an undirected, loop-free structure. The evaluator calls the heuristic\nrepeatedly with different random seeds and time budgets, and for every run it measures:\n  * the L_infinity distance from the candidate’s independence-sequence to the nearest unimodal sequence;\n  * runtime / telemetry information to estimate exploration efficiency;\n  * whether a genuine counterexample (score > 0) was discovered.\nThe heuristic must therefore act like a multi-step search algorithm: track the best violation value\nseen so far, explore a diverse set of forests, and surface enough metadata for downstream selection to\nmake informed decisions. Even if no true counterexample is found, the heuristic should maximize the\n“unimodality violation” metric so that the evolutionary loop receives a smooth fitness landscape.\n\\n\\nFunction to Implement: `search_erdos_993`\\n\\nInput/Output Examples:\nThis task relies on metrics-based evaluation; provide a search heuristic that reports telemetry instead of deterministic I/O pairs.\n\nEvaluation Criteria: {'interface': 'Implement `search_erdos_993(seed: int, budget: int) -> dict`.\\nExpectations for each invocation:\\n  1. Use the provided `seed` to initialize any RNGs so the evaluator can reproduce behavior.\\n  2. Treat `budget` as the maximum number of internal iterations / samples you may draw; you may use\\n     it to control loop bounds, simulated annealing steps, backtracking depth, etc.\\n  3. Return a dictionary with at least the following telemetry fields:\\n       {\\n         \"candidate\": <adjacency list or dict>,\\n         \"best_violation\": <float>,\\n         \"samples_tried\": <int>,\\n         \"notes\": <optional string for debugging>\\n       }\\n     - `candidate` must describe an undirected forest. The evaluator enforces symmetry and absence\\n       of cycles; invalid graphs will be rejected.\\n     - `best_violation` is the highest L_inf deviation you observed during this run; it should be\\n       positive when a real counterexample is found, but can also be a non-positive “progress score”\\n       when the search only approximates unimodality.\\n     - `samples_tried` counts how many candidate forests you evaluated internally; this helps the\\n       evaluator normalize runtime-efficiency metrics.\\n     - `notes` is optional but encouraged (e.g., strategy name, last tweak applied, etc.).\\n  4. If you cannot produce a valid `candidate`, set it to `None` but still report `best_violation`\\n     so that evolution has a continuous signal to optimize.\\n'}\n\nAllowed Standard Library Imports: math, random, itertools, collections. Do not use any other external libraries or packages.\n\nYour Response Format:\nPlease provide *only* the complete Python code for the function `search_erdos_993`. The code should be self-contained or rely only on the allowed imports. Do not include any surrounding text, explanations, comments outside the function, or markdown code fences (like ```python or ```)."""


REPO_ROOT = Path(__file__).resolve().parents[1]

# 确保可以作为独立脚本运行
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from azure_api import get_azure_client
TASK_CONFIG_PATH = REPO_ROOT / "examples" / "erdos_993.yaml"


def load_task_prompt() -> str:
    """读取 YAML 配置并构造一个简短的提示。"""
    if not TASK_CONFIG_PATH.exists():
        raise FileNotFoundError(f"找不到示例配置文件: {TASK_CONFIG_PATH}")

    with TASK_CONFIG_PATH.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    task_id = config.get("task_id", "unknown_task")
    description = config.get("task_description", "").strip()
    function_name = config.get("function_name", "search_erdos_993")
    metrics_info = config.get("metrics_config", {})

    prompt_payload = {
        "task_id": task_id,
        "function_name": function_name,
        "metrics_config": metrics_info,
        "description": description,
    }

    return (
        "请阅读以下 JSON，确认你已成功接收到任务信息。"
        "务必输出以“总结：”开头的一句中文总结：\n"
        f"{json.dumps(prompt_payload, ensure_ascii=False, indent=2)}"
    )


def call_azure(prompt: str):
    """调用 Azure OpenAI，返回模型输出。"""
    client = get_azure_client()
    prompt = _ensure_utf8(prompt)
    response = client.chat_completion_sync(
        messages=[
            {"role": "user", "content": prompt},
        ],
        max_completion_tokens=1024,
    )

    if not response or not getattr(response, "choices", None):
        raise RuntimeError("Azure OpenAI 没有返回任何内容。")

    content = response.choices[0].message.content or ""
    return content.strip(), response


def main() -> int:
    try:
        prompt = _ensure_utf8(USER_PROMPT)
        start = time.perf_counter()
        answer, raw_response = call_azure(prompt)
        elapsed = time.perf_counter() - start
        print("\n=== Azure OpenAI 测试响应 ===")
        if answer:
            print(answer)
        else:
            print("[模型未返回任何文本]")
            debug_payload = (
                raw_response.model_dump_json(indent=2)
                if hasattr(raw_response, "model_dump_json")
                else repr(raw_response)
            )
            print("\n--- 原始响应 ---")
            print(debug_payload)
        print(f"\n运行时间：{elapsed:.2f} 秒")
        print("（若上方输出为合理文本，则说明 API 可用。）")
        return 0
    except Exception as exc:
        print("测试失败：", exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

