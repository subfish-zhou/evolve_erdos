import math
import time
from collections import deque
from itertools import combinations
from typing import Any, Dict, Iterable, List, Sequence, Tuple

INVALID_PENALTY = -1e4


def _to_adjlist(graph: Any) -> List[List[int]]:
    if isinstance(graph, dict):
        n = max(graph.keys()) + 1 if graph else 0
        adj = [[] for _ in range(n)]
        for node, nbrs in graph.items():
            adj[node] = [int(v) for v in nbrs]
    elif isinstance(graph, list):
        adj = [list(map(int, nbrs)) for nbrs in graph]
        n = len(adj)
    else:
        raise TypeError("Candidate must return adjacency list (list[list[int]] or dict[int, list[int]])")

    if n == 0:
        raise ValueError("Empty graph is not allowed.")

    for i in range(n):
        seen = set()
        for j in adj[i]:
            if not (0 <= j < n):
                raise ValueError(f"Neighbor index out of range: {i}->{j}")
            if j == i:
                raise ValueError("Self-loop detected.")
            if j in seen:
                raise ValueError("Duplicate neighbor detected.")
            seen.add(j)
    for i in range(n):
        for j in adj[i]:
            if i not in adj[j]:
                raise ValueError(f"Asymmetric adjacency: {i} not in N({j})")
    return adj


def _is_forest(adj: List[List[int]]) -> bool:
    n = len(adj)
    edge_count = sum(len(v) for v in adj)
    if edge_count % 2:
        return False
    m = edge_count // 2
    seen = [False] * n
    comps = 0
    for start in range(n):
        if seen[start]:
            continue
        comps += 1
        queue = deque([start])
        seen[start] = True
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    queue.append(v)
    return m == n - comps


def _connected_components(adj: List[List[int]]) -> List[List[int]]:
    n = len(adj)
    seen = [False] * n
    comps = []
    for start in range(n):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        comp = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        comps.append(comp)
    return comps


def _poly_add(a: Sequence[int], b: Sequence[int]) -> List[int]:
    n = max(len(a), len(b))
    out = [0] * n
    for i, v in enumerate(a):
        out[i] += v
    for i, v in enumerate(b):
        out[i] += v
    return out


def _poly_mul(a: Sequence[int], b: Sequence[int]) -> List[int]:
    out = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x == 0:
            continue
        for j, y in enumerate(b):
            if y != 0:
                out[i + j] += x * y
    return out


def _indep_poly_tree(adj: List[List[int]], root: int) -> List[int]:
    n = len(adj)
    parent = [-2] * n
    order = []
    stack = [root]
    parent[root] = root
    while stack:
        u = stack.pop()
        order.append(u)
        for v in adj[u]:
            if parent[v] == -2:
                parent[v] = u
                stack.append(v)
    A = {u: [1] for u in order}
    B = {u: [0, 1] for u in order}
    for u in reversed(order):
        for v in adj[u]:
            if parent[v] == u:
                A[u] = _poly_mul(A[u], _poly_add(A[v], B[v]))
                B[u] = _poly_mul(B[u], A[v])
    root_node = order[0]
    return _poly_add(A[root_node], B[root_node])


def _indep_seq_forest(adj: List[List[int]]) -> List[int]:
    comps = _connected_components(adj)
    total = [1]
    for comp in comps:
        poly = _indep_poly_tree(adj, comp[0])
        total = _poly_mul(total, poly)
    return total


def _linf_distance_to_unimodal(seq: Sequence[int]) -> float:
    n = len(seq)
    lo, hi = 0.0, max(1.0, max(seq) - min(seq))

    def feasible(eps: float) -> bool:
        low = [x - eps for x in seq]
        high = [x + eps for x in seq]
        for k in range(n):
            peak = high[k]
            prev = peak
            ok = True
            for i in range(k + 1, n):
                if low[i] > prev:
                    ok = False
                    break
                prev = min(prev, high[i])
            if not ok:
                continue
            nxt = peak
            for i in range(k - 1, -1, -1):
                bound = min(nxt, high[i])
                if low[i] > bound:
                    ok = False
                    break
                nxt = bound
            if ok:
                return True
        return False

    iters = 0
    while not feasible(hi) and iters < 40:
        hi *= 2
        iters += 1
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if feasible(mid):
            hi = mid
        else:
            lo = mid
    return hi


def _score_candidate(candidate: Any, max_nodes: int) -> float:
    adj = _to_adjlist(candidate)
    if len(adj) > max_nodes:
        raise ValueError(f"Candidate graph too large ({len(adj)} nodes).")
    if not _is_forest(adj):
        raise ValueError("Candidate must be a forest.")
    seq = _indep_seq_forest(adj)
    while len(seq) < len(adj) + 1:
        seq.append(0)
    return float(_linf_distance_to_unimodal(seq))


def _extract_candidate_payload(result: Any) -> Tuple[Any, float, int]:
    if isinstance(result, dict):
        candidate = (
            result.get("candidate")
            or result.get("graph")
            or result.get("forest")
            or result.get("adjacency")
        )
        violation = result.get("best_violation")
        samples = int(result.get("samples_tried", 0))
        return candidate, violation if isinstance(violation, (int, float)) else None, samples
    if isinstance(result, (list, tuple)):
        return result, None, 0
    return None, None, 0


def evaluate_candidate(program_module, task_definition) -> Dict[str, float]:
    function_name = task_definition.function_name_to_evolve or "search_erdos_993"
    search_fn = getattr(program_module, function_name, None)
    if not callable(search_fn):
        raise AttributeError(f"Program must define callable `{function_name}`")

    config = task_definition.metrics_config or task_definition.evaluation_criteria or {}
    seeds = list(config.get("seeds", range(5)))
    budget = int(config.get("budget", 512))
    max_nodes = int(config.get("max_nodes", 28))

    best_violation = float("-inf")
    total_samples = 0
    valid_candidates = 0
    attempts = 0
    failures = 0

    start = time.perf_counter()
    for seed in seeds:
        attempts += 1
        try:
            try:
                result = search_fn(seed=seed, budget=budget)
            except TypeError:
                try:
                    result = search_fn(seed, budget)
                except TypeError:
                    result = search_fn()
        except Exception:
            failures += 1
            continue

        candidate, reported_violation, samples = _extract_candidate_payload(result)
        total_samples += samples if samples > 0 else budget

        violation = None
        if isinstance(reported_violation, (int, float)) and math.isfinite(reported_violation):
            violation = float(reported_violation)
        elif candidate is not None:
            try:
                violation = _score_candidate(candidate, max_nodes)
                valid_candidates += 1
            except Exception:
                failures += 1
                violation = None

        if violation is not None:
            best_violation = max(best_violation, violation)
        else:
            failures += 1

    elapsed = time.perf_counter() - start
    if best_violation == float("-inf"):
        best_violation = INVALID_PENALTY

    success = 1.0 if best_violation > 0 else 0.0
    valid_ratio = valid_candidates / max(1, attempts)
    failure_ratio = failures / max(1, attempts)

    metrics = {
        "score": best_violation,
        "success": success,
        "samples": float(total_samples),
        "valid_ratio": valid_ratio,
        "failure_ratio": failure_ratio,
        "elapsed": elapsed,
        "runtime_penalty": -math.log1p(elapsed),
    }
    return metrics

