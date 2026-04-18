# tests/scripts/cl_llm_bench/test_eval_forgetting.py
from scripts.cl_llm_bench.eval_forgetting import forgetting_score

TOL = 1e-6
EPSILON = 1e-9


def test_forgetting_score_zero_when_unchanged() -> None:
    before = {"task0": 0.80, "task1": 0.75, "task2": 0.70}
    after = before.copy()
    scores = forgetting_score(before, after)
    assert all(abs(v) < EPSILON for v in scores.values())


def test_forgetting_score_positive_on_drop() -> None:
    before = {"task0": 0.80, "task1": 0.75, "task2": 0.70}
    after = {"task0": 0.20, "task1": 0.30, "task2": 0.70}
    scores = forgetting_score(before, after)
    assert abs(scores["task0"] - 0.60) < TOL
    assert abs(scores["task1"] - 0.45) < TOL
    assert abs(scores["task2"]) < EPSILON


def test_forgetting_score_clamps_improvement_to_zero() -> None:
    before = {"task0": 0.50}
    after = {"task0": 0.90}
    scores = forgetting_score(before, after)
    assert scores["task0"] == 0.0
