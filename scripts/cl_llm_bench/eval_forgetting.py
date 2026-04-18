"""Forgetting metric per task: accuracy_before(task) - accuracy_after(task).

Returns a dict mapping task_name to a non-negative float (clamped at 0
if the model actually improved on that task). Negative would indicate
post-sequence gain, which we count as zero forgetting.
"""

from __future__ import annotations


def forgetting_score(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for task, acc_before in before.items():
        if task not in after:
            continue
        drop = acc_before - after[task]
        scores[task] = max(drop, 0.0)
    return scores
