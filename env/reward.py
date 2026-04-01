"""Dense reward shaping with step penalty and behavioral penalties."""

from __future__ import annotations

from .graders import HistoryAction, grade_episode
from .models import Reward
from .tasks import TaskSpec


def compute_reward(
    task: TaskSpec,
    history: list[HistoryAction],
    step_count: int,
    max_steps: int,
) -> Reward:
    """
    reward ~= grader_score - step_penalty - behavioral penalties; clipped to [0, 1].
    Step cost: 0.05 per step (after step 1).
    """
    g = grade_episode(task, history)
    step_penalty = 0.05 * max(0, step_count - 1)

    p_wrong = 0.0
    p_fe = 0.0
    p_ig = 0.0
    p_steps = 0.0

    for h in history:
        if h["action_type"] != "classify":
            continue
        c = h["content"].lower()
        exp = task.expected_classification.lower()
        ok = exp in c or exp.replace("_", " ") in c
        if exp == "data_exfiltration":
            ok = ok or "exfil" in c or "exfiltration" in c or "data_exfiltration" in c
        if not ok:
            p_wrong = max(p_wrong, 0.15)

    for h in history:
        if h["action_type"] == "escalate" and not task.allow_escalation:
            p_fe = 0.25
        if h["action_type"] == "ignore":
            p_ig = 0.35

    if step_count >= max_steps:
        p_steps = 0.1

    raw = g - step_penalty - p_wrong - p_fe - p_ig - p_steps
    value = max(0.0, min(1.0, raw))

    return Reward(
        value=value,
        grader_score=g,
        step_penalty=step_penalty,
        classification_component=g * 0.35,
        action_component=g * 0.25,
        keyword_component=g * 0.25,
        escalation_component=g * 0.15,
        penalty_wrong_class=p_wrong,
        penalty_false_escalation=p_fe,
        penalty_ignore_threat=p_ig,
        penalty_too_many_steps=p_steps,
        summary=_summarize(g, value, step_penalty, p_wrong, p_fe, p_ig),
    )


def _summarize(
    g: float,
    v: float,
    sp: float,
    pw: float,
    pfe: float,
    pig: float,
) -> str:
    parts = [
        f"grader={g:.3f}",
        f"shaped={v:.3f}",
        f"step_cost={sp:.3f}",
    ]
    if pw > 0:
        parts.append(f"wrong_class_penalty={pw:.3f}")
    if pfe > 0:
        parts.append(f"false_escalation={pfe:.3f}")
    if pig > 0:
        parts.append(f"ignore_threat={pig:.3f}")
    return "; ".join(parts)
