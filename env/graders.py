"""Deterministic graders: single score in [0.0, 1.0] per episode snapshot."""

from __future__ import annotations

from typing import TypedDict

from .tasks import TaskSpec


class HistoryAction(TypedDict):
    action_type: str
    content: str


def _norm(s: str) -> str:
    return s.strip().lower()


def _history_index_of_classify(history: list[HistoryAction], needle: str) -> int | None:
    n = needle.lower()
    for i, h in enumerate(history):
        if h["action_type"] == "classify" and n in _norm(h["content"]):
            return i
    return None


def _has_keywords(content: str, keywords: tuple[str, ...]) -> bool:
    c = _norm(content)
    return all(k in c for k in keywords)


def grade_episode(task: TaskSpec, history: list[HistoryAction]) -> float:
    """
    Return holistic grader score in [0.0, 1.0].
    Weights: classification 0.35, action sequence 0.25, keywords 0.25, escalation 0.15
    (escalation weight applies fully only when task requires it).
    """
    if not history:
        return 0.0

    cls_score = _score_classification(task, history)
    seq_score = _score_action_sequence(task, history)
    kw_score = _score_keywords(task, history)
    esc_score = _score_escalation_slot(task, history)

    w_cls, w_seq, w_kw, w_esc = 0.35, 0.25, 0.25, 0.15
    if task.terminal_action == "respond":
        w_esc = 0.0
        total_w = w_cls + w_seq + w_kw
        return (w_cls * cls_score + w_seq * seq_score + w_kw * kw_score) / total_w
    total_w = w_cls + w_seq + w_kw + w_esc
    return (
        w_cls * cls_score + w_seq * seq_score + w_kw * kw_score + w_esc * esc_score
    ) / total_w


def _score_classification(task: TaskSpec, history: list[HistoryAction]) -> float:
    exp = task.expected_classification.lower()
    for h in history:
        if h["action_type"] != "classify":
            continue
        c = _norm(h["content"])
        if exp in c or exp.replace("_", " ") in c:
            return 1.0
        if exp == "data_exfiltration" and (
            "exfil" in c or "data exfil" in c or "exfiltration" in c or "data_exfiltration" in c
        ):
            return 1.0
    return 0.0


def _score_action_sequence(task: TaskSpec, history: list[HistoryAction]) -> float:
    """Reward correct ordering: classify before terminal action when required."""
    if task.requires_classify_before_escalate:
        ci = _history_index_of_classify(history, task.expected_classification)
        if ci is None:
            return 0.0
        for h in history[ci + 1 :]:
            if h["action_type"] == task.terminal_action:
                return 1.0
        return 0.3 if any(h["action_type"] == "classify" for h in history) else 0.0

    # easy/medium: classify then respond
    if task.terminal_action == "respond":
        ci = _history_index_of_classify(history, task.expected_classification)
        if ci is None:
            return 0.2
        for h in history[ci + 1 :]:
            if h["action_type"] == "respond":
                return 1.0
        return 0.4
    return 0.5


def _score_keywords(task: TaskSpec, history: list[HistoryAction]) -> float:
    if task.terminal_action == "respond" and task.respond_keywords:
        for h in history:
            if h["action_type"] == "respond" and _has_keywords(
                h["content"], task.respond_keywords
            ):
                return 1.0
        return 0.0
    if task.terminal_action == "escalate" and task.escalate_keywords:
        for h in history:
            if h["action_type"] == "escalate" and _has_keywords(
                h["content"], task.escalate_keywords
            ):
                return 1.0
        return 0.0
    return 0.5


def _score_escalation_slot(task: TaskSpec, history: list[HistoryAction]) -> float:
    if task.terminal_action != "escalate":
        return 1.0
    for h in history:
        if h["action_type"] == "escalate":
            return 1.0 if task.allow_escalation else 0.0
    return 0.0


def terminal_success(task: TaskSpec, history: list[HistoryAction]) -> bool:
    """Episode ends successfully when the last action completes the playbook."""
    if not history:
        return False
    last = history[-1]
    if last["action_type"] == "ignore":
        return False
    if grade_episode(task, history) < 0.88:
        return False
    if task.terminal_action == "respond":
        return last["action_type"] == "respond" and _has_keywords(
            last["content"], task.respond_keywords
        )
    if task.terminal_action == "escalate":
        return last["action_type"] == "escalate" and _has_keywords(
            last["content"], task.escalate_keywords
        )
    return False
