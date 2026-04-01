"""Load task definitions from data/logs.json (deterministic, no randomness)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True)
class TaskSpec:
    id: str
    difficulty: Literal["easy", "medium", "hard"]
    log_id: str
    log_text: str
    severity_hint: str
    expected_classification: str
    terminal_action: Literal["respond", "escalate"]
    allow_escalation: bool
    min_steps: int
    respond_keywords: tuple[str, ...] = ()
    escalate_keywords: tuple[str, ...] = ()
    requires_classify_before_escalate: bool = False


def _data_path() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "logs.json"


def load_tasks() -> list[TaskSpec]:
    raw = json.loads(_data_path().read_text(encoding="utf-8"))
    tasks: list[TaskSpec] = []
    for t in raw["tasks"]:
        tasks.append(_parse_task(t))
    return tasks


def _parse_task(t: dict[str, Any]) -> TaskSpec:
    rk = t.get("respond_keywords") or []
    ek = t.get("escalate_keywords") or []
    return TaskSpec(
        id=t["id"],
        difficulty=t["difficulty"],
        log_id=t["log_id"],
        log_text=t["log_text"],
        severity_hint=t["severity_hint"],
        expected_classification=t["expected_classification"],
        terminal_action=t["terminal_action"],
        allow_escalation=bool(t.get("allow_escalation", False)),
        min_steps=int(t.get("min_steps", 2)),
        respond_keywords=tuple(str(x).lower() for x in rk),
        escalate_keywords=tuple(str(x).lower() for x in ek),
        requires_classify_before_escalate=bool(t.get("requires_classify_before_escalate", False)),
    )


def get_task_by_id(task_id: str) -> TaskSpec | None:
    for t in load_tasks():
        if t.id == task_id:
            return t
    return None


def default_task_order() -> list[str]:
    return [t.id for t in load_tasks()]
