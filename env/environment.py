"""CyberSec environment: reset, step, state."""

from __future__ import annotations

import uuid
from typing import Any

from openenv.core.env_server.interfaces import Environment

from .graders import HistoryAction, grade_episode, terminal_success
from .models import CyberSecAction, CyberSecObservation, CyberSecState, Reward
from .reward import compute_reward
from .tasks import TaskSpec, get_task_by_id, load_tasks


def _action_to_hist(a: CyberSecAction) -> HistoryAction:
    return {"action_type": a.action_type, "content": a.content}


class CyberSecEnvironment(Environment[CyberSecAction, CyberSecObservation, CyberSecState]):
    """SOC log analysis environment."""

    SUPPORTS_CONCURRENT_SESSIONS = False
    MAX_STEPS = 15

    def __init__(self) -> None:
        super().__init__(transform=None, rubric=None)
        self._task: TaskSpec | None = None
        self._history: list[HistoryAction] = []
        self._episode_id: str | None = None
        self._done: bool = False
        self._last_reward_detail: Reward | None = None

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs: Any,
    ) -> CyberSecObservation:
        task_id: str | None = kwargs.get("task_id")
        tasks = load_tasks()
        if task_id:
            self._task = get_task_by_id(task_id)
            if self._task is None:
                raise ValueError(f"Unknown task_id: {task_id}")
        else:
            self._task = tasks[0]
        self._history = []
        self._episode_id = episode_id or str(uuid.uuid4())
        self._done = False
        self._last_reward_detail = None
        return self._observation(initial=True)

    def step(
        self,
        action: CyberSecAction,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> CyberSecObservation:
        if self._task is None or self._episode_id is None:
            raise RuntimeError("Environment not reset")
        if self._done:
            return self._observation(
                initial=False,
                extra_feedback="Episode already finished; call reset().",
            )

        self._history.append(_action_to_hist(action))
        step_count = len(self._history)
        r = compute_reward(self._task, self._history, step_count, self.MAX_STEPS)
        self._last_reward_detail = r

        if terminal_success(self._task, self._history):
            self._done = True
        elif step_count >= self.MAX_STEPS:
            self._done = True

        return self._observation(initial=False, reward_detail=r)

    @property
    def current_log_text(self) -> str | None:
        return self._task.log_text if self._task else None

    @property
    def action_history(self) -> list[HistoryAction]:
        return list(self._history)

    @property
    def state(self) -> CyberSecState:
        t = self._task
        return CyberSecState(
            episode_id=self._episode_id,
            step_count=len(self._history),
            task_id=t.id if t else None,
            log_id=t.log_id if t else None,
            done=self._done,
            last_feedback=self._last_reward_detail.summary if self._last_reward_detail else "",
            last_grader_score=(
                self._last_reward_detail.grader_score if self._last_reward_detail else 0.0
            ),
            history_len=len(self._history),
            max_steps=self.MAX_STEPS,
        )

    def close(self) -> None:
        pass

    def ui_snapshot(self) -> dict[str, Any]:
        """Safe snapshot for web UI (no private attribute access from server)."""
        t = self._task
        if t is None:
            return {}
        return {
            "log_id": t.log_id,
            "log_text": t.log_text,
            "severity_hint": t.severity_hint,
            "history": list(self._history),
            "task_id": t.id,
            "difficulty": t.difficulty,
        }

    def _observation(
        self,
        initial: bool,
        reward_detail: Reward | None = None,
        extra_feedback: str = "",
    ) -> CyberSecObservation:
        t = self._task
        assert t is not None
        md: dict[str, Any] = {
            "log_id": t.log_id,
            "log_text": t.log_text,
            "severity_hint": t.severity_hint,
            "history": list(self._history),
            "task_id": t.id,
            "difficulty": t.difficulty,
            "episode_id": self._episode_id,
            "feedback": "",
        }
        if reward_detail:
            md["feedback"] = reward_detail.summary + (
                (" " + extra_feedback) if extra_feedback else ""
            )
            md["reward_detail"] = reward_detail.model_dump()
            md["last_grader_score"] = reward_detail.grader_score
        elif initial:
            md["feedback"] = "Episode started. Analyze the log, classify, then respond or escalate."
        if extra_feedback and not reward_detail:
            md["feedback"] = extra_feedback

        g = grade_episode(t, self._history) if self._history else 0.0
        md["cumulative_grader_score"] = g

        rew: float | None = None
        if reward_detail is not None:
            rew = reward_detail.value
        elif initial:
            rew = 0.0

        return CyberSecObservation(
            done=self._done,
            reward=rew,
            metadata=md,
        )
