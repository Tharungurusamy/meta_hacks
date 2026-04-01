"""Pydantic wire types for CyberSec-OpenEnv."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field
from openenv.core.env_server.types import Action, Observation, State


ActionType = Literal["classify", "respond", "escalate", "ignore"]


class CyberSecAction(Action):
    """Agent action: classify, respond, escalate, or ignore."""

    action_type: ActionType = Field(..., description="High-level SOC action")
    content: str = Field(
        default="",
        description="Free-text: classification label, response text, or justification",
    )


class CyberSecObservation(Observation):
    """Observation; SOC fields are stored in metadata for JSON compatibility."""

    pass


class CyberSecState(State):
    """Extended state for UI and clients."""

    task_id: str | None = Field(default=None, description="Current task identifier")
    log_id: str | None = Field(default=None)
    done: bool = Field(default=False)
    last_feedback: str = Field(default="")
    last_grader_score: float = Field(default=0.0)
    history_len: int = Field(default=0)
    max_steps: int = Field(default=15)


class Reward(BaseModel):
    """Structured reward with dense components (mirrors observation.reward value)."""

    value: float = Field(description="Scalar reward passed to observation.reward")
    grader_score: float = Field(ge=0.0, le=1.0)
    step_penalty: float = Field(default=0.0)
    classification_component: float = Field(default=0.0)
    action_component: float = Field(default=0.0)
    keyword_component: float = Field(default=0.0)
    escalation_component: float = Field(default=0.0)
    penalty_wrong_class: float = Field(default=0.0)
    penalty_false_escalation: float = Field(default=0.0)
    penalty_ignore_threat: float = Field(default=0.0)
    penalty_too_many_steps: float = Field(default=0.0)
    summary: str = Field(default="")
