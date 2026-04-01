"""HTTP client for the CyberSec REST API (GET /reset, POST /step, GET /state)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from .env.models import CyberSecAction, CyberSecObservation, CyberSecState


@dataclass
class StepResult:
    observation: CyberSecObservation
    reward: float | None
    done: bool


class CyberSecHTTPClient:
    """
    Synchronous REST client (OpenEnv's EnvClient uses WebSocket; this matches our HTTP API).
    """

    def __init__(self, base_url: str, timeout_s: float = 60.0) -> None:
        self._base = base_url.rstrip("/")
        self._client = httpx.Client(timeout=timeout_s)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> CyberSecHTTPClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def reset(self, task_id: str | None = None, **kwargs: Any) -> StepResult:
        params: dict[str, str] = {}
        if task_id is not None:
            params["task_id"] = task_id
        r = self._client.get(f"{self._base}/reset", params=params)
        r.raise_for_status()
        payload = r.json()
        return self._parse_step_payload(payload)

    def step(self, action: CyberSecAction) -> StepResult:
        r = self._client.post(
            f"{self._base}/step",
            json=action.model_dump(),
        )
        r.raise_for_status()
        payload = r.json()
        return self._parse_step_payload(payload)

    def get_state(self) -> CyberSecState:
        r = self._client.get(f"{self._base}/state")
        r.raise_for_status()
        data = r.json()
        return CyberSecState(**data)

    def _parse_step_payload(self, payload: dict[str, Any]) -> StepResult:
        obs = CyberSecObservation(**payload["observation"])
        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=bool(payload.get("done", False)),
        )
