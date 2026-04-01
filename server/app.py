"""FastAPI server: health, GET /reset, POST /step, GET /state, web UI."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
try:
    from env.environment import CyberSecEnvironment
    from env.models import CyberSecAction, CyberSecObservation, CyberSecState
except ImportError:
    from cybersec_openenv.env.environment import CyberSecEnvironment
    from cybersec_openenv.env.models import CyberSecAction, CyberSecObservation, CyberSecState

STATIC = Path(__file__).resolve().parent / "static"
TEMPLATES = Path(__file__).resolve().parent / "templates"

app = FastAPI(title="CyberSec-OpenEnv", version="1.0.0")
templates = Jinja2Templates(directory=str(TEMPLATES))

_env: CyberSecEnvironment | None = None


def get_env() -> CyberSecEnvironment:
    global _env
    if _env is None:
        _env = CyberSecEnvironment()
    return _env


def _serialize_observation(obs: CyberSecObservation) -> dict[str, Any]:
    return obs.model_dump(mode="json")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.get("/reset")
def reset(
    task_id: str | None = Query(default=None),
    seed: int | None = Query(default=None),
    episode_id: str | None = Query(default=None),
) -> dict[str, Any]:
    env = get_env()
    try:
        obs = env.reset(seed=seed, episode_id=episode_id, task_id=task_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    st = env.state
    return {
        "observation": _serialize_observation(obs),
        "reward": obs.reward,
        "done": obs.done,
        "state": st.model_dump(mode="json"),
    }


@app.post("/reset")
def reset_post(body: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Compatibility endpoint for validators expecting POST /reset.
    Accepts optional JSON body: { "task_id": "...", "seed": 123, "episode_id": "..." }.
    """
    env = get_env()
    payload = body or {}
    task_id = payload.get("task_id")
    seed = payload.get("seed")
    episode_id = payload.get("episode_id")
    try:
        obs = env.reset(seed=seed, episode_id=episode_id, task_id=task_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    st = env.state
    return {
        "observation": _serialize_observation(obs),
        "reward": obs.reward,
        "done": obs.done,
        "state": st.model_dump(mode="json"),
    }


@app.post("/step")
def step(action: CyberSecAction) -> dict[str, Any]:
    env = get_env()
    try:
        obs = env.step(action)
    except RuntimeError as e:
        # Common client mistake: calling /step before /reset.
        # Return a clear 4xx response instead of a generic 500.
        raise HTTPException(
            status_code=400,
            detail="Environment not reset. Call GET /reset before POST /step.",
        ) from e
    st = env.state
    return {
        "observation": _serialize_observation(obs),
        "reward": obs.reward,
        "done": obs.done,
        "state": st.model_dump(mode="json"),
    }


@app.get("/state")
def state() -> dict[str, Any]:
    env = get_env()
    st: CyberSecState = env.state
    return st.model_dump(mode="json")


@app.get("/", response_class=HTMLResponse)
def ui_root(request: Request) -> Any:
    return ui_page(request)


@app.get("/web", response_class=HTMLResponse)
def ui_page(request: Request) -> Any:
    env = get_env()
    st = env.state
    snap = env.ui_snapshot()
    if snap:
        snap["feedback"] = st.last_feedback
        snap["last_grader_score"] = st.last_grader_score
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "state": st.model_dump(),
            "observation": snap or None,
        },
    )


if STATIC.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


def main() -> None:
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
