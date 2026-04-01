#!/usr/bin/env python3
"""
Run all tasks with an OpenAI-compatible client (temperature=0) or scripted policy.

Environment variables:
  API_BASE_URL   Base URL for the API (default: http://127.0.0.1:8000)
  MODEL_NAME     Model id for chat completions
  HF_TOKEN       API key (Hugging Face Inference or OpenAI-compatible)

If HF_TOKEN and OPENAI_API_KEY are both unset, uses a deterministic scripted policy.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.tasks import default_task_order, load_tasks
from env.environment import CyberSecEnvironment
from env.models import CyberSecAction


def _scripted_steps(task_id: str) -> list[CyberSecAction]:
    """Optimal reference actions for reproducible baseline scoring."""
    if task_id == "task_01_phishing":
        return [
            CyberSecAction(action_type="classify", content="phishing"),
            CyberSecAction(
                action_type="respond",
                content="Warn the user about a suspicious payroll link; do not enter credentials.",
            ),
        ]
    if task_id == "task_02_brute_force":
        return [
            CyberSecAction(action_type="classify", content="brute_force"),
            CyberSecAction(
                action_type="respond",
                content="Block source IP 198.51.100.22 at the firewall and alert the SOC team.",
            ),
        ]
    if task_id == "task_03_exfil_chain":
        return [
            CyberSecAction(action_type="classify", content="data_exfiltration"),
            CyberSecAction(
                action_type="escalate",
                content="Escalate to incident response for investigation of exfiltration to external object storage.",
            ),
        ]
    raise ValueError(f"No scripted policy for {task_id}")


def _llm_action(
    client: Any,
    model: str,
    task_log: str,
    history: list[dict[str, str]],
) -> CyberSecAction:
    system = (
        "You are a SOC analyst. Output a single JSON object with keys "
        '"action_type" (one of: classify, respond, escalate, ignore) and '
        '"content" (string). Use classify first to label the threat, then '
        "respond or escalate as appropriate. Be concise."
    )
    hist = json.dumps(history, indent=0)
    user = f"Security log:\n{task_log}\n\nActions so far:\n{hist}\n\nNext action JSON only:"
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    text = (resp.choices[0].message.content or "").strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    data = json.loads(text)
    return CyberSecAction(action_type=data["action_type"], content=data.get("content", ""))


def run_episode(
    env: CyberSecEnvironment,
    task_id: str,
    client: Any,
    model: str | None,
    max_steps: int = 15,
    scripted: bool = False,
) -> float:
    env.reset(task_id=task_id)
    tasks = {t.id: t for t in load_tasks()}
    task = tasks[task_id]

    if scripted:
        for a in _scripted_steps(task_id):
            obs = env.step(a)
            if obs.done:
                break
    else:
        assert client is not None and model
        log_text = env.current_log_text or ""
        history: list[dict[str, str]] = []
        for _ in range(max_steps):
            action = _llm_action(client, model, log_text, history)
            obs = env.step(action)
            history.append({"action_type": action.action_type, "content": action.content})
            if obs.done:
                break

    from env.graders import grade_episode

    g = grade_episode(task, env.action_history)
    return float(g)


def main() -> None:
    api_base = os.environ.get("API_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.environ.get("MODEL_NAME", "gpt-4o-mini")
    token = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY", "")
    scripted = not token

    env = CyberSecEnvironment()
    order = default_task_order()
    scores: list[float] = []

    client: Any = None
    if not scripted:
        from openai import OpenAI

        client = OpenAI(
            base_url=api_base,
            api_key=token,
        )

    for tid in order:
        # Local environment loop (no HTTP) for reproducibility and CI
        s = run_episode(env, tid, client, model if not scripted else None, scripted=scripted)
        scores.append(s)
        print(f"task={tid} grader_score={s:.4f}")

    avg = sum(scores) / len(scores) if scores else 0.0
    print(f"average_grader_score={avg:.4f}")
    if scripted:
        print("(scripted policy; set HF_TOKEN and API_BASE_URL for LLM runs)")


if __name__ == "__main__":
    main()
