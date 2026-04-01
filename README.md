---
title: Meta Project
sdk: docker
app_port: 8000
short_description: SOC threat-response simulation with deterministic rewards.
---

# CyberSec-OpenEnv

AI Cybersecurity Incident Response Environment for [OpenEnv](https://meta-pytorch.org/OpenEnv/). The agent reads synthetic SOC-style log lines, classifies threats, and chooses actions (`classify`, `respond`, `escalate`, `ignore`) with dense, deterministic feedback.

## Motivation

Train and evaluate LLM agents on realistic triage: interpret noisy logs under time pressure, avoid false escalations, and complete multi-step playbooks (for example, classify then escalate with an investigation request).

## Action space

JSON body for `POST /step`:

| Field | Description |
|-------|-------------|
| `action_type` | One of: `classify`, `respond`, `escalate`, `ignore` |
| `content` | Free text: label, user-facing response, escalation justification, or notes |
| `metadata` | Optional dict (OpenEnv `Action` base field; defaults to `{}`) |

## Observation space
