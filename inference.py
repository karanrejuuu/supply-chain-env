"""
Round-1 baseline inference runner for Supply Chain OpenEnv.

Required environment variables:
- API_BASE_URL: OpenAI-compatible endpoint URL
- MODEL_NAME: model identifier
- HF_TOKEN: API key/token for the endpoint

This script emits strict structured stdout lines:
[START], [STEP], [END]
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from envs.supply_chain_env.models import SupplyChainAction, SupplyChainObservation
from envs.supply_chain_env.server.environment import SupplyChainEnvironment
from scripts.policies import heuristic_policy

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
BENCHMARK = "supply-chain-disruption-agent"
TASKS = ("easy", "medium", "hard")


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def _build_llm_messages(obs: SupplyChainObservation) -> list[dict[str, str]]:
    system_prompt = (
        "You are a supply-chain operations agent. "
        "Respond with one JSON object only, selecting exactly one action: "
        "reroute, expedite, reallocate, or wait."
    )
    user_prompt = (
        "Observation:\n"
        f"{json.dumps(obs.model_dump(mode='json'), separators=(',', ':'))}\n\n"
        "Return JSON matching: "
        '{"action_type":"reroute|expedite|reallocate|wait","route_id":null,'
        '"supplier_id":null,"warehouse_id":null,"quantity":null}'
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _sanitize_action(data: dict) -> SupplyChainAction:
    try:
        return SupplyChainAction.model_validate(data)
    except Exception:
        return SupplyChainAction(action_type="wait")


def choose_action(client: OpenAI, obs: SupplyChainObservation) -> SupplyChainAction:
    """
    Primary policy uses OpenAI-compatible completion.
    Falls back to heuristic when response is invalid/unavailable.
    """
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=_build_llm_messages(obs),
            temperature=0.1,
            max_tokens=200,
            response_format={"type": "json_object"},
        )
        text = (response.choices[0].message.content or "{}").strip()
        payload = json.loads(text)
        return _sanitize_action(payload)
    except Exception:
        return heuristic_policy(obs)


def run_task(task_name: str, client: OpenAI) -> None:
    env = SupplyChainEnvironment(task_name=task_name)
    rewards: list[float] = []
    steps_taken = 0
    success = False
    score = 0.0

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        observation = env.reset()
        while not observation.done:
            action = choose_action(client, observation)
            observation = env.step(action)

            reward = float(observation.reward or 0.0)
            rewards.append(reward)
            steps_taken = observation.time_step
            action_str = json.dumps(action.model_dump(exclude_none=True), separators=(",", ":"))
            log_step(
                step=steps_taken,
                action=action_str,
                reward=reward,
                done=observation.done,
                error=None,
            )

        state = env.state
        score = float(state.score or 0.0)
        score = min(max(score, 0.0), 1.0)
        success = bool(state.orders_fulfilled == state.total_orders and state.orders_missed == 0)
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main() -> None:
    if not API_KEY:
        raise SystemExit("Missing HF_TOKEN (or OPENAI_API_KEY) environment variable for OpenAI client.")
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    for task_name in TASKS:
        run_task(task_name, client)


if __name__ == "__main__":
    main()
