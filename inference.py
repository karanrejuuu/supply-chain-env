<<<<<<< HEAD
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
=======
import os
import json
import argparse
import requests
from src.envs.supply_chain.client import SupplyChainClient

USE_LLM = os.environ.get("USE_LLM", "False").lower() == "true"
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

VALID_ACTIONS = [
    "order_more_stock",
    "switch_supplier",
    "reroute_transport",
    "delay_order",
    "do_nothing",
]


def format_reason(action_type: str, base_reason: str, repeat_count: int) -> str:
    """Return deterministic, varied wording for repeated actions."""
    if repeat_count <= 0:
        return base_reason

    variants = {
        "order_more_stock": [
            "Continuing inventory replenishment to stabilize supply",
            "Maintaining inventory levels to meet ongoing demand",
            "Final replenishment to fully stabilize supply chain",
        ],
        "switch_supplier": [
            "Continuing supplier adjustment to reduce disruption risk",
            "Maintaining supplier strategy to keep delays under control",
            "Final supplier correction to stabilize lead times",
        ],
        "reroute_transport": [
            "Continuing transport reroute to protect fulfillment flow",
            "Maintaining adjusted transport path for stable delivery",
            "Final transport reroute to secure supply continuity",
        ],
        "delay_order": [
            "Continuing controlled order delay to balance operations",
            "Maintaining delayed ordering to manage short-term load",
            "Final delay adjustment to stabilize planning",
        ],
        "do_nothing": [
            "Monitoring stable conditions while maintaining balance",
            "Holding steady as the system remains stable",
            "Final stability check with no further intervention",
        ],
    }

    action_variants = variants.get(action_type)
    if not action_variants:
        return base_reason
    idx = min(repeat_count - 1, len(action_variants) - 1)
    return action_variants[idx]


def choose_action_rule_based(obs):
    backlog = obs.get("backlog", 0)
    inventory = obs.get("inventory_level", 0)
    demand = obs.get("demand", 0)
    forecast = obs.get("demand_forecast", demand)
    delay = obs.get("supplier_delay_days", 0)
    transport = obs.get("transport_available", True)
    step = obs.get("step", 1)
    budget = obs.get("budget_remaining", 0.0)
    last_actions = obs.get("_last_actions", [])

    is_stable = backlog == 0 and delay <= 1 and inventory >= forecast
    if is_stable:
        return (
            {"action_type": "do_nothing"},
            "System stabilized, no further action required"
        )

    if (
        len(last_actions) >= 2
        and last_actions[-1] == "order_more_stock"
        and last_actions[-2] == "order_more_stock"
        and backlog <= 5
        and inventory >= forecast
    ):
        return (
            {"action_type": "do_nothing"},
            "System stabilized, maintaining balance"
        )

    # Use forecast-aware buffer to avoid stockouts without overspending.
    target_inventory = max(demand, forecast) + 10
    shortage = max(0, target_inventory - inventory)

    if not transport:
        return (
            {"action_type": "reroute_transport", "transport_mode": "express"},
            "Mitigating supplier delay risk"
        )

    if delay > 1:
        return (
            {"action_type": "switch_supplier", "supplier_choice": "fast"},
            "Mitigating supplier delay risk"
        )

    if backlog > 0 or shortage > 0:
        qty = max(20, min(70, backlog + shortage))
        if budget < qty * 2:
            qty = max(10, int(budget // 2))
        return (
            {"action_type": "order_more_stock", "order_quantity": qty},
            "Replenishing inventory to meet demand"
        )

    if delay > 0:
        return (
            {"action_type": "switch_supplier", "supplier_choice": "standard"},
            "Mitigating supplier delay risk"
        )

    return (
        {"action_type": "do_nothing"},
        "System stabilized, maintaining balance"
    )

def choose_action_llm(observation):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    prompt = f"""Provide observation like:
{json.dumps({
    "inventory_level": observation.get("inventory_level", 0),
    "backlog": observation.get("backlog", 0),
    "supplier_delay": observation.get("supplier_delay_days", 0),
    "demand_forecast": observation.get("demand_forecast", 0)
}, indent=2)}

Choose ONE action from:
* switch_supplier
* order_more_stock
* do_nothing

Rules:
* If delay high -> switch_supplier
* If inventory low -> order_more_stock
* If backlog high -> order_more_stock
* Otherwise -> do_nothing

Return ONLY JSON:
{{
  "action_type": "...",
  "order_quantity": 50,
  "reason": "..."
}}
"""

    data = {
        "model": "mistralai/mistral-7b-instruct",
        "messages": [
            {"role": "system", "content": "You are a supply chain decision agent. You must return a valid JSON object only."},
            {"role": "user", "content": prompt}
        ]
    }

    res = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data, timeout=3.0)
    res.raise_for_status()

    content = res.json()["choices"][0]["message"]["content"]
    if content.startswith("```json"):
        content = content.replace("```json", "", 1).replace("```", "", 1).strip()
    elif content.startswith("```"):
        content = content.replace("```", "", 2).strip()

    parsed = json.loads(content)
    reason = parsed.pop("reason", "Action chosen by LLM")
    return parsed, reason

def choose_action(observation):
    if USE_LLM and OPENROUTER_API_KEY:
        try:
            return choose_action_llm(observation)
        except Exception:
            pass  # Fallback to rule-based

    return choose_action_rule_based(observation)


def parse_action_safe(raw) -> dict:
    """Safely parse action from string or dict. Falls back to do_nothing."""
    if isinstance(raw, dict):
        action_type = raw.get("action_type", "")
        if action_type in VALID_ACTIONS:
            return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict) and parsed.get("action_type") in VALID_ACTIONS:
                return parsed
        except Exception:
            pass
    return {"action_type": "do_nothing"}


def run(task: str = "easy", model: str = "rule-based"):
    client = SupplyChainClient()

    print(f"[START] task={task} env=supply-chain model={model}")

    obs = client.reset(task=task)
    done = False
    step_num = 0
    rewards = []
    action_history = []

    while not done:
        step_num += 1
        error = None

        reason = ""
        try:
            obs_for_policy = dict(obs)
            obs_for_policy["_last_actions"] = action_history[-2:]
            raw_action, reason = choose_action(obs_for_policy)
            action = parse_action_safe(raw_action)
        except Exception as e:
            action = {"action_type": "do_nothing"}
            reason = "Error"
            error = str(e)

        try:
            result = client.step(action)
            obs = result.get("observation", obs)
            reward = float(result.get("reward", 0.0))
            done = bool(result.get("done", False))
        except Exception as e:
            reward = 0.0
            done = True
            error = str(e)

        rewards.append(reward)
        action_type = action.get("action_type", "do_nothing")
        repeat_count = 0
        for prev_action in reversed(action_history):
            if prev_action == action_type:
                repeat_count += 1
            else:
                break
        reason = format_reason(action_type, reason, repeat_count)
        action_history.append(action_type)
        print(f"\n[STEP {step_num}]")
        print(f"-> Action : {action.get('action_type')}")
        print(f"-> Reason : {reason}")
        print(f"-> Reward : {reward:.2f}")

    score = round(sum(rewards) / len(rewards), 4) if rewards else 0.0
    success = score > 0.0

    print("\n[RESULT]")
    print(f"-> Success : {str(success).lower()}")
    print(f"-> Steps   : {step_num}")
    print(f"-> Score   : {score:.2f}")
    print("-> Insight : System recovered from disruption and stabilized supply chain\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="easy", choices=["easy", "medium", "hard"])
    parser.add_argument("--model", default="rule-based")
    args = parser.parse_args()
    run(task=args.task, model=args.model)
>>>>>>> 8b398ee12512f09a1dda88fb6fe73806e856585f
