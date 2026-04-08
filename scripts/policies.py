from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any

from envs.supply_chain_env.models import SupplyChainAction as Action, SupplyChainObservation as Observation
from envs.supply_chain_env.server.grader import grade
from envs.supply_chain_env.server.simulator import SupplyChainSimulator


REALLOCATION_QUANTITIES = (5, 10, 15, 20, 25, 30)


def _candidate_actions(obs: Observation) -> list[Action]:
    actions = [Action(action_type="wait")]

    for disruption in obs.disruption_status:
        if disruption.disruption_type == "supplier_delay":
            actions.append(Action(action_type="expedite", supplier_id=disruption.target_id))
        elif disruption.disruption_type == "route_block":
            actions.append(Action(action_type="reroute", route_id=disruption.target_id))

    for warehouse_id in obs.inventory_levels:
        for quantity in REALLOCATION_QUANTITIES:
            actions.append(
                Action(
                    action_type="reallocate",
                    warehouse_id=warehouse_id,
                    quantity=quantity,
                )
            )

    unique_actions: list[Action] = []
    seen: set[tuple[tuple[str, object], ...]] = set()
    for action in actions:
        key = tuple(sorted(action.model_dump(exclude_none=True).items()))
        if key not in seen:
            seen.add(key)
            unique_actions.append(action)
    return unique_actions


def _build_shadow_simulator(obs: Observation) -> SupplyChainSimulator:
    simulator = SupplyChainSimulator()
    simulator.warehouses = dict(obs.inventory_levels)
    simulator.suppliers = {supplier_id: dict(info) for supplier_id, info in obs.supplier_info.items()}
    simulator.routes = {route_id: dict(info) for route_id, info in obs.route_info.items()}
    simulator.orders = [order.model_copy(deep=True) for order in obs.pending_orders]
    simulator.disruptions = [disruption.model_copy(deep=True) for disruption in obs.disruption_status]
    simulator.total_cost = obs.cost_so_far
    simulator.time_step = obs.time_step
    simulator.orders_fulfilled = obs.orders_fulfilled
    simulator.orders_missed = obs.orders_missed
    simulator.total_orders = obs.total_orders
    return simulator


def _rollout_once(obs: Observation, action: Action) -> tuple[Observation, dict[str, float]]:
    simulator = _build_shadow_simulator(obs)
    simulator.apply_action(action)
    simulator.step_time()
    next_obs = simulator.get_observation()
    next_obs.task_name = obs.task_name
    next_obs.max_steps = obs.max_steps
    metrics = simulator.get_metrics()
    metrics["max_steps"] = obs.max_steps
    return next_obs, metrics


def _terminal(obs: Observation) -> bool:
    return obs.time_step >= obs.max_steps or len(obs.pending_orders) == 0


@lru_cache(maxsize=4096)
def _best_score(obs_json: str, depth: int) -> float:
    obs = Observation.model_validate_json(obs_json)
    if depth <= 0 or _terminal(obs):
        metrics = {
            "orders_fulfilled": obs.orders_fulfilled,
            "orders_missed": obs.orders_missed,
            "total_orders": obs.total_orders,
            "total_cost": obs.cost_so_far,
            "time_step": obs.time_step,
            "max_steps": obs.max_steps,
        }
        return grade(metrics)

    best = 0.0
    for action in _candidate_actions(obs):
        next_obs, metrics = _rollout_once(obs, action)
        immediate = grade(metrics)
        if depth > 1 and not _terminal(next_obs):
            immediate = max(immediate, _best_score(next_obs.model_dump_json(), depth - 1))
        best = max(best, immediate)
    return best


def _search_depth(obs: Observation) -> int:
    if obs.max_steps >= 20 or len(obs.pending_orders) >= 4:
        return 4
    if len(obs.pending_orders) >= 2:
        return 3
    return 2


def heuristic_policy(obs: Observation) -> Action:
    """Search-backed baseline using the real simulator dynamics and grading signal."""
    depth = _search_depth(obs)
    best_action = Action(action_type="wait")
    best_key: tuple[float, float] | None = None

    for action in _candidate_actions(obs):
        next_obs, metrics = _rollout_once(obs, action)
        score = grade(metrics)
        if depth > 1 and not _terminal(next_obs):
            score = max(score, _best_score(next_obs.model_dump_json(), depth - 1))
        key = (score, -metrics["total_cost"])
        if best_key is None or key > best_key:
            best_key = key
            best_action = action

    return best_action


def get_optimal_heuristic_action(obs: Observation) -> Action:
    return heuristic_policy(obs)


def llm_policy(
    obs: Observation,
    client: Any,
    model_name: str | None = None,
    structured_output: bool = True,
) -> Action:
    """Use an LLM to decide the next action. Falls back to heuristic on failure."""
    if client is None:
        return heuristic_policy(obs)

    system_prompt = (
        "You are a supply chain optimization agent. "
        "Return a single JSON object with one of these actions: reroute, expedite, reallocate, wait. "
        "Only include the fields needed for that action."
    )
    user_prompt = (
        "Current observation:\n"
        f"{json.dumps(obs.model_dump(mode='json'), indent=2)}\n\n"
        "Return JSON matching this schema: "
        '{"action_type":"reroute|expedite|reallocate|wait","route_id":null,"supplier_id":null,"warehouse_id":null,"quantity":null}'
    )

    request_kwargs = {
        "model": model_name or os.environ.get("SUPPLY_CHAIN_LLM_MODEL", "Qwen/Qwen2.5-72B-Instruct"),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 200,
        "temperature": 0.1,
    }
    if structured_output:
        request_kwargs["response_format"] = {"type": "json_object"}

    try:
        response = client.chat.completions.create(**request_kwargs)
        raw = response.choices[0].message.content or "{}"
        if isinstance(raw, list):
            raw = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in raw)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```", 2)[1].strip()
            if raw.startswith("json"):
                raw = raw[4:].strip()
        action_data = json.loads(raw)
        return Action.model_validate(action_data)
    except Exception:
        return heuristic_policy(obs)
