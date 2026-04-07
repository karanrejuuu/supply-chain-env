"""
Baseline inference script for the Supply Chain Disruption Environment.

Runs a heuristic policy on all 3 tasks and prints grader scores.
Optionally uses an OpenAI-compatible LLM (via HF_TOKEN) for action selection.

Usage:
    python scripts/run_inference.py
    python scripts/run_inference.py --mode llm
"""

import argparse
import json
import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from env.environment import SupplyChainEnv
from env.models import Action, Observation
from env.grader import grade
from env.tasks import TASKS


# ======================================================================
# Heuristic policy
# ======================================================================

def heuristic_policy(obs: Observation) -> Action:
    """
    Priority-based rule agent:
      1. Expedite any delayed supplier
      2. Reroute any blocked route
      3. Reallocate inventory to the warehouse with greatest pending demand
      4. Wait if nothing else to do
    """
    # Check for supplier disruptions
    for sid, info in obs.supplier_info.items():
        if not info.get("active", True):
            return Action(action_type="expedite", supplier_id=sid)

    # Check for route disruptions
    for rid, info in obs.route_info.items():
        if not info.get("active", True):
            return Action(action_type="reroute", route_id=rid)

    # Reallocate: find warehouse with lowest inventory that has pending orders
    if obs.pending_orders:
        # Find destination warehouses with pending orders
        demand_by_wh: dict[str, int] = {}
        for order in obs.pending_orders:
            wh = order.get("destination_warehouse", "")
            qty = order.get("quantity", 0)
            demand_by_wh[wh] = demand_by_wh.get(wh, 0) + qty

        # Find warehouse with highest unmet demand
        best_wh = None
        best_deficit = 0
        for wh, demand in demand_by_wh.items():
            inv = obs.inventory_levels.get(wh, 0)
            deficit = demand - inv
            if deficit > best_deficit:
                best_deficit = deficit
                best_wh = wh

        if best_wh and best_deficit > 0:
            # Find a source warehouse with surplus
            for src_wh, src_inv in obs.inventory_levels.items():
                if src_wh != best_wh and src_inv > 20:
                    transfer = min(src_inv - 10, best_deficit)
                    if transfer > 0:
                        return Action(
                            action_type="reallocate",
                            warehouse_id=best_wh,
                            quantity=transfer,
                        )

    return Action(action_type="wait")


# ======================================================================
# LLM policy (OpenAI-compatible via HF_TOKEN)
# ======================================================================

def llm_policy(obs: Observation, client: "openai.OpenAI") -> Action:
    """Use an LLM to decide the next action."""
    system_prompt = (
        "You are a supply chain optimization agent. Given the current state, "
        "choose the best action. Respond with valid JSON matching the Action schema:\n"
        '{"action_type": "reroute|expedite|reallocate|wait", '
        '"route_id": "...", "supplier_id": "...", "warehouse_id": "...", "quantity": N}\n'
        "Only include fields relevant to your chosen action_type."
    )
    user_prompt = f"Current observation:\n{json.dumps(obs.model_dump(), indent=2)}"

    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-72B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=200,
        temperature=0.0,
    )
    raw = response.choices[0].message.content.strip()
    # Extract JSON from response
    try:
        if "```" in raw:
            raw = raw.split("```")[1].strip()
            if raw.startswith("json"):
                raw = raw[4:].strip()
        action_data = json.loads(raw)
        return Action(**action_data)
    except Exception:
        return Action(action_type="wait")


# ======================================================================
# Main runner
# ======================================================================

def run_task(task_name: str, policy_fn, client=None) -> dict:
    """Run a single task and return metrics + score."""
    env = SupplyChainEnv(task_name=task_name)
    obs = env.reset()

    print(f"\n{'=' * 65}")
    print(f"  Task: {task_name.upper()} — {env.get_task().description[:60]}...")
    print(f"{'=' * 65}")
    print(f"  Warehouses: {obs.inventory_levels}")
    print(f"  Orders:     {len(obs.pending_orders)} pending")
    print(f"  Disruptions:{len(obs.disruption_status)} active\n")

    total_reward = 0.0
    done = False
    info = {}

    while not done:
        if client:
            action = policy_fn(obs, client)
        else:
            action = policy_fn(obs)

        obs, reward, done, info = env.step(action)
        total_reward += reward

        step = info.get("step", "?")
        print(
            f"  Step {step:02d} | {action.action_type:<12} "
            f"| R: {reward:+.3f} | Cost: {obs.cost_so_far:.1f} | Done: {done}"
        )

    # Grade
    score = grade(info)

    print(f"\n  {'─' * 50}")
    print(f"  Fulfilled: {info.get('orders_fulfilled', 0)}/{info.get('total_orders', 0)}")
    print(f"  Total Cost: {info.get('total_cost', 0):.2f}")
    print(f"  Steps Used: {info.get('time_step', 0)}/{info.get('max_steps', '?')}")
    print(f"  Grader Score: {score:.4f}")

    return {"task": task_name, "score": score, **info}


def main() -> None:
    parser = argparse.ArgumentParser(description="Supply Chain Env — Baseline Inference")
    parser.add_argument(
        "--mode", choices=["heuristic", "llm"], default="heuristic",
        help="Policy mode: heuristic (default) or llm (requires HF_TOKEN)",
    )
    args = parser.parse_args()

    client = None
    policy_fn = heuristic_policy

    if args.mode == "llm":
        try:
            import openai
        except ImportError:
            print("ERROR: openai package required for LLM mode. pip install openai")
            sys.exit(1)

        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            print("ERROR: Set HF_TOKEN environment variable for LLM inference.")
            sys.exit(1)

        client = openai.OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=hf_token,
        )
        policy_fn = llm_policy

    print("\n" + "=" * 65)
    print("  SUPPLY CHAIN DISRUPTION AGENT — BASELINE EVALUATION")
    print("=" * 65)

    results = []
    for task_name in ["easy", "medium", "hard"]:
        result = run_task(task_name, policy_fn, client)
        results.append(result)

    # Summary table
    print(f"\n\n{'=' * 65}")
    print("  RESULTS SUMMARY")
    print(f"{'=' * 65}")
    print(f"  {'Task':<10} {'Score':>8} {'Fulfilled':>12} {'Cost':>10} {'Steps':>8}")
    print(f"  {'─' * 50}")
    for r in results:
        fulfilled = f"{r.get('orders_fulfilled', 0)}/{r.get('total_orders', 0)}"
        print(
            f"  {r['task']:<10} {r['score']:>8.4f} {fulfilled:>12} "
            f"{r.get('total_cost', 0):>10.2f} {r.get('time_step', 0):>8}"
        )
    avg_score = sum(r["score"] for r in results) / len(results)
    print(f"  {'─' * 50}")
    print(f"  {'AVERAGE':<10} {avg_score:>8.4f}")
    print(f"{'=' * 65}\n")


if __name__ == "__main__":
    main()
