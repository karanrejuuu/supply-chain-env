"""
Advanced inference script for the Supply Chain Disruption Environment.
Handles batch execution, comprehensive logging, LLM fallbacks, and metric plots.

Usage:
    python scripts/run_inference.py --mode heuristic --batch 3
    python scripts/run_inference.py --mode llm --batch 3
"""

import argparse
import json
import os
import sys
import copy
from typing import Any

import pandas as pd
import matplotlib.pyplot as plt

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.envs.supply_chain_env.server.environment import SupplyChainEnvironment as SupplyChainEnv
from src.envs.supply_chain_env.models import SupplyChainAction as Action, SupplyChainObservation as Observation
from src.envs.supply_chain_env.server.grader import grade
from src.envs.supply_chain_env.server.tasks import TASKS


# ======================================================================
# Heuristic policy
# ======================================================================

def heuristic_policy(obs: Observation) -> Action:
    """
    Cost-minimizing heuristic:
      Evaluates available fix candidates (reroute, expedite, reallocate)
      and chooses the one with the lowest immediate cost. Priorities are
      dynamically balanced by cost. Wait is returned if no actions are viable.
    """
    candidates = []
    
    # 1. Evaluate reroute costs
    for rid, info in obs.route_info.items():
        if not info.get("active", True):
            candidates.append({"type": "reroute", "id": rid, "cost": info.get("cost", 20)})
            
    # 2. Evaluate expedite costs
    for sid, info in obs.supplier_info.items():
        if not info.get("active", True):
            # Expedite cost calculation based on simulator
            cost = 30 * info.get("capacity", 1) / 100
            candidates.append({"type": "expedite", "id": sid, "cost": cost})
            
    # 3. Evaluate reallocate costs
    if obs.pending_orders:
        demand_by_wh: dict[str, int] = {}
        for o in obs.pending_orders:
            wh = o.get("destination_warehouse", "")
            demand_by_wh[wh] = demand_by_wh.get(wh, 0) + o.get("quantity", 0)
            
        for wh, demand in demand_by_wh.items():
            inv = obs.inventory_levels.get(wh, 0)
            if demand > inv:
                deficit = demand - inv
                for src_wh, src_inv in obs.inventory_levels.items():
                    if src_wh != wh and src_inv > deficit:
                        qty = deficit
                        # Reallocation cost is roughly 0.5 per unit
                        candidates.append({
                            "type": "reallocate", 
                            "id": wh, 
                            "src": src_wh,
                            "qty": qty, 
                            "cost": qty * 0.5
                        })
                        break
                        
    if candidates:
        # Prioritize lower cost actions
        candidates.sort(key=lambda x: x["cost"])
        best = candidates[0]
        
        if best["type"] == "reroute":
            return Action(action_type="reroute", route_id=best["id"])
        elif best["type"] == "expedite":
            return Action(action_type="expedite", supplier_id=best["id"])
        elif best["type"] == "reallocate":
            return Action(action_type="reallocate", warehouse_id=best["id"], quantity=best["qty"])
            
    return Action(action_type="wait")

def get_optimal_heuristic_action(obs: Observation) -> Action:
    """Helper to expose optimal heuristic action for accuracy tracking."""
    return heuristic_policy(obs)


# ======================================================================
# LLM policy (OpenAI-compatible via HF_TOKEN)
# ======================================================================

def llm_policy(obs: Observation, client: Any) -> Action:
    """Use an LLM to decide the next action. Fallback to heuristic on failure."""
    system_prompt = (
        "You are a supply chain optimization agent. Given the current state, "
        "choose the best action. Respond with valid JSON matching the Action schema:\n"
        '{"action_type": "reroute|expedite|reallocate|wait", '
        '"route_id": "...", "supplier_id": "...", "warehouse_id": "...", "quantity": N}\n'
        "Only include fields relevant to your chosen action_type.\n"
        "Remember to balance taking actions that resolve disruptions "
        "vs waiting or transferring inventory efficiently."
    )
    user_prompt = f"Current observation:\n{json.dumps(obs.model_dump(), indent=2)}"

    try:
        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-72B-Instruct",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=200,
            temperature=0.2, # Slight variation for batches
        )
        raw = response.choices[0].message.content.strip()
        if "```" in raw:
            raw = raw.split("```")[1].strip()
            if raw.startswith("json"):
                raw = raw[4:].strip()
        action_data = json.loads(raw)
        return Action(**action_data)
    except Exception as e:
        print(f"  [LLM Error: {e}] -> Falling back to heuristic policy.")
        return heuristic_policy(obs)


# ======================================================================
# Logging & Visualization
# ======================================================================

def save_plots_and_logs(logs: list, task_name: str, mode: str, run_idx: int) -> None:
    """Generate CSV and PNG plots for the completed run."""
    os.makedirs("results", exist_ok=True)
    suffix = f"{task_name}_{mode}_run{run_idx}"
    
    # Exclude complex nested dicts for CSV by unrolling them
    csv_logs = []
    for log in logs:
        flat = {
            "step": log["step"],
            "action": log["action"],
            "cost_so_far": log["cost_so_far"],
            "pending_orders": log["pending_orders"],
            "active_disruptions": log["active_disruptions"]
        }
        for wh, inv in log["inventories"].items():
            flat[f"inv_{wh}"] = inv
        csv_logs.append(flat)
        
    df = pd.DataFrame(csv_logs)
    csv_path = os.path.join("results", f"log_{suffix}.csv")
    df.to_csv(csv_path, index=False)
    
    # Plot 1: Inventory over time
    plt.figure(figsize=(10, 6))
    for wh in logs[0]["inventories"].keys():
        col = f"inv_{wh}"
        if col in df.columns:
            plt.plot(df["step"], df[col], marker='o', label=wh)
    plt.title(f"Warehouse Inventory over Time - {task_name.upper()} ({mode})")
    plt.xlabel("Step")
    plt.ylabel("Inventory Count")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("results", f"plot_inventory_{suffix}.png"))
    plt.close()
    
    # Plot 2: Cost Progression
    plt.figure(figsize=(10, 6))
    plt.plot(df["step"], df["cost_so_far"], color='red', marker='x', label="Total Cost")
    plt.title(f"Cost Progression - {task_name.upper()} ({mode})")
    plt.xlabel("Step")
    plt.ylabel("Cumulative Cost")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("results", f"plot_cost_{suffix}.png"))
    plt.close()

    # Plot 3: Actions taken
    plt.figure(figsize=(8, 6))
    action_counts = df["action"].value_counts().drop("DONE", errors="ignore")
    if not action_counts.empty:
        action_counts.plot(kind="bar", color="skyblue")
        plt.title(f"Actions Taken - {task_name.upper()} ({mode})")
        plt.xlabel("Action Type")
        plt.ylabel("Frequency")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join("results", f"plot_actions_{suffix}.png"))
    plt.close()


# ======================================================================
# Batch evaluation
# ======================================================================

def run_single_simulation(task_name: str, mode: str, client: Any, baseline_cost: float, run_idx: int) -> dict:
    """Run one configured instance of the scenario."""
    env = SupplyChainEnv(task_name=task_name)
    obs = env.reset()
    
    logs = []
    done = False
    
    order_fulfillment_times = []
    total_reward = 0.0
    action_match_count = 0
    total_actions = 0
    reroutes = 0
    prev_fulfilled = 0
    
    while not done:
        # Get optimal baseline action for computing "accuracy" 
        optimal_action = get_optimal_heuristic_action(obs)
        
        # Select current policy action
        if mode == "llm":
            action = llm_policy(obs, client)
        else:
            action = heuristic_policy(obs)
            
        # Accuracy Check: Does our selected action match the baseline optimal?
        is_optimal = (
            action.action_type == optimal_action.action_type and
            action.route_id == optimal_action.route_id and 
            action.supplier_id == optimal_action.supplier_id and
            action.warehouse_id == optimal_action.warehouse_id
        )
        if is_optimal:
            action_match_count += 1
        total_actions += 1
        
        if action.action_type == "reroute":
            reroutes += 1
            
        # Snapshot state prior to stepping for log
        logs.append({
            "step": obs.time_step,
            "action": action.action_type,
            "cost_so_far": obs.cost_so_far,
            "pending_orders": len(obs.pending_orders),
            "active_disruptions": len(obs.disruption_status),
            "inventories": copy.deepcopy(obs.inventory_levels)
        })
        
        # Step the environment
        obs = env.step(action)
        state = env.state
        done = obs.done
        
        # Track fulfilment timing delays metrics
        currently_fulfilled = state.orders_fulfilled
        for _ in range(currently_fulfilled - prev_fulfilled):
            # Records time step taken to fulfil unit orders
            order_fulfillment_times.append(obs.time_step)
        prev_fulfilled = currently_fulfilled

    # Post-episode recording
    logs.append({
        "step": obs.time_step,
        "action": "DONE",
        "cost_so_far": obs.cost_so_far,
        "pending_orders": len(obs.pending_orders),
        "active_disruptions": len(obs.disruption_status),
        "inventories": copy.deepcopy(obs.inventory_levels)
    })
    
    save_plots_and_logs(logs, task_name, mode, run_idx)
    
    state = env.state
    # Compute Final Evaluation Metrics
    score = state.score
    final_cost = state.total_cost
    
    # Cost Efficiency (%)
    cost_efficiency = 0.0
    if baseline_cost > 0:
        cost_efficiency = ((baseline_cost - final_cost) / baseline_cost) * 100.0
        
    avg_delay = sum(order_fulfillment_times) / max(1, len(order_fulfillment_times))
    action_accuracy = (action_match_count / max(1, total_actions)) * 100.0
    
    return {
        "task": task_name,
        "run": run_idx,
        "score": score,
        "cost": final_cost,
        "efficiency": cost_efficiency,
        "fulfilled": state.orders_fulfilled,
        "total_orders": state.total_orders,
        "avg_delay": avg_delay,
        "action_accuracy": action_accuracy,
        "reroutes": reroutes,
        "time_step": state.step_count
    }

def get_baseline_cost(task_name: str) -> float:
    """Run the pure baseline heuristic exactly once to calculate strict baseline cost."""
    env = SupplyChainEnv(task_name=task_name)
    obs = env.reset()
    done = False
    while not done:
        action = heuristic_policy(obs)
        obs = env.step(action)
        done = obs.done
    return env.state.total_cost


# ======================================================================
# Main CLI
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Supply Chain Env — Advanced Evaluation")
    parser.add_argument(
        "--mode", choices=["heuristic", "llm"], default="heuristic",
        help="Policy mode: heuristic (default) or llm (requires HF_TOKEN)",
    )
    parser.add_argument(
        "--batch", type=int, default=1,
        help="Number of simulations to run per task for batch evaluation.",
    )
    args = parser.parse_args()

    client = None
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

    print("\n" + "=" * 80)
    print("  SUPPLY CHAIN DISRUPTION AGENT — ADVANCED EVALUATION")
    print("=" * 80)

    tasks_to_eval = ["easy", "medium", "hard"]
    all_results = []

    try:
        import pandas  # test for logging
    except ImportError:
        print("ERROR: pandas or matplotlib library missing. Please pip install pandas matplotlib.")
        sys.exit(1)

    for task_name in tasks_to_eval:
        print(f"\nEvaluating Task: {task_name.upper()}")
        baseline_cost = get_baseline_cost(task_name)
        
        for run_idx in range(args.batch):
            print(f"  -> Run {run_idx+1}/{args.batch} [{args.mode}]")
            result = run_single_simulation(task_name, args.mode, client, baseline_cost, run_idx)
            all_results.append(result)

    # -----------------------------------------------------
    # Build Display & CSV Summaries
    # -----------------------------------------------------
    df_res = pd.DataFrame(all_results)
    os.makedirs("results", exist_ok=True)
    df_res.to_csv("results/batch_summary.csv", index=False)

    print(f"\n\n{'=' * 105}")
    print("  FINAL AGGREGATED METRICS (Averaged across runs)")
    print(f"{'=' * 105}")
    print(f"  {'Task':<8} | {'Score':>7} | {'Cost':>7} | {'Eff.(%)':>7} | {'Fulf%':>6} | {'Delay':>6} | {'Acc(%)':>6} | {'Reroutes':>8}")
    print(f"  {'-' * 100}")

    for task_name in tasks_to_eval:
        subset = df_res[df_res["task"] == task_name]
        if subset.empty:
            continue
            
        avg_score = subset["score"].mean()
        avg_cost = subset["cost"].mean()
        avg_eff = subset["efficiency"].mean()
        avg_fulf = (subset["fulfilled"] / subset["total_orders"]).mean() * 100
        avg_delay = subset["avg_delay"].mean()
        avg_acc = subset["action_accuracy"].mean()
        avg_reroutes = subset["reroutes"].mean()
            
        print(f"  {task_name:<8} | {avg_score:>7.3f} | {avg_cost:>7.1f} | {avg_eff:>7.1f} | {avg_fulf:>6.1f} | {avg_delay:>6.1f} | {avg_acc:>6.1f} | {avg_reroutes:>8.1f}")

    print(f"{'=' * 105}\n")
    print(f"Logs and plots saved to the 'results' directory.")

if __name__ == "__main__":
    main()
