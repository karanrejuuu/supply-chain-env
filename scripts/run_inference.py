"""
Rule-based baseline inference script for the Supply Chain Disruption Agent.

Run with:
    python scripts/run_inference.py
"""
import sys
import os

# Ensure project root is on the path when running from the scripts/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from env.environment import SupplyChainEnv
from env.models import Action
from env.grader import grade


def choose_action(obs) -> str:
    """
    Simple priority-based rule agent.

    Priority order:
      1. Fix delayed supplier (prerequisite for effective restocking)
      2. Fix blocked transport (prerequisite for delivery)
      3. Order stock when both routes are clear and inventory is insufficient
      4. Do nothing otherwise (should not happen in well-formed scenarios)
    """
    if obs.supplier_status == "delayed":
        return "switch_supplier"
    if obs.transport_status == "blocked":
        return "reroute_transport"
    if obs.inventory < obs.demand:
        return "order_more_stock"
    return "do_nothing"


def main() -> None:
    task_name = "hard"
    env = SupplyChainEnv(task_name=task_name)
    obs = env.reset()

    print("=" * 60)
    print(f"  Supply Chain Disruption Agent — Task: {task_name.upper()}")
    print("=" * 60)
    print(f"\nInitial State:\n  {obs.model_dump()}\n")

    actions_taken: list[str] = []
    total_reward: float = 0.0
    step_count: int = 0
    done: bool = False

    while not done:
        action_type = choose_action(obs)
        action = Action(action_type=action_type)
        obs, reward, done, info = env.step(action)

        actions_taken.append(action_type)
        total_reward += reward
        step_count += 1

        print(
            f"  Step {step_count:02d} | Action: {action_type:<22} "
            f"| Reward: {reward:+.2f} | Done: {done}"
        )

    print("\n" + "=" * 60)
    print("  Episode Complete")
    print("=" * 60)
    print(f"\nFinal State:\n  {obs.model_dump()}")
    print(f"\nTotal Steps  : {step_count}")
    print(f"Total Reward : {total_reward:.4f}")
    print(f"Actions Taken: {actions_taken}")

    optimal = env._task["optimal_solution"]
    score = grade(actions_taken, optimal)
    print(f"\nGrader Score : {score:.4f}  (optimal: {optimal})")
    print("=" * 60)


if __name__ == "__main__":
    main()
