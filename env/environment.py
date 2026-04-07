"""
SupplyChainEnv — OpenEnv-compatible environment.

Wraps the SupplyChainSimulator with a standard reset/step interface,
strong reward shaping, done conditions, and rich info dicts.
"""

from __future__ import annotations

import copy
from typing import Any

from env.models import Action, Observation
from env.simulator import SupplyChainSimulator
from env.tasks import TASKS, TaskDefinition


class SupplyChainEnv:
    """
    OpenEnv-compatible environment simulating supply chain disruptions.

    The agent must recover from disruptions (supplier delays, route blocks,
    inventory imbalances) through corrective actions to satisfy customer
    demand as efficiently as possible.
    """

    def __init__(self, task_name: str = "hard") -> None:
        if task_name not in TASKS:
            raise ValueError(
                f"Unknown task '{task_name}'. Choose from: {list(TASKS.keys())}"
            )
        self._task_name = task_name
        self._task: TaskDefinition = TASKS[task_name]
        self._sim = SupplyChainSimulator()
        self._done = False
        self._step_count = 0
        self._cumulative_reward = 0.0
        self._action_history: list[str] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self) -> Observation:
        """Reset environment to the initial state of the current task."""
        self._done = False
        self._step_count = 0
        self._cumulative_reward = 0.0
        self._action_history = []

        obs = self._sim.reset(
            seed=self._task.seed,
            warehouses=self._task.warehouses,
            suppliers=self._task.suppliers,
            routes=self._task.routes,
            orders=self._task.orders,
            disruptions=self._task.disruptions,
        )
        return obs

    def step(self, action: Action) -> tuple[Observation, float, bool, dict]:
        """
        Apply one agent action, advance time, compute shaped reward.

        Returns (observation, reward, done, info).
        """
        if self._done:
            raise RuntimeError(
                "Episode is done. Call reset() to start a new episode."
            )

        self._step_count += 1
        self._action_history.append(action.action_type)

        # Snapshot cost before action for delta tracking
        cost_before = self._sim.total_cost

        # --- Apply action & get action reward ---
        action_reward = self._sim.apply_action(action)

        # --- Advance simulation clock & get time-step reward ---
        time_reward = self._sim.step_time()

        # Cost incurred this step
        step_cost = self._sim.total_cost - cost_before

        # --- Shaped reward ---
        reward = self._shape_reward(action, action_reward, time_reward, step_cost)
        self._cumulative_reward += reward

        # --- Done conditions ---
        if self._sim.all_orders_done:
            self._done = True
            reward += 2.0  # terminal bonus for full fulfilment
        elif self._step_count >= self._task.max_steps:
            self._done = True
            # Penalise any unfulfilled orders at timeout
            unfulfilled = len(self._sim.orders)
            reward -= unfulfilled * 0.5

        # --- Build info dict ---
        metrics = self._sim.get_metrics()
        metrics["max_steps"] = self._task.max_steps
        info = {
            "task": self._task_name,
            "step": self._step_count,
            "cumulative_reward": self._cumulative_reward + reward,
            "action_history": list(self._action_history),
            **metrics,
        }

        obs = self._sim.get_observation()
        return obs, reward, self._done, info

    def get_task(self) -> TaskDefinition:
        """Return the current task definition."""
        return self._task

    # ------------------------------------------------------------------
    # Reward shaping
    # ------------------------------------------------------------------

    def _shape_reward(
        self, action: Action, action_reward: float, time_reward: float,
        step_cost: float,
    ) -> float:
        """
        Compose final step reward from multiple signals.

        Positive:
          - Fulfilling orders (from time_reward)
          - Resolving disruptions (from action_reward)
          - Efficient routing (low cost actions get bonus)

        Negative:
          - Unmet demand / missed deadlines (from time_reward)
          - High cost actions
          - Unnecessary / invalid actions (from action_reward)
          - Repeated useless actions (wait streaks)
        """
        reward = 0.0

        # Base signals
        reward += action_reward
        reward += time_reward

        # Cost efficiency shaping: penalise expensive single-step costs
        if step_cost > 40:
            reward -= 0.3  # heavy cost penalty
        elif step_cost > 20:
            reward -= 0.1  # moderate cost penalty
        elif 0 < step_cost <= 10:
            reward += 0.1  # bonus for cost-efficient actions

        # Penalise repeated identical non-productive actions
        if len(self._action_history) >= 3:
            last3 = self._action_history[-3:]
            if last3[0] == last3[1] == last3[2] and last3[0] in ("wait",):
                reward -= 0.4  # extra penalty for triple repeat

        # Small time pressure: each step has a tiny cost
        reward -= 0.05

        return round(reward, 4)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def done(self) -> bool:
        return self._done

    @property
    def action_history(self) -> list[str]:
        return list(self._action_history)
