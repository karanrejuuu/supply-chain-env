import copy
from env.models import Observation, Action
from env.tasks import TASKS


class SupplyChainEnv:
    """
    OpenEnv-compatible environment simulating supply chain disruptions.
    The agent must recover from disruptions (supplier delays, transport blockages,
    low inventory) within a configurable number of steps.
    """

    MAX_STEPS: int = 20

    def __init__(self, task_name: str = "hard") -> None:
        if task_name not in TASKS:
            raise ValueError(
                f"Unknown task '{task_name}'. Choose from: {list(TASKS.keys())}"
            )
        self._task_name = task_name
        self._task = TASKS[task_name]
        self._optimal: list[str] = self._task["optimal_solution"]
        self._optimal_index: int = 0
        self._state: Observation = copy.deepcopy(self._task["initial_state"])
        self._done: bool = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self) -> Observation:
        """Reset to the initial state of the current task."""
        self._state = copy.deepcopy(self._task["initial_state"])
        self._done = False
        self._optimal_index = 0
        return copy.deepcopy(self._state)

    def step(self, action: Action) -> tuple[Observation, float, bool, dict]:
        """
        Apply action, compute reward, advance state.

        Returns
        -------
        observation : Observation
        reward      : float
        done        : bool
        info        : dict
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        reward = self._compute_reward(action)
        self._apply_action(action)

        self._state.time_elapsed += 1

        # Terminal condition
        if self._state.inventory >= self._state.demand:
            self._done = True
            reward += 1.0  # terminal reward

        if self._state.time_elapsed >= self.MAX_STEPS:
            self._done = True

        info = {
            "task": self._task_name,
            "time_elapsed": self._state.time_elapsed,
            "optimal_index": self._optimal_index,
        }

        return copy.deepcopy(self._state), reward, self._done, info

    def state(self) -> Observation:
        """Return the current observation (read-only copy)."""
        return copy.deepcopy(self._state)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_action(self, action: Action) -> None:
        act = action.action_type

        if act == "order_more_stock":
            self._state.inventory += 30
            self._state.cost += 50.0

        elif act == "switch_supplier":
            self._state.supplier_status = "ok"
            self._state.cost += 30.0

        elif act == "reroute_transport":
            self._state.transport_status = "ok"
            self._state.cost += 20.0

        elif act == "delay_order":
            pass  # no state improvement

        elif act == "do_nothing":
            pass  # no state improvement

    def _compute_reward(self, action: Action) -> float:
        act = action.action_type

        # Penalty actions — evaluated first regardless of optimal index
        if act == "delay_order":
            return -0.2
        if act == "do_nothing":
            return -0.3

        reward = 0.0

        # Check if this matches the next expected optimal action
        if self._optimal_index < len(self._optimal):
            if act == self._optimal[self._optimal_index]:
                reward += 0.3
                self._optimal_index += 1
            else:
                reward -= 0.3

        # Check if the action improves system state
        if self._is_improving(act):
            reward += 0.2

        return reward

    def _is_improving(self, act: str) -> bool:
        """Return True if the action meaningfully improves current state."""
        if act == "order_more_stock":
            return self._state.inventory < self._state.demand
        if act == "switch_supplier":
            return self._state.supplier_status == "delayed"
        if act == "reroute_transport":
            return self._state.transport_status == "blocked"
        return False
