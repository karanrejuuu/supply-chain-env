import uuid

from .grader import grade
from .simulator import SupplyChainSimulator
from .tasks import TASKS
from ..models import SupplyChainAction, SupplyChainObservation, SupplyChainState


class SupplyChainEnvironment:
    """Stateful environment wrapper with reset/step semantics."""

    def __init__(self, task_name: str = "hard") -> None:
        if task_name not in TASKS:
            task_name = "hard"
        self._task_name = task_name
        self._task = TASKS[task_name]
        self._sim = SupplyChainSimulator()
        self._done = False
        self._step_count = 0
        self._cumulative_reward = 0.0
        self._action_history: list[str] = []
        self._episode_id = str(uuid.uuid4())
        self._last_reward = 0.0
        self._last_raw_reward = 0.0
        self._sim.reset(
            seed=self._task.seed,
            warehouses=self._task.warehouses,
            suppliers=self._task.suppliers,
            routes=self._task.routes,
            orders=self._task.orders,
            disruptions=self._task.disruptions,
        )

    def reset(self) -> SupplyChainObservation:
        self._done = False
        self._step_count = 0
        self._cumulative_reward = 0.0
        self._action_history = []
        self._episode_id = str(uuid.uuid4())
        self._last_reward = 0.0
        self._last_raw_reward = 0.0

        observation = self._sim.reset(
            seed=self._task.seed,
            warehouses=self._task.warehouses,
            suppliers=self._task.suppliers,
            routes=self._task.routes,
            orders=self._task.orders,
            disruptions=self._task.disruptions,
        )
        return self._build_obs(observation)

    def step(self, action: SupplyChainAction) -> SupplyChainObservation:
        if self._done:
            return self._build_obs(self._sim.get_observation())

        self._step_count += 1
        self._action_history.append(action.action_type)
        prev_fulfilled = self._sim.orders_fulfilled
        prev_missed = self._sim.orders_missed
        prev_disruptions = len(self._sim.disruptions)
        prev_pending = self._sim.total_orders - self._sim.orders_fulfilled - self._sim.orders_missed

        cost_before = self._sim.total_cost
        action_reward = self._sim.apply_action(action)
        time_reward = self._sim.step_time()
        step_cost = self._sim.total_cost - cost_before
        fulfilled_delta = self._sim.orders_fulfilled - prev_fulfilled
        missed_delta = self._sim.orders_missed - prev_missed
        disruptions_resolved = max(0, prev_disruptions - len(self._sim.disruptions))
        pending_now = self._sim.total_orders - self._sim.orders_fulfilled - self._sim.orders_missed
        pending_delta = max(0, prev_pending - pending_now)

        raw_reward = self._shape_reward(
            action=action,
            action_reward=action_reward,
            time_reward=time_reward,
            step_cost=step_cost,
            fulfilled_delta=fulfilled_delta,
            missed_delta=missed_delta,
            disruptions_resolved=disruptions_resolved,
            pending_delta=pending_delta,
            remaining_disruptions=len(self._sim.disruptions),
        )
        self._cumulative_reward += raw_reward
        self._last_raw_reward = raw_reward
        self._last_reward = self._normalize_reward(raw_reward)

        if self._sim.all_orders_fulfilled:
            self._done = True
            self._cumulative_reward += 2.0
            self._last_raw_reward += 2.0
            self._last_reward = self._normalize_reward(self._last_raw_reward)
        elif self._step_count >= self._task.max_steps:
            self._done = True
            unfulfilled = self._sim.total_orders - self._sim.orders_fulfilled
            penalty = unfulfilled * 0.5
            self._cumulative_reward -= penalty
            self._last_raw_reward -= penalty
            self._last_reward = self._normalize_reward(self._last_raw_reward)

        return self._build_obs(self._sim.get_observation())

    def _build_obs(self, observation: SupplyChainObservation) -> SupplyChainObservation:
        return SupplyChainObservation(
            inventory_levels=observation.inventory_levels,
            pending_orders=observation.pending_orders,
            disruption_status=observation.disruption_status,
            cost_so_far=observation.cost_so_far,
            time_step=observation.time_step,
            task_name=self._task_name,
            max_steps=self._task.max_steps,
            orders_fulfilled=self._sim.orders_fulfilled,
            orders_missed=self._sim.orders_missed,
            total_orders=self._sim.total_orders,
            supplier_info=observation.supplier_info,
            route_info=observation.route_info,
            done=self._done,
            reward=self._last_reward,
        )

    @property
    def state(self) -> SupplyChainState:
        metrics = self._sim.get_metrics()
        metrics["max_steps"] = self._task.max_steps
        score = grade(metrics)
        return SupplyChainState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            task_name=self._task_name,
            total_cost=self._sim.total_cost,
            orders_fulfilled=self._sim.orders_fulfilled,
            orders_missed=self._sim.orders_missed,
            total_orders=self._sim.total_orders,
            score=score,
        )

    def _shape_reward(
        self,
        action: SupplyChainAction,
        action_reward: float,
        time_reward: float,
        step_cost: float,
        fulfilled_delta: int,
        missed_delta: int,
        disruptions_resolved: int,
        pending_delta: int,
        remaining_disruptions: int,
    ) -> float:
        reward = action_reward + time_reward
        # Dense progress signals to improve learning stability.
        reward += 0.35 * fulfilled_delta
        reward -= 0.45 * missed_delta
        reward += 0.1 * pending_delta

        # Incentivize incident resolution through corrective actions.
        if disruptions_resolved > 0 and action.action_type in {"expedite", "reroute"}:
            reward += 0.2 * disruptions_resolved

        if step_cost > 40:
            reward -= 0.3
        elif step_cost > 20:
            reward -= 0.1
        elif 0 < step_cost <= 10:
            reward += 0.1

        if len(self._action_history) >= 3:
            last3 = self._action_history[-3:]
            if last3[0] == last3[1] == last3[2] and last3[0] == "wait":
                reward -= 0.4

        if action.action_type == "wait" and remaining_disruptions > 0:
            reward -= 0.2

        reward -= 0.05
        return round(reward, 4)

    @staticmethod
    def _normalize_reward(raw_reward: float) -> float:
        """
        Map internal reward (negative/positive) into public [0, 1] range.
        Chosen bounds cover observed shaping magnitudes with headroom.
        """
        min_raw = -2.0
        max_raw = 6.0
        scaled = (raw_reward - min_raw) / (max_raw - min_raw)
        return round(max(0.0, min(1.0, scaled)), 4)
