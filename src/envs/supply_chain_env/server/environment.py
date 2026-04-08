import uuid
import copy
from typing import Any

# OpenEnv Environment base abstraction
try:
    from core.env_server import Environment as BaseEnvironment
except ImportError:
    class BaseEnvironment: pass

from .simulator import SupplyChainSimulator
from .tasks import TASKS, TaskDefinition
from .grader import grade
from ..models import SupplyChainAction, SupplyChainObservation, SupplyChainState

class SupplyChainEnvironment(BaseEnvironment):
    """
    OpenEnv-compatible microservice environment for supply chains.
    """
    def __init__(self, task_name: str = "hard") -> None:
        if task_name not in TASKS:
            task_name = "hard"
        self._task_name = task_name
        self._task = TASKS[task_name]
        self._sim = SupplyChainSimulator()
        self._done = False
        self._step_count = 0
        self._cumulative_reward = 0.0
        self._action_history = []
        self._episode_id = str(uuid.uuid4())
        self._last_reward = 0.0
        
        # Initialize simulator implicitly
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

        pydantic_obs = self._sim.reset(
            seed=self._task.seed,
            warehouses=self._task.warehouses,
            suppliers=self._task.suppliers,
            routes=self._task.routes,
            orders=self._task.orders,
            disruptions=self._task.disruptions,
        )
        return self._build_obs(pydantic_obs)

    def step(self, action: SupplyChainAction) -> SupplyChainObservation:
        if self._done:
            obs = self._sim.get_observation()
            return self._build_obs(obs)

        self._step_count += 1
        self._action_history.append(action.action_type)
        
        # OpenEnv Actions are dataclasses, our internal Simulator uses Pydantic.
        from .simulator_models import Action as PydanticAction
        pydantic_action = PydanticAction(
            action_type=action.action_type,
            route_id=action.route_id,
            supplier_id=action.supplier_id,
            warehouse_id=action.warehouse_id,
            quantity=action.quantity
        )

        cost_before = self._sim.total_cost
        action_reward = self._sim.apply_action(pydantic_action)
        time_reward = self._sim.step_time()
        step_cost = self._sim.total_cost - cost_before

        reward = self._shape_reward(action, action_reward, time_reward, step_cost)
        self._cumulative_reward += reward
        self._last_reward = reward

        if self._sim.all_orders_done:
            self._done = True
            self._cumulative_reward += 2.0
            self._last_reward += 2.0
        elif self._step_count >= self._task.max_steps:
            self._done = True
            unfulfilled = len(self._sim.orders)
            self._cumulative_reward -= unfulfilled * 0.5
            self._last_reward -= unfulfilled * 0.5

        pydantic_obs = self._sim.get_observation()
        return self._build_obs(pydantic_obs)

    def _build_obs(self, obs) -> SupplyChainObservation:
        return SupplyChainObservation(
            inventory_levels=obs.inventory_levels,
            pending_orders=obs.pending_orders,
            disruption_status=obs.disruption_status,
            cost_so_far=obs.cost_so_far,
            time_step=obs.time_step,
            supplier_info=obs.supplier_info,
            route_info=obs.route_info,
            done=self._done,
            reward=self._last_reward
        )

    @property
    def state(self) -> SupplyChainState:
        metrics = self._sim.get_metrics()
        metrics["max_steps"] = self._task.max_steps
        info = {
            "task": self._task_name,
            "step": self._step_count,
            "cumulative_reward": self._cumulative_reward,
            **metrics
        }
        score = grade(info)

        return SupplyChainState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            task_name=self._task_name,
            total_cost=self._sim.total_cost,
            orders_fulfilled=self._sim.orders_fulfilled,
            total_orders=self._sim.total_orders,
            score=score
        )
        
    def _shape_reward(self, action, action_reward, time_reward, step_cost):
        reward = action_reward + time_reward
        if step_cost > 40: reward -= 0.3
        elif step_cost > 20: reward -= 0.1
        elif 0 < step_cost <= 10: reward += 0.1
        if len(self._action_history) >= 3:
            last3 = self._action_history[-3:]
            if last3[0] == last3[1] == last3[2] and last3[0] == "wait":
                reward -= 0.4
        reward -= 0.05
        return round(reward, 4)
