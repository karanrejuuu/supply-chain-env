from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union

# Attempt to import core OpenEnv abstractions
try:
    from core.env_server import Action, Observation, State
except ImportError:
    # Local fallback for type checking if openenv not fully installed
    class Action: pass
    class Observation: pass
    class State: pass

@dataclass
class SupplyChainAction(Action):
    action_type: str
    route_id: Optional[str] = None
    supplier_id: Optional[str] = None
    warehouse_id: Optional[str] = None
    quantity: Optional[int] = None

@dataclass
class SupplyChainObservation(Observation):
    inventory_levels: Dict[str, int]
    pending_orders: List[Dict[str, Any]]
    disruption_status: List[Dict[str, Any]]
    cost_so_far: float
    time_step: int
    supplier_info: Dict[str, Any]
    route_info: Dict[str, Any]
    done: bool
    reward: float

@dataclass
class SupplyChainState(State):
    episode_id: Optional[str]
    step_count: int
    task_name: str
    total_cost: float
    orders_fulfilled: int
    total_orders: int
    score: Optional[float] = None
