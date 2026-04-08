from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel, Field


class SupplyChainAction(BaseModel):
    action_type: str = Field(..., pattern=r"^(reroute|expedite|reallocate|wait)$")
    route_id: Optional[str] = Field(default=None)
    supplier_id: Optional[str] = Field(default=None)
    warehouse_id: Optional[str] = Field(default=None)
    quantity: Optional[int] = Field(default=None, ge=0)


class DisruptionEvent(BaseModel):
    disruption_type: str = Field(...)
    target_id: str = Field(...)
    severity: float = Field(default=1.0, ge=0.0, le=1.0)
    remaining_steps: int = Field(default=3, ge=0)


class OrderItem(BaseModel):
    order_id: str
    destination_warehouse: str
    quantity: int = Field(ge=1)
    deadline: int = Field(ge=1)
    fulfilled: bool = False
    missed: bool = False


class SupplyChainObservation(BaseModel):
    inventory_levels: dict[str, int] = Field(...)
    pending_orders: list[OrderItem] = Field(default_factory=list)
    disruption_status: list[DisruptionEvent] = Field(default_factory=list)
    cost_so_far: float = Field(default=0.0, ge=0.0)
    time_step: int = Field(default=0, ge=0)
    task_name: Optional[str] = Field(default=None)
    max_steps: int = Field(default=0, ge=0)
    orders_fulfilled: int = Field(default=0, ge=0)
    orders_missed: int = Field(default=0, ge=0)
    total_orders: int = Field(default=0, ge=0)
    supplier_info: dict[str, dict] = Field(default_factory=dict)
    route_info: dict[str, dict] = Field(default_factory=dict)
    done: bool = Field(default=False)
    reward: float = Field(default=0.0)


class SupplyChainState(BaseModel):
    episode_id: Optional[str] = None
    step_count: int
    task_name: str
    total_cost: float
    orders_fulfilled: int
    orders_missed: int
    total_orders: int
    score: Optional[float] = None


class SupplyChainReward(BaseModel):
    value: float = Field(default=0.0, ge=0.0, le=1.0)
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    efficiency: float = Field(default=0.0, ge=0.0, le=1.0)
    resilience: float = Field(default=0.0, ge=0.0, le=1.0)
    penalties: float = Field(default=0.0, ge=0.0)


@dataclass(frozen=True)
class StepResult:
    observation: SupplyChainObservation
    reward: float
    done: bool
