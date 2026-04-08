from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field

class Action(BaseModel):
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

class Observation(BaseModel):
    inventory_levels: dict[str, int] = Field(...)
    pending_orders: list[dict] = Field(...)
    disruption_status: list[dict] = Field(...)
    cost_so_far: float = Field(default=0.0, ge=0.0)
    time_step: int = Field(default=0, ge=0)
    supplier_info: dict[str, dict] = Field(default_factory=dict)
    route_info: dict[str, dict] = Field(default_factory=dict)
