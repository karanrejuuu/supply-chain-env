"""
Pydantic models for the Supply Chain Disruption Environment.

Defines the Action and Observation schemas used by the environment,
simulator, grader, and inference scripts.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class Action(BaseModel):
    """Agent action in the supply chain environment."""

    action_type: str = Field(
        ...,
        description="One of: reroute, expedite, reallocate, wait",
        pattern=r"^(reroute|expedite|reallocate|wait)$",
    )
    route_id: Optional[str] = Field(
        default=None, description="Target route for reroute actions"
    )
    supplier_id: Optional[str] = Field(
        default=None, description="Target supplier for expedite actions"
    )
    warehouse_id: Optional[str] = Field(
        default=None, description="Target warehouse for reallocate actions"
    )
    quantity: Optional[int] = Field(
        default=None, ge=0, description="Units to move/order"
    )


class DisruptionEvent(BaseModel):
    """An active disruption in the supply chain."""

    disruption_type: str = Field(
        ..., description="Type: supplier_delay, route_block, demand_spike"
    )
    target_id: str = Field(..., description="ID of the affected entity")
    severity: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Severity 0.0-1.0"
    )
    remaining_steps: int = Field(
        default=3, ge=0, description="Steps until auto-resolve (0 = permanent until fixed)"
    )


class OrderItem(BaseModel):
    """A pending customer order."""

    order_id: str
    destination_warehouse: str
    quantity: int = Field(ge=1)
    deadline: int = Field(ge=1, description="Must be fulfilled by this time step")
    fulfilled: bool = False


class Observation(BaseModel):
    """Observable state returned to the agent each step."""

    inventory_levels: dict[str, int] = Field(
        ..., description="Inventory per warehouse"
    )
    pending_orders: list[dict] = Field(
        ..., description="List of pending order dicts"
    )
    disruption_status: list[dict] = Field(
        ..., description="List of active disruption dicts"
    )
    cost_so_far: float = Field(default=0.0, ge=0.0)
    time_step: int = Field(default=0, ge=0)
    supplier_info: dict[str, dict] = Field(
        default_factory=dict,
        description="Supplier id -> {capacity, reliability, active}",
    )
    route_info: dict[str, dict] = Field(
        default_factory=dict,
        description="Route id -> {from, to, cost, delay_prob, active}",
    )
