from pydantic import BaseModel
from typing import Literal, Optional


class Observation(BaseModel):
    order_id: str

    # Core
    inventory_level: int
    demand: int
    backlog: int

    # Supply risk
    supplier_delay_days: int
    supplier_reliability: float

    # Logistics
    transport_available: bool
    transport_cost_multiplier: float

    # Financial
    cost_incurred: float
    budget_remaining: float

    # Time
    time_elapsed: int
    max_time: int

    # Uncertainty
    demand_forecast: int


class Action(BaseModel):
    action_type: Literal[
        "order_more_stock",
        "switch_supplier",
        "reroute_transport",
        "delay_order",
        "do_nothing"
    ]
    order_quantity: int = 20
    supplier_choice: str = "standard"
    transport_mode: str = "normal"


class Reward(BaseModel):
    value: float