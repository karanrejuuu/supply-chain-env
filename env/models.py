from pydantic import BaseModel, field_validator
from typing import Literal


class Observation(BaseModel):
    inventory: int
    demand: int
    supplier_status: Literal["ok", "delayed"]
    transport_status: Literal["ok", "blocked"]
    time_elapsed: int
    cost: float


class Action(BaseModel):
    action_type: Literal[
        "order_more_stock",
        "switch_supplier",
        "reroute_transport",
        "delay_order",
        "do_nothing",
    ]
