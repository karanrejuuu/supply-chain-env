from env.models import Observation
from typing import TypedDict


class Task(TypedDict):
    initial_state: Observation
    optimal_solution: list[str]
    description: str


TASKS: dict[str, Task] = {
    "easy": Task(
        initial_state=Observation(
            inventory=10,
            demand=50,
            supplier_status="ok",
            transport_status="ok",
            time_elapsed=0,
            cost=0.0,
        ),
        optimal_solution=["order_more_stock"],
        description=(
            "Simple stock-out scenario: inventory is well below demand but "
            "both supplier and transport are operational. A single reorder resolves the situation."
        ),
    ),
    "medium": Task(
        initial_state=Observation(
            inventory=10,
            demand=50,
            supplier_status="delayed",
            transport_status="ok",
            time_elapsed=0,
            cost=0.0,
        ),
        optimal_solution=["switch_supplier", "order_more_stock"],
        description=(
            "Supplier disruption scenario: primary supplier is delayed. "
            "The agent must first switch to an alternate supplier before restocking."
        ),
    ),
    "hard": Task(
        initial_state=Observation(
            inventory=5,
            demand=80,
            supplier_status="delayed",
            transport_status="blocked",
            time_elapsed=0,
            cost=0.0,
        ),
        optimal_solution=["switch_supplier", "reroute_transport", "order_more_stock"],
        description=(
            "Full disruption scenario: critically low inventory, primary supplier delayed, "
            "and transport route blocked. Requires coordinated multi-step recovery."
        ),
    ),
}
