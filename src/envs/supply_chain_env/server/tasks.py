"""
Task definitions for the Supply Chain Disruption Environment.

Three difficulty tiers — easy, medium, hard — each with:
- initial simulator state
- success condition
- max steps
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TaskDefinition:
    """Immutable specification of a single evaluation task."""

    name: str
    description: str
    max_steps: int
    seed: int
    warehouses: dict[str, int]
    suppliers: dict[str, dict[str, Any]]
    routes: dict[str, dict[str, Any]]
    orders: list[dict[str, Any]]
    disruptions: list[dict[str, Any]]
    success_description: str


# ======================================================================
# EASY — Single supplier delay
# ======================================================================
EASY_TASK = TaskDefinition(
    name="easy",
    description=(
        "Single supplier delay: your primary supplier is delayed, "
        "causing a shortage at warehouse-A. One disruption to fix, "
        "then fulfil all orders before their deadlines."
    ),
    max_steps=10,
    seed=100,
    warehouses={"warehouse-A": 40, "warehouse-B": 60},
    suppliers={
        "supplier-1": {"capacity": 100, "reliability": 0.5, "active": False},
        "supplier-2": {"capacity": 80, "reliability": 0.9, "active": True},
    },
    routes={
        "route-1": {
            "from_warehouse": "warehouse-B",
            "to_warehouse": "warehouse-A",
            "cost": 15,
            "delay_prob": 0.1,
            "flow_rate": 10,
            "active": True,
        },
    },
    orders=[
        {"order_id": "ORD-001", "destination_warehouse": "warehouse-A",
         "quantity": 30, "deadline": 8},
        {"order_id": "ORD-002", "destination_warehouse": "warehouse-A",
         "quantity": 20, "deadline": 9},
    ],
    disruptions=[
        {"disruption_type": "supplier_delay", "target_id": "supplier-1",
         "severity": 0.7, "remaining_steps": 0},
    ],
    success_description="Fulfil all orders with minimal delay and cost.",
)

# ======================================================================
# MEDIUM — Multi-warehouse imbalance
# ======================================================================
MEDIUM_TASK = TaskDefinition(
    name="medium",
    description=(
        "Multi-warehouse imbalance: warehouse-A is heavily stocked while "
        "warehouse-C is nearly empty. Orders target warehouse-C. "
        "The agent must reallocate inventory and manage transport decisions."
    ),
    max_steps=15,
    seed=200,
    warehouses={"warehouse-A": 120, "warehouse-B": 30, "warehouse-C": 5},
    suppliers={
        "supplier-1": {"capacity": 100, "reliability": 0.8, "active": True},
        "supplier-2": {"capacity": 60, "reliability": 0.7, "active": True},
    },
    routes={
        "route-1": {
            "from_warehouse": "warehouse-A",
            "to_warehouse": "warehouse-B",
            "cost": 10,
            "delay_prob": 0.1,
            "flow_rate": 8,
            "active": True,
        },
        "route-2": {
            "from_warehouse": "warehouse-B",
            "to_warehouse": "warehouse-C",
            "cost": 12,
            "delay_prob": 0.2,
            "flow_rate": 6,
            "active": True,
        },
        "route-3": {
            "from_warehouse": "warehouse-A",
            "to_warehouse": "warehouse-C",
            "cost": 25,
            "delay_prob": 0.3,
            "flow_rate": 10,
            "active": True,
        },
    },
    orders=[
        {"order_id": "ORD-101", "destination_warehouse": "warehouse-C",
         "quantity": 25, "deadline": 10},
        {"order_id": "ORD-102", "destination_warehouse": "warehouse-C",
         "quantity": 20, "deadline": 12},
        {"order_id": "ORD-103", "destination_warehouse": "warehouse-B",
         "quantity": 15, "deadline": 14},
    ],
    disruptions=[],
    success_description="Rebalance inventory across warehouses and fulfil all orders.",
)

# ======================================================================
# HARD — Cascading disruptions
# ======================================================================
HARD_TASK = TaskDefinition(
    name="hard",
    description=(
        "Cascading disruptions: supplier-1 has failed and route-2 is blocked. "
        "Inventory is low across all warehouses. Multiple orders are urgent. "
        "The agent must dynamically adapt — expediting the supplier, rerouting "
        "deliveries, and reallocating stock to meet demand."
    ),
    max_steps=20,
    seed=300,
    warehouses={"warehouse-A": 25, "warehouse-B": 10, "warehouse-C": 5},
    suppliers={
        "supplier-1": {"capacity": 150, "reliability": 0.6, "active": False},
        "supplier-2": {"capacity": 80, "reliability": 0.85, "active": True},
        "supplier-3": {"capacity": 50, "reliability": 0.9, "active": True},
    },
    routes={
        "route-1": {
            "from_warehouse": "warehouse-A",
            "to_warehouse": "warehouse-B",
            "cost": 10,
            "delay_prob": 0.1,
            "flow_rate": 8,
            "active": True,
        },
        "route-2": {
            "from_warehouse": "warehouse-B",
            "to_warehouse": "warehouse-C",
            "cost": 15,
            "delay_prob": 0.4,
            "flow_rate": 6,
            "active": False,
        },
        "route-3": {
            "from_warehouse": "warehouse-A",
            "to_warehouse": "warehouse-C",
            "cost": 30,
            "delay_prob": 0.2,
            "flow_rate": 10,
            "active": True,
        },
    },
    orders=[
        {"order_id": "ORD-201", "destination_warehouse": "warehouse-B",
         "quantity": 20, "deadline": 8},
        {"order_id": "ORD-202", "destination_warehouse": "warehouse-C",
         "quantity": 15, "deadline": 10},
        {"order_id": "ORD-203", "destination_warehouse": "warehouse-C",
         "quantity": 25, "deadline": 14},
        {"order_id": "ORD-204", "destination_warehouse": "warehouse-A",
         "quantity": 10, "deadline": 6},
    ],
    disruptions=[
        {"disruption_type": "supplier_delay", "target_id": "supplier-1",
         "severity": 1.0, "remaining_steps": 0},
        {"disruption_type": "route_block", "target_id": "route-2",
         "severity": 0.9, "remaining_steps": 0},
    ],
    success_description=(
        "Expedite the failed supplier, unblock routes, reallocate inventory, "
        "and fulfil all orders despite cascading failures."
    ),
)


# ======================================================================
# Task registry
# ======================================================================
TASKS: dict[str, TaskDefinition] = {
    "easy": EASY_TASK,
    "medium": MEDIUM_TASK,
    "hard": HARD_TASK,
}
