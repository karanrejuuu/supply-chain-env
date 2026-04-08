"""
SupplyChainSimulator - core simulation engine.

Manages warehouses, suppliers, routes, orders, and disruptions.
Uses seeded RNG for deterministic replay.
"""

from __future__ import annotations

import copy
import random
from typing import Any

from ..models import (
    DisruptionEvent,
    OrderItem,
    SupplyChainAction as Action,
    SupplyChainObservation as Observation,
)


class SupplyChainSimulator:
    """
    Deterministic supply chain simulation with:
    - Multi-warehouse inventory tracking
    - Supplier capacity and reliability
    - Route cost and delay modeling
    - Order fulfillment lifecycle
    - Disruption injection and resolution
    """

    def __init__(self) -> None:
        self.warehouses: dict[str, int] = {}
        self.suppliers: dict[str, dict[str, Any]] = {}
        self.routes: dict[str, dict[str, Any]] = {}
        self.orders: list[OrderItem] = []
        self.disruptions: list[DisruptionEvent] = []
        self.total_cost: float = 0.0
        self.time_step: int = 0
        self.orders_fulfilled: int = 0
        self.orders_missed: int = 0
        self.total_orders: int = 0
        self._rng: random.Random = random.Random(0)
        self._prev_action: str | None = None
        self._repeated_wait_count: int = 0

    def reset(
        self,
        seed: int = 42,
        warehouses: dict[str, int] | None = None,
        suppliers: dict[str, dict[str, Any]] | None = None,
        routes: dict[str, dict[str, Any]] | None = None,
        orders: list[dict] | None = None,
        disruptions: list[dict] | None = None,
    ) -> Observation:
        """Reset simulator to a given initial state."""
        self._rng = random.Random(seed)
        self.time_step = 0
        self.total_cost = 0.0
        self.orders_fulfilled = 0
        self.orders_missed = 0
        self._prev_action = None
        self._repeated_wait_count = 0

        self.warehouses = copy.deepcopy(warehouses) if warehouses else {}
        self.suppliers = copy.deepcopy(suppliers) if suppliers else {}
        self.routes = copy.deepcopy(routes) if routes else {}
        self.orders = [OrderItem(**order) for order in orders] if orders else []
        self.disruptions = [DisruptionEvent(**event) for event in disruptions] if disruptions else []
        self.total_orders = len(self.orders)
        return self._build_observation()

    def apply_action(self, action: Action) -> float:
        """Apply an agent action and return incremental reward."""
        reward = 0.0
        act = action.action_type

        if act == "wait":
            if self._prev_action == "wait":
                self._repeated_wait_count += 1
            else:
                self._repeated_wait_count = 0
        else:
            self._repeated_wait_count = 0
        self._prev_action = act

        if act == "reroute":
            reward += self._handle_reroute(action)
        elif act == "expedite":
            reward += self._handle_expedite(action)
        elif act == "reallocate":
            reward += self._handle_reallocate(action)
        elif act == "wait":
            reward += self._handle_wait()

        return reward

    def step_time(self) -> float:
        """
        Advance simulation by one time step.

        - Tick down disruption timers
        - Process deliveries via active routes
        - Check order deadlines
        """
        self.time_step += 1
        reward = 0.0

        surviving: list[DisruptionEvent] = []
        for disruption in self.disruptions:
            if disruption.remaining_steps > 0:
                disruption.remaining_steps -= 1
                if disruption.remaining_steps > 0:
                    surviving.append(disruption)
            else:
                surviving.append(disruption)
        self.disruptions = surviving

        warehouse_ids = list(self.warehouses.keys())
        for supplier_id, supplier in self.suppliers.items():
            if not supplier.get("active", True):
                continue
            if self._supplier_disrupted(supplier_id):
                continue

            capacity = supplier.get("capacity", 0)
            reliability = supplier.get("reliability", 1.0)
            delivery = int(capacity * reliability * 0.1)
            if delivery > 0 and warehouse_ids:
                self.warehouses[warehouse_ids[0]] += delivery

        for route_id, route in self.routes.items():
            if not route.get("active", True):
                continue
            if self._route_disrupted(route_id):
                continue

            source = route.get("from_warehouse")
            destination = route.get("to_warehouse")
            flow_rate = route.get("flow_rate", 5)
            if source and destination and source in self.warehouses and destination in self.warehouses:
                transfer = min(self.warehouses[source], flow_rate)
                if transfer > 0:
                    self.warehouses[source] -= transfer
                    self.warehouses[destination] += transfer

        for order in self.orders:
            if order.fulfilled or order.missed:
                continue

            destination = order.destination_warehouse
            if destination in self.warehouses and self.warehouses[destination] >= order.quantity:
                self.warehouses[destination] -= order.quantity
                order.fulfilled = True
                self.orders_fulfilled += 1
                reward += 1.0
            elif self.time_step >= order.deadline:
                order.missed = True
                self.orders_missed += 1
                reward -= 1.5

        return reward

    def inject_disruption(
        self,
        disruption_type: str,
        target_id: str,
        severity: float = 1.0,
        remaining_steps: int = 0,
    ) -> None:
        """Inject a new disruption event into the simulation."""
        self.disruptions.append(
            DisruptionEvent(
                disruption_type=disruption_type,
                target_id=target_id,
                severity=severity,
                remaining_steps=remaining_steps,
            )
        )
        if disruption_type == "supplier_delay" and target_id in self.suppliers:
            self.suppliers[target_id]["active"] = False
        elif disruption_type == "route_block" and target_id in self.routes:
            self.routes[target_id]["active"] = False

    def compute_costs(self) -> dict[str, float]:
        """Return cost breakdown."""
        return {
            "total_cost": self.total_cost,
            "orders_fulfilled": self.orders_fulfilled,
            "orders_missed": self.orders_missed,
            "fulfillment_rate": (
                self.orders_fulfilled / self.total_orders if self.total_orders > 0 else 1.0
            ),
        }

    def _build_observation(self) -> Observation:
        return Observation(
            inventory_levels=copy.deepcopy(self.warehouses),
            pending_orders=[order for order in self.orders if not order.fulfilled and not order.missed],
            disruption_status=copy.deepcopy(self.disruptions),
            cost_so_far=self.total_cost,
            time_step=self.time_step,
            orders_fulfilled=self.orders_fulfilled,
            orders_missed=self.orders_missed,
            total_orders=self.total_orders,
            supplier_info=copy.deepcopy(self.suppliers),
            route_info=copy.deepcopy(self.routes),
        )

    def get_observation(self) -> Observation:
        return self._build_observation()

    def _handle_reroute(self, action: Action) -> float:
        route_id = action.route_id
        if not route_id or route_id not in self.routes:
            return -0.3

        resolved = False
        remaining: list[DisruptionEvent] = []
        for disruption in self.disruptions:
            if disruption.disruption_type == "route_block" and disruption.target_id == route_id:
                resolved = True
            else:
                remaining.append(disruption)

        if resolved:
            self.disruptions = remaining
            self.routes[route_id]["active"] = True
            self.total_cost += self.routes[route_id].get("cost", 20)
            return 0.5

        self.total_cost += 10
        return -0.2

    def _handle_expedite(self, action: Action) -> float:
        supplier_id = action.supplier_id
        if not supplier_id or supplier_id not in self.suppliers:
            return -0.3

        resolved = False
        remaining: list[DisruptionEvent] = []
        for disruption in self.disruptions:
            if disruption.disruption_type == "supplier_delay" and disruption.target_id == supplier_id:
                resolved = True
            else:
                remaining.append(disruption)

        if resolved:
            self.disruptions = remaining
            self.suppliers[supplier_id]["active"] = True
            self.total_cost += 30 * self.suppliers[supplier_id].get("capacity", 1) / 100
            return 0.5

        self.total_cost += 15
        return -0.2

    def _handle_reallocate(self, action: Action) -> float:
        warehouse_id = action.warehouse_id
        quantity = action.quantity or 0
        if not warehouse_id or warehouse_id not in self.warehouses or quantity <= 0:
            return -0.3

        source = None
        best_inventory = -1
        for warehouse_name, inventory in self.warehouses.items():
            if warehouse_name != warehouse_id and inventory >= quantity and inventory > best_inventory:
                source = warehouse_name
                best_inventory = inventory

        if source is None:
            return -0.2

        self.warehouses[source] -= quantity
        self.warehouses[warehouse_id] += quantity
        self.total_cost += quantity * 0.5
        return 0.3

    def _handle_wait(self) -> float:
        if self._repeated_wait_count >= 2:
            return -0.5
        if self._repeated_wait_count == 1:
            return -0.3
        return -0.1

    def _route_disrupted(self, route_id: str) -> bool:
        return any(
            disruption.disruption_type == "route_block" and disruption.target_id == route_id
            for disruption in self.disruptions
        )

    def _supplier_disrupted(self, supplier_id: str) -> bool:
        return any(
            disruption.disruption_type == "supplier_delay" and disruption.target_id == supplier_id
            for disruption in self.disruptions
        )

    @property
    def all_orders_fulfilled(self) -> bool:
        """True when every order has been fulfilled successfully."""
        return self.orders_fulfilled == self.total_orders and self.total_orders > 0

    def get_metrics(self) -> dict[str, float]:
        return {
            "orders_fulfilled": self.orders_fulfilled,
            "orders_missed": self.orders_missed,
            "total_orders": self.total_orders,
            "fulfillment_rate": (
                self.orders_fulfilled / self.total_orders if self.total_orders > 0 else 1.0
            ),
            "total_cost": self.total_cost,
            "time_step": self.time_step,
        }
