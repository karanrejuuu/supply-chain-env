"""
SupplyChainSimulator — core simulation engine.

Manages warehouses, suppliers, routes, orders, and disruptions.
Uses seeded RNG for deterministic replay.
"""

from __future__ import annotations

import copy
import random
from typing import Any

from env.models import Action, DisruptionEvent, OrderItem, Observation


class SupplyChainSimulator:
    """
    Deterministic supply chain simulation with:
    - Multi-warehouse inventory tracking
    - Supplier capacity & reliability
    - Route cost & delay modelling
    - Order fulfilment lifecycle
    - Disruption injection & resolution
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

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

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
        self.orders = (
            [OrderItem(**o) for o in orders] if orders else []
        )
        self.disruptions = (
            [DisruptionEvent(**d) for d in disruptions] if disruptions else []
        )
        self.total_orders = len(self.orders)

        return self._build_observation()

    # ------------------------------------------------------------------
    # Core step
    # ------------------------------------------------------------------

    def apply_action(self, action: Action) -> float:
        """
        Apply an agent action. Returns incremental reward.
        """
        reward = 0.0
        act = action.action_type

        # Track repeated wait
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
        Returns incremental reward from time-step events.
        """
        self.time_step += 1
        reward = 0.0

        # --- Tick disruptions ---
        surviving: list[DisruptionEvent] = []
        for d in self.disruptions:
            if d.remaining_steps > 0:
                d.remaining_steps -= 1
                if d.remaining_steps > 0:
                    surviving.append(d)
                # else: auto-resolved
            else:
                surviving.append(d)  # permanent until agent fixes it
        self.disruptions = surviving

        # --- Supplier replenishment ---
        # Active, non-disrupted suppliers deliver inventory each step
        wh_list = list(self.warehouses.keys())
        for sid, sup in self.suppliers.items():
            if not sup.get("active", True):
                continue
            if self._supplier_disrupted(sid):
                continue
            # Deliver a fraction of capacity (reliability-scaled)
            capacity = sup.get("capacity", 0)
            reliability = sup.get("reliability", 1.0)
            delivery = int(capacity * reliability * 0.1)  # 10% per step
            if delivery > 0 and wh_list:
                target_wh = wh_list[0]  # default: first warehouse
                self.warehouses[target_wh] += delivery

        # --- Route deliveries ---
        for rid, route in self.routes.items():
            if not route.get("active", True):
                continue
            # Check if route is disrupted
            if self._route_disrupted(rid):
                continue
            # Deliver goods: move from source to destination warehouse
            src = route.get("from_warehouse")
            dst = route.get("to_warehouse")
            flow = route.get("flow_rate", 5)
            if src and dst and src in self.warehouses and dst in self.warehouses:
                transfer = min(self.warehouses[src], flow)
                if transfer > 0:
                    self.warehouses[src] -= transfer
                    self.warehouses[dst] += transfer

        # --- Check order fulfilment / deadlines ---
        still_pending: list[OrderItem] = []
        for order in self.orders:
            if order.fulfilled:
                continue
            wh = order.destination_warehouse
            if wh in self.warehouses and self.warehouses[wh] >= order.quantity:
                # Fulfil order
                self.warehouses[wh] -= order.quantity
                order.fulfilled = True
                self.orders_fulfilled += 1
                reward += 1.0  # positive reward for fulfilling
            elif self.time_step >= order.deadline:
                # Missed deadline
                self.orders_missed += 1
                reward -= 1.5  # penalty for missed order
            else:
                still_pending.append(order)
        self.orders = still_pending

        return reward

    def inject_disruption(self, disruption_type: str, target_id: str,
                          severity: float = 1.0, remaining_steps: int = 0) -> None:
        """Inject a new disruption event into the simulation."""
        self.disruptions.append(
            DisruptionEvent(
                disruption_type=disruption_type,
                target_id=target_id,
                severity=severity,
                remaining_steps=remaining_steps,
            )
        )
        # Apply immediate effects
        if disruption_type == "supplier_delay":
            if target_id in self.suppliers:
                self.suppliers[target_id]["active"] = False
        elif disruption_type == "route_block":
            if target_id in self.routes:
                self.routes[target_id]["active"] = False

    def compute_costs(self) -> dict[str, float]:
        """Return cost breakdown."""
        return {
            "total_cost": self.total_cost,
            "orders_fulfilled": self.orders_fulfilled,
            "orders_missed": self.orders_missed,
            "fulfillment_rate": (
                self.orders_fulfilled / self.total_orders
                if self.total_orders > 0
                else 1.0
            ),
        }

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _build_observation(self) -> Observation:
        return Observation(
            inventory_levels=copy.deepcopy(self.warehouses),
            pending_orders=[
                o.model_dump() for o in self.orders if not o.fulfilled
            ],
            disruption_status=[d.model_dump() for d in self.disruptions],
            cost_so_far=self.total_cost,
            time_step=self.time_step,
            supplier_info=copy.deepcopy(self.suppliers),
            route_info=copy.deepcopy(self.routes),
        )

    def get_observation(self) -> Observation:
        """Public accessor for current observation."""
        return self._build_observation()

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _handle_reroute(self, action: Action) -> float:
        """Reroute: activate a blocked route or switch to alternate route."""
        rid = action.route_id
        if not rid or rid not in self.routes:
            return -0.3  # invalid action penalty

        # Check if there's actually a disruption on this route
        resolved = False
        remaining = []
        for d in self.disruptions:
            if d.disruption_type == "route_block" and d.target_id == rid:
                resolved = True
            else:
                remaining.append(d)

        if resolved:
            self.disruptions = remaining
            self.routes[rid]["active"] = True
            cost = self.routes[rid].get("cost", 20)
            self.total_cost += cost
            return 0.5  # good action: fixed a real problem
        else:
            self.total_cost += 10  # wasted resources
            return -0.2  # unnecessary action

    def _handle_expedite(self, action: Action) -> float:
        """Expedite: reactivate a delayed supplier at extra cost."""
        sid = action.supplier_id
        if not sid or sid not in self.suppliers:
            return -0.3  # invalid action

        resolved = False
        remaining = []
        for d in self.disruptions:
            if d.disruption_type == "supplier_delay" and d.target_id == sid:
                resolved = True
            else:
                remaining.append(d)

        if resolved:
            self.disruptions = remaining
            self.suppliers[sid]["active"] = True
            cost = 30 * self.suppliers[sid].get("capacity", 1) / 100
            self.total_cost += cost
            return 0.5
        else:
            self.total_cost += 15
            return -0.2

    def _handle_reallocate(self, action: Action) -> float:
        """Reallocate: move inventory from one warehouse to another."""
        wid = action.warehouse_id
        qty = action.quantity or 0
        if not wid or wid not in self.warehouses or qty <= 0:
            return -0.3

        # Find a source warehouse with surplus (pick highest inventory)
        source = None
        best_inv = -1
        for w, inv in self.warehouses.items():
            if w != wid and inv >= qty and inv > best_inv:
                source = w
                best_inv = inv

        if source is None:
            return -0.2  # no source available

        self.warehouses[source] -= qty
        self.warehouses[wid] += qty
        transport_cost = qty * 0.5
        self.total_cost += transport_cost
        return 0.3  # partial reward for proactive rebalancing

    def _handle_wait(self) -> float:
        """Wait: do nothing. Penalise repeated waits."""
        if self._repeated_wait_count >= 2:
            return -0.5  # escalating penalty for repeated useless waits
        if self._repeated_wait_count == 1:
            return -0.3
        return -0.1  # mild penalty: time is wasting

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _route_disrupted(self, route_id: str) -> bool:
        for d in self.disruptions:
            if d.disruption_type == "route_block" and d.target_id == route_id:
                return True
        return False

    def _supplier_disrupted(self, supplier_id: str) -> bool:
        for d in self.disruptions:
            if d.disruption_type == "supplier_delay" and d.target_id == supplier_id:
                return True
        return False

    @property
    def all_orders_done(self) -> bool:
        """True when no pending unfulfilled orders remain."""
        return len(self.orders) == 0

    def get_metrics(self) -> dict:
        """Return full metrics dict for grading."""
        return {
            "orders_fulfilled": self.orders_fulfilled,
            "orders_missed": self.orders_missed,
            "total_orders": self.total_orders,
            "fulfillment_rate": (
                self.orders_fulfilled / self.total_orders
                if self.total_orders > 0
                else 1.0
            ),
            "total_cost": self.total_cost,
            "time_step": self.time_step,
        }
