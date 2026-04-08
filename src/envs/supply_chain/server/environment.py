import random
import uuid


TASK_CONFIGS = {
    "easy": {
        "demand": 30,
        "supplier_delay_days": 2,
        "budget": 500.0,
        "transport_available": True,
        "transport_cost_multiplier": 1.0,
        "supplier_reliability": 0.95,
        "max_time": 15,
        "initial_inventory": 25,
    },
    "medium": {
        "demand": 50,
        "supplier_delay_days": 3,
        "budget": 350.0,
        "transport_available": True,
        "transport_cost_multiplier": 1.5,
        "supplier_reliability": 0.75,
        "max_time": 12,
        "initial_inventory": 60,
    },
    "hard": {
        "demand": 80,
        "supplier_delay_days": 5,
        "budget": 200.0,
        "transport_available": False,
        "transport_cost_multiplier": 2.5,
        "supplier_reliability": 0.50,
        "max_time": 10,
        "initial_inventory": 40,
    },
}


class SupplyChainEnv:
    def __init__(self, task: str = "easy"):
        self.task = task if task in TASK_CONFIGS else "easy"
        self.cfg = TASK_CONFIGS[self.task]
        self.state = {}
        self.prev_observation = None
        self.steps = 0
        self.done = False
        self.stable_streak = 0

    def reset(self):
        cfg = self.cfg
        self.steps = 0
        self.done = False
        self.prev_observation = None
        self.stable_streak = 0
        demand = cfg["demand"]
        noise = random.randint(-5, 5)
        self.state = {
            "order_id": str(uuid.uuid4()),
            "inventory_level": cfg["initial_inventory"],
            "demand": demand,
            "backlog": 0,
            "supplier_delay_days": cfg["supplier_delay_days"],
            "supplier_reliability": cfg["supplier_reliability"],
            "transport_available": cfg["transport_available"],
            "transport_cost_multiplier": cfg["transport_cost_multiplier"],
            "cost_incurred": 0.0,
            "budget_remaining": cfg["budget"],
            "time_elapsed": 0,
            "max_time": cfg["max_time"],
            "demand_forecast": max(0, demand + noise),
        }
        return dict(self.state)

    def step(self, action: dict):
        if self.done:
            return dict(self.state), 0.0, True

        prev = self.prev_observation
        s = self.state
        action_type = action.get("action_type", "do_nothing")

        cost_delta = 0.0

        # --- Action Logic ---
        if action_type == "order_more_stock":
            qty = int(action.get("order_quantity", 20))
            qty = max(0, min(qty, 200))  # clamp
            if s["supplier_delay_days"] == 0:
                s["inventory_level"] += qty
            # cost always incurred
            cost_delta += qty * 2

        elif action_type == "switch_supplier":
            choice = action.get("supplier_choice", "standard")
            if choice == "fast":
                s["supplier_delay_days"] = 0
                cost_delta += 40
            else:
                s["supplier_delay_days"] = max(0, s["supplier_delay_days"] - 1)
                cost_delta += 10

        elif action_type == "reroute_transport":
            mode = action.get("transport_mode", "normal")
            s["transport_available"] = True
            if mode == "express":
                cost_delta += 30
                s["transport_cost_multiplier"] = max(1.0, s["transport_cost_multiplier"] - 0.5)
            else:
                cost_delta += 10

        elif action_type == "delay_order":
            s["backlog"] += max(1, int(s["demand"] * 0.2))
            cost_delta += 5

        elif action_type == "do_nothing":
            pass  # no change

        # --- Dynamics ---
        # Demand consumes inventory
        effective_supply = s["inventory_level"] if s["supplier_delay_days"] == 0 else max(0, s["inventory_level"] - 5)
        net = effective_supply - s["demand"]
        if net >= 0:
            s["inventory_level"] = net
        else:
            s["inventory_level"] = 0
            s["backlog"] += abs(net)

        # Reduce delay naturally each step
        if s["supplier_delay_days"] > 0:
            s["supplier_delay_days"] = max(0, s["supplier_delay_days"] - 1)

        # Apply cost multiplier to transport-related costs
        if not s["transport_available"]:
            cost_delta *= s["transport_cost_multiplier"]

        s["cost_incurred"] += cost_delta
        s["budget_remaining"] = max(0.0, s["budget_remaining"] - cost_delta)

        # Update time
        self.steps += 1
        s["time_elapsed"] = self.steps

        # Update forecast with small noise
        noise = random.randint(-3, 3)
        s["demand_forecast"] = max(0, s["demand"] + noise)

        # --- Reward Shaping ---
        self.state["step"] = self.steps
        observation = dict(self.state)

        inventory = observation["inventory_level"]
        backlog = observation["backlog"]
        delay = observation["supplier_delay_days"]
        demand = observation["demand_forecast"]
        reward = 0.0

        if prev is None:
            reward -= 0.5

        reward -= (backlog / 15.0) * 1.0
        reward -= (delay / 2.5) * 0.9

        if demand > 0 and inventory < demand:
            reward -= ((demand - inventory) / demand) * 0.9

        if action_type == "switch_supplier":
            if delay > 1:
                reward += 0.8
            else:
                reward -= 0.3

        elif action_type == "order_more_stock":
            if inventory < demand:
                reward += 0.6
            if backlog > 0:
                reward += 0.5

        elif action_type == "do_nothing":
            if backlog == 0 and delay <= 1:
                reward += 0.7
            else:
                reward -= 0.5

        if prev is not None:
            prev_backlog = prev["backlog"]
            prev_delay = prev["supplier_delay_days"]
            prev_inventory = prev["inventory_level"]

            if observation["backlog"] < prev_backlog:
                reward += 0.5

            if observation["supplier_delay_days"] < prev_delay:
                reward += 0.5

            if observation["inventory_level"] > prev_inventory:
                reward += 0.3

        # Light late-stage stability bonus for smoother demo progression.
        if backlog == 0:
            reward += 0.2
        if delay <= 1:
            reward += 0.1

        # Clamp reward
        reward = float(max(-1.0, min(1.0, reward)))

        # --- Done Condition ---
        self.done = False
        max_steps = 5
        is_stable = (
            observation.get("backlog", 0) == 0
            and observation.get("supplier_delay_days", 0) <= 1
            and observation.get("inventory_level", 0) >= observation.get("demand_forecast", 0)
        )

        if is_stable:
            self.done = True
        elif self.steps >= max_steps:
            self.done = True

        self.prev_observation = observation
        self.state["step"] = self.steps
        return dict(self.state), reward, self.done
