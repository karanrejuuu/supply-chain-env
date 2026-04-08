"""
Deterministic grader for the Supply Chain Disruption Environment.

Returns a score in [0.0, 1.0] based on:
  - fulfilment rate  (50%)
  - cost efficiency  (30%)
  - speed            (20%)

No randomness. Fully reproducible.
"""

from __future__ import annotations


def grade(metrics: dict) -> float:
    """
    Score an episode based on final metrics.

    Parameters
    ----------
    metrics : dict
        Must contain:
        - orders_fulfilled : int
        - total_orders     : int
        - total_cost       : float
        - time_step        : int   (steps used)
        - max_steps        : int   (budget for the task)

    Returns
    -------
    float in [0.0, 1.0]
    """
    total_orders = metrics.get("total_orders", 1)
    orders_fulfilled = metrics.get("orders_fulfilled", 0)
    total_cost = metrics.get("total_cost", 0.0)
    time_used = metrics.get("time_step", 0)
    max_steps = metrics.get("max_steps", 20)

    # --- Fulfilment rate (0.0 – 1.0) ---
    fulfillment_rate = (
        orders_fulfilled / total_orders if total_orders > 0 else 1.0
    )

    # --- Cost efficiency (0.0 – 1.0) ---
    # Lower cost is better. We use a reference max cost to normalise.
    # Reference: worst case ≈ max_steps * 50 (expensive actions every step)
    max_plausible_cost = max_steps * 50.0
    if max_plausible_cost > 0:
        cost_ratio = min(total_cost / max_plausible_cost, 1.0)
        cost_efficiency = 1.0 - cost_ratio
    else:
        cost_efficiency = 1.0

    # --- Speed (0.0 – 1.0) ---
    # Finishing in fewer steps is better.
    if max_steps > 0:
        speed = 1.0 - (time_used / max_steps)
    else:
        speed = 1.0

    # --- Weighted composite ---
    score = (
        0.5 * fulfillment_rate
        + 0.3 * cost_efficiency
        + 0.2 * max(speed, 0.0)
    )

    return round(max(0.0, min(1.0, score)), 4)
