"""
Grader for the Supply Chain Disruption Agent environment.

grade() compares actions taken by an agent against the reference optimal
sequence and returns a deterministic score in [0.0, 1.0].
"""


def grade(actions_taken: list[str], optimal_actions: list[str]) -> float:
    """
    Score an agent's action sequence against the optimal sequence.

    Scoring rules
    -------------
    * Base score is built by stepping through optimal actions in order.
    * Each optimal action found (in order) in actions_taken adds
      ``1 / len(optimal_actions)`` to the base score.
    * Extra actions beyond the optimal length penalise the score by
      ``0.05`` per extra step, capped so the score never goes below 0.
    * Final score is clamped to [0.0, 1.0].

    Parameters
    ----------
    actions_taken   : Actions produced by the agent (in order).
    optimal_actions : Ground-truth optimal action sequence.

    Returns
    -------
    float in [0.0, 1.0]
    """
    if not optimal_actions:
        return 1.0 if not actions_taken else 0.0

    n_optimal = len(optimal_actions)
    matched = 0
    search_start = 0  # advance pointer so order is respected

    for optimal_act in optimal_actions:
        for idx in range(search_start, len(actions_taken)):
            if actions_taken[idx] == optimal_act:
                matched += 1
                search_start = idx + 1
                break

    base_score: float = matched / n_optimal

    # Penalise extra steps
    extra_steps = max(0, len(actions_taken) - n_optimal)
    penalty = extra_steps * 0.05
    score = base_score - penalty

    return round(max(0.0, min(1.0, score)), 4)
