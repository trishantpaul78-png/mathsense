"""
2PL Item Response Theory - pure Python / numpy, no LLM calls.

Model
-----
    P(correct | theta, a, b) = 1 / (1 + exp(-a (theta - b)))

    a = discrimination (0.5 .. 2)
    b = difficulty     (-2 .. +2)
    theta ~ N(0, 1) prior

Ability estimation
------------------
MAP with a N(0,1) prior, solved by Newton-Raphson on the log posterior

    L(theta) = sum_i [ u_i log P_i + (1-u_i) log(1-P_i) ] - (theta-mu)^2 / (2 sigma^2)
    L'(theta)  = sum_i a_i (u_i - P_i) - (theta-mu)/sigma^2
    L''(theta) = -sum_i a_i^2 P_i (1-P_i) - 1/sigma^2

    theta_new = theta - L'(theta)/L''(theta),  SE = sqrt(-1/L'')

Item information
----------------
    I_i(theta) = a_i^2 P_i (1 - P_i)

The next item is the unasked question of the target skill with the highest
information at the current theta (maximum-information selection).
"""

import math
from statistics import NormalDist

import numpy as np

PRIOR_MEAN = 0.0
PRIOR_SD = 1.0
THETA_BOUNDS = (-4.0, 4.0)


def probability_correct(theta: float, a: float = 1.0, b: float = 0.0) -> float:
    """2PL probability of a correct answer (scalar in, float out)."""
    z = float(a) * (float(theta) - float(b))
    z = max(-35.0, min(35.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def probability_curve(theta: np.ndarray, a: float, b: float) -> np.ndarray:
    z = np.clip(float(a) * (np.asarray(theta, dtype=float) - float(b)), -35, 35)
    return 1.0 / (1.0 + np.exp(-z))


def item_information(theta: float, a: float, b: float) -> float:
    """Fisher information of a 2PL item at theta: a^2 P (1-P)."""
    p = probability_correct(theta, a, b)
    return float(a) ** 2 * p * (1.0 - p)


def _item_params(response: dict) -> tuple[float, float]:
    """Read (a, b) from a response dict, falling back to the question bank."""
    if "discrimination" in response and "difficulty" in response:
        return float(response["discrimination"]), float(response["difficulty"])
    if "a" in response and "b" in response:
        return float(response["a"]), float(response["b"])

    import question_bank

    question = question_bank.get_question(response.get("question_id", ""))
    if question:
        return float(question["discrimination"]), float(question["difficulty"])
    return 1.0, 0.0


def estimate_ability(
    responses: list[dict],
    prior_mean: float = PRIOR_MEAN,
    prior_sd: float = PRIOR_SD,
    max_iter: int = 60,
    tol: float = 1e-6,
) -> tuple[float, float]:
    """
    MAP ability estimate with a normal prior.

    `responses` need `correct` plus either (a, b) / (discrimination, difficulty)
    or a question_id that exists in data/questions.json.
    Returns (theta, standard_error).
    """
    rows = [
        (*_item_params(r), 1.0 if r.get("correct") else 0.0) for r in responses
    ]
    if not rows:
        return float(prior_mean), float(prior_sd)

    prior_var = float(prior_sd) ** 2
    theta = float(prior_mean)

    for _ in range(max_iter):
        gradient = -(theta - prior_mean) / prior_var
        information = 1.0 / prior_var

        for a, b, u in rows:
            p = probability_correct(theta, a, b)
            gradient += a * (u - p)
            information += a * a * p * (1.0 - p)

        if information <= 1e-9:
            break

        step = gradient / information
        theta_new = theta + step
        theta_new = max(THETA_BOUNDS[0], min(THETA_BOUNDS[1], theta_new))

        if abs(theta_new - theta) < tol:
            theta = theta_new
            break
        theta = theta_new

    se = 1.0 / math.sqrt(max(information, 1e-9))
    return float(theta), float(se)


def ability_percentile(theta: float) -> float:
    """Percentile of theta under the reference N(0, 1) population."""
    return 100.0 * NormalDist().cdf(float(theta))


def select_next_question(
    theta: float,
    candidates: list[dict],
    asked_ids: set[str] | None = None,
) -> dict | None:
    """Maximum-information item among unasked candidates."""
    asked = set(asked_ids or set())
    pool = [q for q in candidates if q.get("id") not in asked]
    if not pool:
        return None

    def info(question: dict) -> float:
        return item_information(
            theta, float(question["discrimination"]), float(question["difficulty"])
        )

    return max(pool, key=info)


def expected_score(theta: float, questions: list[dict]) -> float:
    """Expected number correct on a list of items at theta (diagnostic aid)."""
    if not questions:
        return 0.0
    return float(
        np.mean(
            [
                probability_correct(
                    theta, q["discrimination"], q["difficulty"]
                )
                for q in questions
            ]
        )
    )


def item_information_curve(
    theta_values: np.ndarray, a: float, b: float
) -> np.ndarray:
    p = probability_curve(theta_values, a, b)
    return (float(a) ** 2) * p * (1.0 - p)
