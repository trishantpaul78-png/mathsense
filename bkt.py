"""
Bayesian Knowledge Tracing (BKT) - pure Python, no LLM calls.

For every skill we keep P(L) = probability the student has learned it.
One response triggers two steps:

  1) Evidence update (Bayes):
       correct : P(L|correct) = P(L)(1-s) / [P(L)(1-s) + (1-P(L))g]
       wrong   : P(L|wrong)   = P(L)s     / [P(L)s     + (1-P(L))(1-g)]

  2) Learning transition:
       P(L_next) = P(L|obs) + (1 - P(L|obs)) * p_learn

where s = p_slip and g = p_guess. Both steps are clamped to (0, 1).
The full mastery history per student per skill is written to
mastery_snapshots, so the UI can plot trajectories.
"""

from config import BKT_DEFAULTS, BKT_SKILL_OVERRIDES

import db

_EPS = 1e-4


def _clamp(p: float) -> float:
    return float(min(1.0 - _EPS, max(_EPS, p)))


def params_for(skill: str) -> dict:
    """Default BKT parameters merged with any per-skill override."""
    params = dict(BKT_DEFAULTS)
    params.update(BKT_SKILL_OVERRIDES.get(skill, {}))
    return params


def step(
    p_known: float,
    correct: bool,
    p_learn: float | None = None,
    p_slip: float | None = None,
    p_guess: float | None = None,
) -> float:
    """One BKT update. Returns the new P(known) after the learning transition."""
    p_learn = BKT_DEFAULTS["p_learn"] if p_learn is None else p_learn
    p_slip = BKT_DEFAULTS["p_slip"] if p_slip is None else p_slip
    p_guess = BKT_DEFAULTS["p_guess"] if p_guess is None else p_guess

    p = _clamp(p_known)

    if correct:
        numerator = p * (1.0 - p_slip)
        denominator = numerator + (1.0 - p) * p_guess
    else:
        numerator = p * p_slip
        denominator = numerator + (1.0 - p) * (1.0 - p_guess)

    posterior = numerator / denominator if denominator > 0 else p
    posterior = _clamp(posterior)

    # Learning opportunity: even a wrong answer can move mastery up slightly,
    # which is exactly the "productive struggle" behaviour BKT models.
    return _clamp(posterior + (1.0 - posterior) * p_learn)


def _skill_ids() -> list[str]:
    import question_bank

    return question_bank.skill_ids()


def initial_mastery() -> dict[str, float]:
    """Starting P(known) for every skill in the curriculum."""
    return {skill: params_for(skill)["p_init"] for skill in _skill_ids()}


class BKTModel:
    """In-memory tracker, useful for simulations and unit-style checks."""

    def __init__(self, skill_ids: list[str] | None = None):
        self.skills = list(skill_ids) if skill_ids else _skill_ids()
        self._mastery = {skill: params_for(skill)["p_init"] for skill in self.skills}
        self._history: dict[str, list[float]] = {skill: [] for skill in self.skills}

    def update(self, skill: str, correct: bool) -> float:
        """Update one skill and return the new mastery probability."""
        if skill not in self._mastery:
            self.skills.append(skill)
            self._mastery[skill] = params_for(skill)["p_init"]
            self._history[skill] = []
        params = params_for(skill)
        new_value = step(
            self._mastery[skill],
            correct,
            params["p_learn"],
            params["p_slip"],
            params["p_guess"],
        )
        self._mastery[skill] = new_value
        self._history[skill].append(new_value)
        return new_value

    def mastery(self, skill: str) -> float:
        return self._mastery.get(skill, params_for(skill)["p_init"])

    def snapshot(self) -> dict[str, float]:
        return dict(self._mastery)

    def history(self, skill: str) -> list[float]:
        return list(self._history.get(skill, []))

    def mean_mastery(self, observed_only: bool = False) -> float:
        values = list(self._mastery.values())
        return sum(values) / len(values) if values else 0.0


def get_mastery(student_id: int) -> dict[str, float]:
    """Latest persisted mastery for every skill (unseen skills get p_init)."""
    mastery = initial_mastery()
    for row in db.get_mastery_rows(int(student_id)):
        skill = row["skill"]
        if skill in mastery:
            mastery[skill] = float(row["mastery"])
    return mastery


def update(student_id: int, skill: str, correct: bool) -> float:
    """Persist one BKT step for a student and return the new mastery."""
    current = get_mastery(int(student_id)).get(skill, params_for(skill)["p_init"])
    params = params_for(skill)
    new_value = step(
        current, correct, params["p_learn"], params["p_slip"], params["p_guess"]
    )
    db.save_mastery_snapshot(int(student_id), skill, new_value)
    return new_value


def mastery_history(student_id: int, skill: str) -> list[dict]:
    return db.get_mastery_history(int(student_id), skill)


def weakest_skills(mastery: dict[str, float], k: int = 3) -> list[tuple[str, float]]:
    return sorted(mastery.items(), key=lambda item: item[1])[:k]


def observed_mastery(student_id: int) -> dict[str, float]:
    """Only the skills the student has actually attempted."""
    seen = {row["skill"] for row in db.get_responses(int(student_id))}
    if not seen:
        return {}
    return {skill: value for skill, value in get_mastery(student_id).items() if skill in seen}
