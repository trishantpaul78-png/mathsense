"""
Early-warning risk model and prerequisite root-cause finder.

Features (one row per student, computed from the response log)
--------------------------------------------------------------
recent_accuracy   mean correctness over the last 5 answers
accuracy_trend    least-squares slope of correctness over the last 10 answers
rt_z              z-score of the student's mean response time vs the cohort baseline
hint_rate         mean hints used per question
wrong_streak      trailing run of wrong answers (capped at 10)
avg_mastery       mean BKT mastery over the skills actually attempted
skills_below_04   count of attempted skills with mastery < 0.40

The classifier is a RandomForest trained on simulator.py's synthetic cohort.
`predict_risk` returns a probability plus the top-3 local contributions
(importance x standardised feature value, signed). This is a simple,
explainable attribution - not SHAP.
"""

import pickle

import numpy as np
import pandas as pd

import bkt
import config
import db
import question_bank

FEATURE_NAMES = [
    "recent_accuracy",
    "accuracy_trend",
    "rt_z",
    "hint_rate",
    "wrong_streak",
    "avg_mastery",
    "skills_below_04",
]

FEATURE_LABELS = {
    "recent_accuracy": "Recent accuracy",
    "accuracy_trend": "Accuracy trend",
    "rt_z": "Response-time z-score",
    "hint_rate": "Hint usage",
    "wrong_streak": "Wrong-answer streak",
    "avg_mastery": "Average mastery",
    "skills_below_04": "Skills below 40% mastery",
}


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def _replay_mastery(responses: list[dict]) -> dict[str, float]:
    """BKT replay from the configured priors using the observed answers."""
    mastery = {
        skill: bkt.params_for(skill)["p_init"] for skill in question_bank.skill_ids()
    }
    for response in responses:
        skill = response.get("skill")
        if skill is None:
            continue
        params = bkt.params_for(skill)
        mastery[skill] = bkt.step(
            mastery.get(skill, params["p_init"]),
            bool(response.get("correct")),
            params["p_learn"],
            params["p_slip"],
            params["p_guess"],
        )
    return mastery


def extract_features(
    responses: list[dict],
    mastery: dict[str, float] | None = None,
    rt_mean: float | None = None,
    rt_sd: float | None = None,
) -> dict[str, float]:
    """Build the seven risk features from a chronological response log."""
    rt_mean = config.DEFAULT_RT_MEAN if rt_mean is None else float(rt_mean)
    rt_sd = config.DEFAULT_RT_SD if rt_sd is None else float(rt_sd)
    if rt_sd <= 0:
        rt_sd = 1.0

    if not responses:
        empty_mastery = mastery or {
            skill: bkt.params_for(skill)["p_init"] for skill in question_bank.skill_ids()
        }
        return {
            "recent_accuracy": 0.0,
            "accuracy_trend": 0.0,
            "rt_z": 0.0,
            "hint_rate": 0.0,
            "wrong_streak": 0.0,
            "avg_mastery": float(np.mean(list(empty_mastery.values()))),
            "skills_below_04": 0.0,
        }

    correctness = np.array([1.0 if r.get("correct") else 0.0 for r in responses])
    times = np.array([float(r.get("time_taken") or 0.0) for r in responses])
    hints = np.array([float(r.get("hints_used") or 0) for r in responses])

    recent_accuracy = float(correctness[-config.RECENT_WINDOW :].mean())

    window = correctness[-10:]
    if len(window) >= 3:
        accuracy_trend = float(np.polyfit(np.arange(len(window)), window, 1)[0])
    else:
        accuracy_trend = 0.0

    rt_z = float(np.clip((times.mean() - rt_mean) / rt_sd, -4.0, 4.0))

    hint_rate = float(np.clip(hints.mean(), 0.0, 4.0))

    wrong_streak = 0
    for value in correctness[::-1]:
        if value >= 0.5:
            break
        wrong_streak += 1
    wrong_streak = float(min(wrong_streak, 10))

    mastery_map = mastery if mastery is not None else _replay_mastery(responses)
    observed = {r.get("skill") for r in responses if r.get("skill")}
    if observed:
        avg_mastery = float(
            np.mean([mastery_map[s] for s in observed if s in mastery_map])
        )
        skills_below = float(
            sum(
                1
                for s in observed
                if mastery_map.get(s, 1.0) < config.MASTERY_THRESHOLD
            )
        )
    else:
        avg_mastery = float(np.mean(list(mastery_map.values()))) if mastery_map else 0.5
        skills_below = 0.0

    return {
        "recent_accuracy": round(recent_accuracy, 4),
        "accuracy_trend": round(accuracy_trend, 4),
        "rt_z": round(rt_z, 4),
        "hint_rate": round(hint_rate, 4),
        "wrong_streak": wrong_streak,
        "avg_mastery": round(avg_mastery, 4),
        "skills_below_04": skills_below,
    }


def feature_row(features: dict[str, float]) -> pd.DataFrame:
    """One-row frame with the training column order (keeps sklearn quiet)."""
    return pd.DataFrame(
        [{name: float(features.get(name, 0.0)) for name in FEATURE_NAMES}],
        columns=FEATURE_NAMES,
    )


# ---------------------------------------------------------------------------
# Model lifecycle
# ---------------------------------------------------------------------------
def load_model() -> dict:
    with open(config.RISK_MODEL_PATH, "rb") as handle:
        return pickle.load(handle)


def ensure_model(force: bool = False) -> dict:
    """Load models/risk_model.pkl, training it on first use if missing."""
    if not force and config.RISK_MODEL_PATH.exists():
        try:
            return load_model()
        except Exception:  # corrupt pickle -> retrain rather than crash the app
            pass

    import simulator

    bundle = simulator.train_and_evaluate(save=True, verbose=False)
    return bundle


def train_model() -> dict:
    return ensure_model(force=True)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def risk_label(probability: float) -> str:
    for threshold, label in config.RISK_BANDS:
        if probability >= threshold:
            return label
    return "Low"


def _top_contributions(
    bundle: dict, features: dict[str, float], k: int = 3
) -> list[dict]:
    importances = bundle.get("importances", {})
    means = bundle.get("feature_means", {})
    stds = bundle.get("feature_stds", {})

    scored = []
    for name in FEATURE_NAMES:
        value = float(features.get(name, 0.0))
        std = float(stds.get(name, 1.0) or 1.0)
        z = (value - float(means.get(name, 0.0))) / std
        contribution = float(importances.get(name, 0.0)) * z
        scored.append(
            {
                "feature": name,
                "label": FEATURE_LABELS.get(name, name),
                "value": round(value, 3),
                "contribution": round(contribution, 4),
                "direction": "raises" if contribution > 0 else "lowers",
            }
        )

    scored.sort(key=lambda row: abs(row["contribution"]), reverse=True)
    return scored[:k]


def predict_risk(student_id: int, bundle: dict | None = None) -> dict:
    """
    Risk probability + human-readable drivers for one student.

    Returns a dict even with zero responses (the features are then cold-start
    values), but the UI should ask for at least a few answers.
    """
    student_id = int(student_id)
    responses = db.get_responses(student_id)
    mastery = bkt.get_mastery(student_id)
    features = extract_features(responses, mastery=mastery)

    bundle = bundle or ensure_model()
    model = bundle["model"]
    probability = float(model.predict_proba(feature_row(features))[0, 1])

    observed = sorted(
        {r["skill"] for r in responses if r.get("skill")},
        key=lambda skill: mastery.get(skill, 0.5),
    )
    if observed:
        weakest = observed[0]
    else:
        weakest = min(mastery, key=mastery.get) if mastery else None

    root_cause = find_root_cause(weakest, mastery) if weakest else None

    return {
        "student_id": student_id,
        "risk": probability,
        "risk_percent": round(100.0 * probability, 1),
        "label": risk_label(probability),
        "features": features,
        "top_factors": _top_contributions(bundle, features),
        "weakest_skill": weakest,
        "root_cause": root_cause,
        "n_responses": len(responses),
    }


# ---------------------------------------------------------------------------
# Prerequisite root-cause finder
# ---------------------------------------------------------------------------
def find_root_cause(
    skill: str | None, mastery: dict[str, float]
) -> dict | None:
    """
    Walk the prerequisite DAG backwards from `skill` and return the earliest
    ancestor whose mastery is below config.ROOT_CAUSE_THRESHOLD (0.50).

    "Earliest" = smallest depth in the DAG; ties are broken by lower mastery.
    If no ancestor is below the threshold, the weakest ancestor is returned
    with found=False so the UI can say "no clear prerequisite gap".
    """
    if not skill:
        return None

    mastery = mastery or {}
    levels = question_bank.skill_levels()
    ancestors = question_bank.ancestors(skill)

    failing = [
        node
        for node in ancestors
        if float(mastery.get(node, 0.5)) < config.ROOT_CAUSE_THRESHOLD
    ]

    if failing:
        chosen = min(
            failing, key=lambda node: (levels.get(node, 99), float(mastery.get(node, 1.0)))
        )
        found = True
        reason = "earliest prerequisite below 50% mastery"
    else:
        pool = ancestors or {skill}
        chosen = min(pool, key=lambda node: float(mastery.get(node, 1.0)))
        found = False
        reason = (
            "no prerequisite below 50% mastery; weakest ancestor shown"
            if ancestors
            else "no prerequisites for this skill"
        )

    return {
        "skill": chosen,
        "name": question_bank.skill_name(chosen),
        "mastery": float(mastery.get(chosen, 0.5)),
        "found": found,
        "reason": reason,
        "path": question_bank.path_between(chosen, skill),
    }
