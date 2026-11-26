"""
Synthetic cohort generator + risk-model evaluation harness.

Run directly to train and persist the model:

    python simulator.py

What the simulation does
------------------------
Each synthetic student has a hidden ability theta ~ N(0, 1), a personal
learning rate, slip/guess behaviour and per-skill initial mastery that depends
on theta. Over a 20-question session they answer questions chosen toward their
weak skills; theta and BKT mastery both update as they learn.

The hidden label is:

    at_risk = (mean final BKT mastery across all skills) < 0.50

The classifier only ever sees features derivable from the response log
(accuracy, trend, timing, hints, streak, replayed mastery), never the hidden
mastery or theta - so the evaluation is honest about the label.

Reported metrics: accuracy, precision, recall, F1, ROC-AUC, plus "earliness"
(how many questions into the session at-risk students are first flagged), and
a baseline of "recent accuracy < 50%" for comparison.
"""

import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

import bkt
import config
import irt
import question_bank
import risk


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
def _replay_mastery(responses: list[dict]) -> dict[str, float]:
    mastery = {
        skill: bkt.params_for(skill)["p_init"] for skill in question_bank.skill_ids()
    }
    for response in responses:
        skill = response["skill"]
        params = bkt.params_for(skill)
        mastery[skill] = bkt.step(
            mastery[skill],
            response["correct"],
            params["p_learn"],
            params["p_slip"],
            params["p_guess"],
        )
    return mastery


def simulate_students(
    n: int = config.N_SYNTHETIC_STUDENTS,
    seed: int = 42,
    name_prefix: str = "Sim Student",
    n_items: int | None = None,
) -> list[dict]:
    """Generate `n` synthetic learners with response logs and hidden labels."""
    rng = np.random.default_rng(seed)
    skills = question_bank.skill_ids()
    banks = {skill: question_bank.questions_for_skill(skill) for skill in skills}
    banks = {skill: pool for skill, pool in banks.items() if pool}
    n_items = int(n_items or config.ITEMS_PER_SESSION)

    records = []
    for index in range(n):
        theta_start = float(rng.normal(0.0, 1.0))
        theta = theta_start
        learning_rate = float(rng.uniform(0.02, 0.22))
        slip = float(rng.uniform(0.05, 0.18))
        guess = float(rng.uniform(0.15, 0.30))

        base_level = 0.10 + 0.70 / (1.0 + np.exp(-1.2 * theta_start))
        hidden_mastery = {}
        for skill in skills:
            value = base_level + float(rng.normal(0.0, 0.10))
            hidden_mastery[skill] = float(np.clip(value, 0.03, 0.95))

        # Give most students one "blocked" prerequisite, like real cohorts.
        blocked = skills[int(rng.integers(0, len(skills)))]
        hidden_mastery[blocked] = float(
            np.clip(hidden_mastery[blocked] - 0.20, 0.02, 0.95)
        )

        responses = []
        for _ in range(n_items):
            # Weak skills get more practice: weight ~ (1.35 - mastery).
            weights = np.array(
                [max(0.05, 1.35 - hidden_mastery[skill]) for skill in skills]
            )
            weights = weights / weights.sum()
            skill = skills[int(rng.choice(len(skills), p=weights))]
            pool = banks[skill]
            question = pool[int(rng.integers(0, len(pool)))]

            p_true = irt.probability_correct(
                theta, question["discrimination"], question["difficulty"]
            )
            p_observed = p_true * (1.0 - slip) + (1.0 - p_true) * guess
            correct = bool(rng.random() < p_observed)

            hint_lambda = max(
                0.05, 0.35 + (0.9 if not correct else 0.0) + 0.7 * (1.0 - hidden_mastery[skill])
            )
            hints = int(rng.poisson(hint_lambda))

            mean_rt = (
                config.DEFAULT_RT_MEAN
                * (1.20 - 0.45 * hidden_mastery[skill])
                * (1.10 if not correct else 0.92)
            )
            time_taken = float(np.clip(rng.normal(mean_rt, 6.0), 3.0, 150.0))

            responses.append(
                {
                    "question_id": question["id"],
                    "skill": skill,
                    "difficulty": float(question["difficulty"]),
                    "discrimination": float(question["discrimination"]),
                    "correct": correct,
                    "time_taken": round(time_taken, 2),
                    "hints_used": hints,
                }
            )

            hidden_mastery[skill] = bkt.step(
                hidden_mastery[skill], correct, learning_rate, slip, guess
            )
            theta = float(
                np.clip(
                    theta + learning_rate * (1.0 if correct else -0.55), -4.0, 4.0
                )
            )

        final_mean = float(np.mean(list(hidden_mastery.values())))
        records.append(
            {
                "name": f"{name_prefix} {index + 1:02d}",
                "index": index,
                "theta_start": theta_start,
                "theta_end": theta,
                "learning_rate": learning_rate,
                "slip": slip,
                "guess": guess,
                "responses": responses,
                "hidden_mastery": hidden_mastery,
                "final_mastery": final_mean,
                "at_risk": bool(final_mean < 0.50),
                "mastery": _replay_mastery(responses),
            }
        )
    return records


def generate_demo_class(
    n: int = config.DEMO_CLASS_SIZE, seed: int = 2024, prefix: str = "Demo Student"
) -> list[dict]:
    """Records ready to be written into SQLite by the teacher dashboard."""
    return simulate_students(
        n=n, seed=seed, name_prefix=prefix, n_items=config.ITEMS_PER_SESSION
    )


def build_dataset(
    n_students: int = config.N_SYNTHETIC_STUDENTS, seed: int = 42
) -> list[dict]:
    """Simulate students and attach full-session + early-session feature snapshots."""
    records = simulate_students(n=n_students, seed=seed)
    rng = np.random.default_rng(seed + 1)
    for record in records:
        responses = record["responses"]
        record["features"] = risk.extract_features(responses)
        # A random prefix so the classifier also learns what partial evidence
        # looks like. The label still comes from the student's final outcome.
        if len(responses) > 3:
            cut = int(rng.integers(3, len(responses) + 1))
        else:
            cut = len(responses)
        record["features_early"] = risk.extract_features(responses[:cut])
    return records


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def _metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict:
    auc = (
        float(roc_auc_score(y_true, y_score))
        if len(set(np.asarray(y_true).tolist())) > 1
        else float("nan")
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": auc,
    }


def _first_flag(model, responses: list[dict]) -> int | None:
    """Question index at which the model first flags the student (from Q3)."""
    for k in range(3, len(responses) + 1):
        features = risk.extract_features(responses[:k])
        probability = float(model.predict_proba(risk.feature_row(features))[0, 1])
        if probability >= config.RISK_THRESHOLD:
            return k
    return None


def train_and_evaluate(
    n_students: int = config.N_SYNTHETIC_STUDENTS,
    test_fraction: float = 0.25,
    seed: int = 7,
    save: bool = True,
    verbose: bool = True,
) -> dict:
    """Train the risk classifier, evaluate against the baseline, save the bundle."""
    students = build_dataset(n_students=n_students, seed=seed)
    at_risk = np.array([int(record["at_risk"]) for record in students])
    indices = np.arange(len(students))

    # Split by STUDENT so the two snapshots of one learner never straddle the
    # train/test boundary (that would leak the label).
    idx_train, idx_test = train_test_split(
        indices,
        test_size=test_fraction,
        random_state=seed,
        stratify=at_risk,
    )

    train_rows = []
    train_labels = []
    for i in idx_train:
        record = students[int(i)]
        train_rows.append(record["features"])
        train_rows.append(record["features_early"])
        train_labels.extend([int(record["at_risk"]), int(record["at_risk"])])

    X_train = pd.DataFrame(train_rows, columns=risk.FEATURE_NAMES)
    y_train = np.array(train_labels)
    X_test = pd.DataFrame(
        [students[int(i)]["features"] for i in idx_test], columns=risk.FEATURE_NAMES
    )
    y_test = np.array([int(students[int(i)]["at_risk"]) for i in idx_test])

    model = RandomForestClassifier(
        n_estimators=240,
        max_depth=6,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= config.RISK_THRESHOLD).astype(int)
    model_metrics = _metrics(y_test, predictions, probabilities)

    # Baseline: flag when recent accuracy < 50%.
    baseline_pred = (X_test["recent_accuracy"].to_numpy() < 0.50).astype(int)
    baseline_score = 1.0 - X_test["recent_accuracy"].to_numpy()
    baseline_metrics = _metrics(y_test, baseline_pred, baseline_score)

    if len(set(y_test.tolist())) > 1:
        fpr, tpr, _ = roc_curve(y_test, probabilities)
    else:
        fpr, tpr = np.array([0.0, 1.0]), np.array([0.0, 1.0])
    confusion = confusion_matrix(y_test, predictions, labels=[0, 1])

    at_risk_indices = [
        int(idx) for idx, label in zip(idx_test, y_test) if int(label) == 1
    ]
    positions = []
    for idx in at_risk_indices:
        position = _first_flag(model, students[idx]["responses"])
        if position is not None:
            positions.append(position)
    flagged_fraction = len(positions) / max(1, len(at_risk_indices))
    mean_earliness = float(np.mean(positions)) if positions else float("nan")

    bundle = {
        "model": model,
        "feature_names": list(risk.FEATURE_NAMES),
        "feature_labels": dict(risk.FEATURE_LABELS),
        "feature_means": X_train.mean().to_dict(),
        "feature_stds": X_train.std(ddof=0).replace(0, 1).to_dict(),
        "importances": dict(zip(risk.FEATURE_NAMES, model.feature_importances_)),
        "metrics": model_metrics,
        "baseline": baseline_metrics,
        "roc": {
            "fpr": [float(v) for v in fpr],
            "tpr": [float(v) for v in tpr],
            "auc": model_metrics["roc_auc"],
        },
        "confusion": [[int(v) for v in row] for row in confusion],
        "earliness": {
            "positions": positions,
            "mean_questions": mean_earliness,
            "flagged_fraction": float(flagged_fraction),
            "n_at_risk_test": len(at_risk_indices),
            "n_items": int(config.ITEMS_PER_SESSION),
        },
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_train_students": int(len(idx_train)),
        "n_test_students": int(len(idx_test)),
        "at_risk_rate": float(at_risk.mean()),
        "seed": int(seed),
    }

    if save:
        config.RISK_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(config.RISK_MODEL_PATH, "wb") as handle:
            pickle.dump(bundle, handle)

    if verbose:
        _report(bundle)

    return bundle


def _report(bundle: dict) -> None:
    rows = []
    for name, metrics in (
        ("RandomForest (7 features)", bundle["metrics"]),
        ("Baseline: recent accuracy < 50%", bundle["baseline"]),
    ):
        rows.append(
            {
                "model": name,
                "accuracy": round(metrics["accuracy"], 3),
                "precision": round(metrics["precision"], 3),
                "recall": round(metrics["recall"], 3),
                "f1": round(metrics["f1"], 3),
                "roc_auc": round(metrics["roc_auc"], 3),
            }
        )

    print("=" * 78)
    print("Synthetic cohort evaluation (hidden label: mean final mastery < 0.50)")
    print(
        f"train = {bundle['n_train_students']} students ({bundle['n_train']} snapshots)   "
        f"test = {bundle['n_test_students']} students   "
        f"at-risk rate = {bundle['at_risk_rate']:.0%}"
    )
    print("-" * 78)
    print(pd.DataFrame(rows).to_string(index=False))
    print("-" * 78)
    early = bundle["earliness"]
    print(
        "Earliness: flagged "
        f"{early['flagged_fraction']:.0%} of at-risk test students after a mean of "
        f"{early['mean_questions']:.1f} / {early['n_items']} questions."
    )
    print("Feature importances:")
    for name, value in sorted(
        bundle["importances"].items(), key=lambda item: item[1], reverse=True
    ):
        print(f"  {name:20s} {value:.3f}")
    print("=" * 78)
    print(f"Saved model -> {config.RISK_MODEL_PATH}")


def main() -> None:
    train_and_evaluate(
        n_students=config.N_SYNTHETIC_STUDENTS,
        seed=7,
        save=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
