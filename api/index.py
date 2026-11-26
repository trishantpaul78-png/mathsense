"""
Flask API backend for Vercel deployment of Adaptive Maths Gap Lab.

Replaces the Streamlit UI with JSON API endpoints consumed by a static
HTML/JS frontend. All backend modules (irt, bkt, risk, etc.) are reused
unchanged.
"""

import sys
import os
import json
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup – import backend modules from the project root
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Patch paths for Vercel's read-only filesystem before importing config
if os.environ.get("VERCEL"):
    os.environ.setdefault("VERCEL_RUNTIME", "1")

import config

# On Vercel the project root is read-only except /tmp
if os.environ.get("VERCEL"):
    config.DB_PATH = Path("/tmp/app.db")
    # Ensure data/models paths point at the deployed bundle
    config.DATA_DIR = Path(ROOT) / "data"
    config.MODELS_DIR = Path(ROOT) / "models"
    config.SKILLS_PATH = config.DATA_DIR / "skills.json"
    config.QUESTIONS_PATH = config.DATA_DIR / "questions.json"
    config.RISK_MODEL_PATH = config.MODELS_DIR / "risk_model.pkl"

# Now import backend modules (they read config at import time)
import db
# Re-point db module's DB_PATH after our patch
db.DB_PATH = config.DB_PATH
db.init_db()

import bkt
import irt
import risk
import question_bank as qb
import llm
import simulator

import numpy as np
from flask import Flask, request, jsonify

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(__name__)

# Cache the risk model bundle at module level
_model_bundle = None


def _get_bundle():
    global _model_bundle
    if _model_bundle is None:
        if os.environ.get("VERCEL"):
            tmp_model = Path("/tmp/risk_model.pkl")
            if config.RISK_MODEL_PATH.exists():
                try:
                    _model_bundle = risk.load_model()
                    return _model_bundle
                except Exception:
                    pass
            config.RISK_MODEL_PATH = tmp_model
        _model_bundle = risk.ensure_model()
    return _model_bundle


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
SKILLS = None
SKILL_IDS = None
SKILL_SHORT = None


def _load_skill_meta():
    global SKILLS, SKILL_IDS, SKILL_SHORT
    if SKILLS is None:
        SKILLS = qb.load_skills()
        SKILL_IDS = [s["id"] for s in SKILLS]
        SKILL_SHORT = {s["id"]: s.get("short", s["name"]) for s in SKILLS}


def _json_error(message, status=400):
    return jsonify({"error": str(message)}), status


# ---------------------------------------------------------------------------
# CORS middleware (for local dev)
# ---------------------------------------------------------------------------
@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response


@app.route("/api/<path:path>", methods=["OPTIONS"])
def options_handler(path):
    return "", 204


# ---------------------------------------------------------------------------
# Meta / Skills
# ---------------------------------------------------------------------------
@app.route("/api/skills")
def api_skills():
    _load_skill_meta()
    levels = qb.skill_levels()
    edges = list(qb.prerequisite_edges())
    skills_out = []
    for s in SKILLS:
        skills_out.append({
            "id": s["id"],
            "name": s["name"],
            "short": s.get("short", s["name"]),
            "level": levels.get(s["id"], 0),
            "prerequisites": s.get("prerequisites", []),
        })
    return jsonify({"skills": skills_out, "edges": edges})


@app.route("/api/prereq-graph")
def api_prereq_graph():
    _load_skill_meta()
    levels = qb.skill_levels()
    grouped = {}
    for sid in SKILL_IDS:
        grouped.setdefault(levels.get(sid, 0), []).append(sid)

    positions = {}
    for level, group in grouped.items():
        group.sort(key=lambda sid: SKILL_SHORT[sid])
        count = len(group)
        for index, sid in enumerate(group):
            positions[sid] = {"x": level * 1.7, "y": (count - 1) / 2.0 - index}

    edges = []
    for parent, child in qb.prerequisite_edges():
        edges.append({"from": parent, "to": child})

    nodes = []
    for sid in SKILL_IDS:
        nodes.append({
            "id": sid,
            "name": qb.skill_name(sid),
            "short": SKILL_SHORT[sid],
            "x": positions[sid]["x"],
            "y": positions[sid]["y"],
        })

    return jsonify({"nodes": nodes, "edges": edges})


# ---------------------------------------------------------------------------
# Student Management
# ---------------------------------------------------------------------------
@app.route("/api/student", methods=["POST"])
def api_create_student():
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    if not name:
        return _json_error("Name is required")
    student_id = db.get_or_create_student(name)
    student = db.get_student(student_id)
    return jsonify({"student_id": student_id, "name": student["name"]})


@app.route("/api/students")
def api_list_students():
    students = db.list_students()
    return jsonify({"students": students})


@app.route("/api/student/<int:student_id>")
def api_get_student(student_id):
    student = db.get_student(student_id)
    if not student:
        return _json_error("Student not found", 404)
    count = db.count_responses(student_id)
    return jsonify({**student, "responses_count": count})


# ---------------------------------------------------------------------------
# Quiz
# ---------------------------------------------------------------------------
@app.route("/api/question")
def api_get_question():
    student_id = request.args.get("student_id", type=int)
    if not student_id:
        return _json_error("student_id is required")

    _load_skill_meta()
    responses = db.get_responses(student_id)
    mastery = bkt.get_mastery(student_id)
    theta, se = irt.estimate_ability(responses)

    question, target_skill = qb.pick_next_question(student_id, mastery, responses, None)
    if question is None:
        return jsonify({
            "done": True,
            "message": "All seed questions answered.",
            "theta": round(theta, 2),
            "se": round(se, 2),
            "percentile": round(irt.ability_percentile(theta)),
            "mean_mastery": round(float(np.mean(list(mastery.values()))), 3),
            "responses_count": len(responses),
        })

    return jsonify({
        "done": False,
        "question": {
            "id": question["id"],
            "skill": question["skill"],
            "skill_name": qb.skill_name(question["skill"]),
            "question": question["question"],
            "options": question["options"],
            "difficulty": float(question["difficulty"]),
            "discrimination": float(question["discrimination"]),
            "hint": question.get("hint", ""),
        },
        "target_skill": target_skill,
        "theta": round(theta, 2),
        "se": round(se, 2),
        "percentile": round(irt.ability_percentile(theta)),
        "mean_mastery": round(float(np.mean(list(mastery.values()))), 3),
        "responses_count": len(responses),
    })


@app.route("/api/answer", methods=["POST"])
def api_submit_answer():
    data = request.get_json(silent=True) or {}
    student_id = data.get("student_id")
    question_id = data.get("question_id")
    choice_index = data.get("choice_index")
    time_taken = data.get("time_taken", 10.0)
    hint_used = data.get("hint_used", False)

    if not all([student_id, question_id, choice_index is not None]):
        return _json_error("student_id, question_id, and choice_index are required")

    # Look up the question
    question = qb.get_question(str(question_id))
    if not question:
        return _json_error("Question not found", 404)

    correct = int(choice_index) == int(question["answer_index"])
    correct_text = question["options"][int(question["answer_index"])]

    # Record the response
    db.add_response(
        int(student_id), str(question_id), question["skill"],
        correct, float(time_taken), 1 if hint_used else 0,
    )
    new_mastery = bkt.update(int(student_id), question["skill"], correct)

    # Get explanation if wrong
    ai_explanation = None
    if not correct:
        try:
            student_answer = question["options"][int(choice_index)]
            ai_explanation = llm.explain_mistake(question, student_answer, correct_text)
        except Exception:
            ai_explanation = None

    # Updated stats
    responses = db.get_responses(int(student_id))
    mastery = bkt.get_mastery(int(student_id))
    theta, se = irt.estimate_ability(responses)

    return jsonify({
        "correct": correct,
        "correct_answer": correct_text,
        "correct_index": int(question["answer_index"]),
        "new_mastery": round(new_mastery, 3),
        "explanation": question.get("explanation", ""),
        "ai_explanation": ai_explanation,
        "skill_name": qb.skill_name(question["skill"]),
        "theta": round(theta, 2),
        "se": round(se, 2),
        "percentile": round(irt.ability_percentile(theta)),
        "mean_mastery": round(float(np.mean(list(mastery.values()))), 3),
        "responses_count": len(responses),
    })


# ---------------------------------------------------------------------------
# Mastery
# ---------------------------------------------------------------------------
@app.route("/api/mastery")
def api_mastery():
    _load_skill_meta()
    student_id = request.args.get("student_id", type=int)
    if not student_id:
        return _json_error("student_id is required")

    mastery = bkt.get_mastery(student_id)
    responses = db.get_responses(student_id)
    theta, se = irt.estimate_ability(responses)

    mastery_list = []
    for sid in SKILL_IDS:
        mastery_list.append({
            "skill": sid,
            "name": qb.skill_name(sid),
            "short": SKILL_SHORT[sid],
            "mastery": round(mastery.get(sid, 0.0), 3),
        })

    recent = []
    for row in responses[-8:][::-1]:
        recent.append({
            "skill": SKILL_SHORT.get(row["skill"], row["skill"]),
            "correct": row["correct"],
            "time_taken": round(float(row["time_taken"]), 1),
            "hints": int(row["hints_used"]),
        })

    return jsonify({
        "mastery": mastery_list,
        "mastery_map": {sid: round(mastery.get(sid, 0.0), 3) for sid in SKILL_IDS},
        "theta": round(theta, 2),
        "se": round(se, 2),
        "percentile": round(irt.ability_percentile(theta)),
        "mean_mastery": round(float(np.mean(list(mastery.values()))), 3),
        "responses_count": len(responses),
        "recent": recent,
    })


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
@app.route("/api/report")
def api_report():
    _load_skill_meta()
    student_id = request.args.get("student_id", type=int)
    if not student_id:
        return _json_error("student_id is required")

    student = db.get_student(student_id)
    if not student:
        return _json_error("Student not found", 404)

    responses = db.get_responses(student_id)
    mastery = bkt.get_mastery(student_id)
    theta, se = irt.estimate_ability(responses)

    # Risk prediction
    risk_info = None
    try:
        bundle = _get_bundle()
        risk_info = risk.predict_risk(student_id, bundle=bundle)
    except Exception:
        pass

    # Root cause
    observed = sorted(
        {r["skill"] for r in responses}, key=lambda sid: mastery.get(sid, 0.5)
    )
    target_skill = observed[0] if observed else min(mastery, key=mastery.get)
    root_info = risk.find_root_cause(target_skill, mastery)

    # Build output
    mastery_data = []
    for sid in SKILL_IDS:
        mastery_data.append({
            "skill": sid, "short": SKILL_SHORT[sid],
            "name": qb.skill_name(sid),
            "mastery": round(mastery.get(sid, 0.0), 3),
        })

    result = {
        "student": student,
        "responses_count": len(responses),
        "theta": round(theta, 2),
        "se": round(se, 2),
        "percentile": round(irt.ability_percentile(theta)),
        "mean_mastery": round(float(np.mean(list(mastery.values()))), 3),
        "mastery": mastery_data,
        "mastery_map": {sid: round(mastery.get(sid, 0.0), 3) for sid in SKILL_IDS},
        "target_skill": target_skill,
        "target_skill_name": qb.skill_name(target_skill),
    }

    if risk_info:
        result["risk"] = {
            "probability": round(risk_info["risk"], 3),
            "percent": risk_info["risk_percent"],
            "label": risk_info["label"],
            "n_responses": risk_info["n_responses"],
            "top_factors": risk_info["top_factors"],
        }

    if root_info:
        result["root_cause"] = {
            "skill": root_info["skill"],
            "name": root_info["name"],
            "mastery": round(root_info["mastery"], 3),
            "found": root_info["found"],
            "reason": root_info["reason"],
            "path": root_info["path"],
            "path_names": [SKILL_SHORT.get(s, s) for s in root_info["path"]],
        }

    return jsonify(result)


# ---------------------------------------------------------------------------
# Study Plan
# ---------------------------------------------------------------------------
@app.route("/api/plan", methods=["POST"])
def api_plan():
    _load_skill_meta()
    data = request.get_json(silent=True) or {}
    student_id = data.get("student_id")
    if not student_id:
        return _json_error("student_id is required")

    student = db.get_student(int(student_id))
    if not student:
        return _json_error("Student not found", 404)

    responses = db.get_responses(int(student_id))
    mastery = bkt.get_mastery(int(student_id))
    theta, se = irt.estimate_ability(responses)

    risk_info = None
    try:
        bundle = _get_bundle()
        risk_info = risk.predict_risk(int(student_id), bundle=bundle)
    except Exception:
        pass

    observed = sorted(
        {r["skill"] for r in responses}, key=lambda sid: mastery.get(sid, 0.5)
    )
    target_skill = observed[0] if observed else min(mastery, key=mastery.get)
    root_info = risk.find_root_cause(target_skill, mastery)

    weak = sorted(mastery.items(), key=lambda item: item[1])[:3]
    summary = {
        "student": {"name": student["name"], "questions_answered": len(responses),
                     "ability_theta": round(theta, 2),
                     "ability_percentile": round(irt.ability_percentile(theta))},
        "mastery": {SKILL_SHORT[sid]: round(mastery.get(sid, 0.0), 2) for sid in SKILL_IDS},
        "weakest_skills": [s for s, _ in weak],
        "root_cause": root_info["skill"] if root_info else None,
        "root_cause_name": qb.skill_name(root_info["skill"]) if root_info else None,
    }
    if risk_info:
        summary["risk"] = {
            "probability": round(risk_info["risk"], 3),
            "band": risk_info["label"],
            "top_factors": [{"feature": f["label"], "value": f["value"]} for f in risk_info["top_factors"]],
        }

    plan = llm.personalized_plan(summary)

    # Enrich with skill names
    plan["root_cause_skill_name"] = qb.skill_name(plan["root_cause_skill"])
    for ex in plan.get("exercises", []):
        ex["skill_name"] = qb.skill_name(ex.get("skill", ""))
    we = plan.get("worked_example", {})
    if we:
        we["skill_name"] = qb.skill_name(we.get("skill", ""))

    return jsonify(plan)


# ---------------------------------------------------------------------------
# Teacher Dashboard
# ---------------------------------------------------------------------------
@app.route("/api/demo-class", methods=["POST"])
def api_demo_class():
    _load_skill_meta()
    db.delete_students_like("Demo Student%")
    records = simulator.generate_demo_class(n=config.DEMO_CLASS_SIZE)
    start = datetime.now()
    for record in records:
        sid = db.create_student(record["name"])
        for idx, resp in enumerate(record["responses"]):
            db.add_response(
                sid, resp["question_id"], resp["skill"], resp["correct"],
                resp["time_taken"], resp["hints_used"],
                timestamp=(start + timedelta(minutes=3 * idx)).isoformat(timespec="seconds"),
            )
        for skill, value in record["mastery"].items():
            db.save_mastery_snapshot(
                sid, skill, value,
                timestamp=(start + timedelta(minutes=3 * len(record["responses"]))).isoformat(timespec="seconds"),
            )
    return jsonify({"count": len(records), "message": f"Loaded {len(records)} demo students."})


@app.route("/api/dashboard")
def api_dashboard():
    _load_skill_meta()
    students = db.list_students()
    if not students:
        return jsonify({"students": [], "heatmap": None, "risk_table": [], "stats": {}})

    try:
        bundle = _get_bundle()
    except Exception as e:
        return _json_error(f"Risk model unavailable: {e}", 500)

    latest = db.get_all_mastery()
    names = [s["name"] for s in students]
    defaults = {sid: bkt.params_for(sid)["p_init"] for sid in SKILL_IDS}

    # Heatmap data
    heatmap_z = []
    for name in names:
        row = [round(latest.get((name, sid), defaults[sid]), 3) for sid in SKILL_IDS]
        heatmap_z.append(row)

    heatmap = {
        "z": heatmap_z,
        "x": [SKILL_SHORT[sid] for sid in SKILL_IDS],
        "y": names,
    }

    # Risk table
    import pandas as pd
    matrix_data = {sid: [latest.get((name, sid), defaults[sid]) for name in names] for sid in SKILL_IDS}
    matrix = pd.DataFrame(matrix_data, index=names)

    rows = []
    for student in students:
        sid = int(student["id"])
        try:
            info = risk.predict_risk(sid, bundle=bundle)
        except Exception:
            continue
        root = info.get("root_cause")
        rows.append({
            "name": student["name"],
            "risk_pct": round(100 * info["risk"], 1),
            "band": info["label"],
            "avg_mastery_pct": round(100 * float(matrix.loc[student["name"]].mean()), 1),
            "weakest_skill": qb.skill_short(info["weakest_skill"]) if info["weakest_skill"] else "-",
            "root_cause": root["name"] if root else "-",
            "answers": info["n_responses"],
        })

    rows.sort(key=lambda r: r["risk_pct"], reverse=True)

    # Stats for class insight
    matrix_np = np.array(heatmap_z)
    col_means = matrix_np.mean(axis=0)
    weakest_indices = np.argsort(col_means)[:3]
    weakest_skills = [
        {"skill": SKILL_IDS[i], "name": qb.skill_name(SKILL_IDS[i]), "mastery": round(float(col_means[i]), 3)}
        for i in weakest_indices
    ]

    n_at_risk = sum(1 for r in rows if r["risk_pct"] >= 50)
    at_risk_rows = [r for r in rows if r["risk_pct"] >= 50][:8]

    stats = {
        "n_students": len(students),
        "n_at_risk": n_at_risk,
        "class_avg_mastery": round(float(matrix_np.mean()), 3),
        "weakest_skills": weakest_skills,
        "at_risk": [
            {"name": r["name"], "risk": round(r["risk_pct"] / 100, 2),
             "weakest_skill": r["weakest_skill"], "root_cause": r["root_cause"]}
            for r in at_risk_rows
        ],
    }

    return jsonify({"heatmap": heatmap, "risk_table": rows, "stats": stats})


@app.route("/api/class-insight", methods=["POST"])
def api_class_insight():
    data = request.get_json(silent=True) or {}
    text = llm.teacher_summary(data)
    return jsonify({"insight": text})


# ---------------------------------------------------------------------------
# Model Evaluation
# ---------------------------------------------------------------------------
@app.route("/api/evaluation")
def api_evaluation():
    try:
        bundle = _get_bundle()
    except Exception as e:
        return _json_error(f"Could not load risk model: {e}", 500)

    metrics = bundle["metrics"]
    baseline = bundle["baseline"]
    earliness = bundle["earliness"]

    return jsonify({
        "caption": {
            "trained_at": bundle["trained_at"],
            "n_train_students": bundle["n_train_students"],
            "n_train": bundle["n_train"],
            "n_test_students": bundle["n_test_students"],
            "at_risk_rate": round(bundle["at_risk_rate"], 2),
        },
        "comparison": [
            {"model": "RandomForest (7 features)", **{k: round(v, 3) for k, v in metrics.items()}},
            {"model": "Baseline: recent accuracy < 50%", **{k: round(v, 3) for k, v in baseline.items()}},
        ],
        "roc": bundle["roc"],
        "confusion": bundle["confusion"],
        "earliness": {
            "positions": earliness["positions"],
            "mean_questions": round(earliness["mean_questions"], 1) if not np.isnan(earliness["mean_questions"]) else None,
            "flagged_fraction": round(earliness["flagged_fraction"], 2),
            "n_at_risk_test": earliness["n_at_risk_test"],
            "n_items": earliness["n_items"],
        },
        "importances": {risk.FEATURE_LABELS.get(k, k): round(v, 3) for k, v in bundle["importances"].items()},
        "feature_labels": risk.FEATURE_LABELS,
    })


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.route("/api/health")
def api_health():
    return jsonify({
        "status": "ok",
        "llm": llm.status_message(),
        "db": str(config.DB_PATH),
        "vercel": bool(os.environ.get("VERCEL")),
    })


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error", "detail": str(e)}), 500

