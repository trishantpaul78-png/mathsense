"""
Adaptive Maths Gap Lab - Streamlit front end.

Four tabs
---------
1. Student Quiz        adaptive IRT practice, instant feedback, live mastery
2. My Report           radar, risk gauge, prerequisite DAG, Groq study plan
3. Teacher Dashboard   class heatmap, ranked risk list, Groq class insight
4. Model Evaluation    ROC, confusion matrix, earliness, baseline comparison

Run with:  streamlit run app.py
"""

import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import bkt
import config
import db
import irt
import llm
import question_bank as qb
import risk
import simulator

st.set_page_config(
    page_title="Adaptive Maths Gap Lab",
    page_icon="📐",
    layout="wide",
)

SKILLS = qb.load_skills()
SKILL_IDS = [s["id"] for s in SKILLS]
SKILL_SHORT = {s["id"]: s.get("short", s["name"]) for s in SKILLS}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _init_state() -> None:
    defaults = {
        "student_id": None,
        "current_question": None,
        "q_start": None,
        "hint_used": False,
        "feedback": None,
        "extra_questions": [],
        "class_insight": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _active_student() -> dict | None:
    student_id = st.session_state.get("student_id")
    if not student_id:
        return None
    student = db.get_student(int(student_id))
    if student is None:
        st.session_state.student_id = None
    return student


def _mastery_color(value: float) -> str:
    if value >= 0.70:
        return "#16a34a"
    if value >= 0.50:
        return "#84cc16"
    if value >= config.MASTERY_THRESHOLD:
        return "#f59e0b"
    return "#dc2626"


def _disclaimer() -> None:
    st.markdown(
        """
        <div style="background:#fff7ed;border:1px solid #fdba74;border-radius:10px;
                    padding:10px 14px;margin:4px 0 14px 0;color:#7c2d12;font-size:0.92rem;">
          <strong>Please read:</strong> This tool identifies learning gaps, not clinical
          conditions. Teacher review required.
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="Preparing the risk model (first run trains it)...")
def _model_bundle() -> dict:
    return risk.ensure_model()


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def mastery_bar_figure(mastery: dict[str, float], title: str = "Mastery by skill") -> go.Figure:
    order = sorted(SKILL_IDS, key=lambda sid: mastery.get(sid, 0.0))
    values = [float(mastery.get(sid, 0.0)) for sid in order]
    labels = [SKILL_SHORT[sid] for sid in order]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=[_mastery_color(v) for v in values],
            text=[f"{v:.0%}" for v in values],
            textposition="outside",
            hovertemplate="%{y}: %{x:.0%}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        height=390,
        margin=dict(l=10, r=20, t=40, b=10),
        template="plotly_white",
        xaxis=dict(range=[0, 1.12], tickformat=".0%"),
        showlegend=False,
    )
    return fig


def mastery_radar_figure(mastery: dict[str, float]) -> go.Figure:
    values = [float(mastery.get(sid, 0.0)) for sid in SKILL_IDS]
    labels = [SKILL_SHORT[sid] for sid in SKILL_IDS]
    fig = go.Figure(
        go.Scatterpolar(
            r=values + [values[0]],
            theta=labels + [labels[0]],
            fill="toself",
            line=dict(color="#2563eb", width=2),
            fillcolor="rgba(37, 99, 235, 0.18)",
            hovertemplate="%{theta}: %{r:.0%}<extra></extra>",
        )
    )
    fig.update_layout(
        height=430,
        margin=dict(l=30, r=30, t=30, b=20),
        template="plotly_white",
        polar=dict(radialaxis=dict(range=[0, 1], tickformat=".0%")),
        showlegend=False,
    )
    return fig


def risk_gauge_figure(probability: float) -> go.Figure:
    value = round(100 * float(probability), 1)
    color = "#dc2626" if probability >= 0.66 else "#f59e0b" if probability >= 0.40 else "#16a34a"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": "%"},
            title={"text": "Risk of falling behind", "font": {"size": 15}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 40], "color": "#dcfce7"},
                    {"range": [40, 66], "color": "#fef3c7"},
                    {"range": [66, 100], "color": "#fee2e2"},
                ],
                "threshold": {
                    "line": {"color": "#111827", "width": 3},
                    "thickness": 0.75,
                    "value": 50,
                },
            },
        )
    )
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=45, b=10), template="plotly_white")
    return fig


def prereq_figure(
    mastery: dict[str, float], root_cause: str | None = None, active: str | None = None
) -> go.Figure:
    levels = qb.skill_levels()
    grouped: dict[int, list[str]] = {}
    for sid in SKILL_IDS:
        grouped.setdefault(levels.get(sid, 0), []).append(sid)

    positions: dict[str, tuple[float, float]] = {}
    for level, group in grouped.items():
        group.sort(key=lambda sid: SKILL_SHORT[sid])
        count = len(group)
        for index, sid in enumerate(group):
            positions[sid] = (level * 1.7, (count - 1) / 2.0 - index)

    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for parent, child in qb.prerequisite_edges():
        x1, y1 = positions[parent]
        x2, y2 = positions[child]
        edge_x += [x1, x2, None]
        edge_y += [y1, y2, None]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(color="#cbd5e1", width=2),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    xs = [positions[sid][0] for sid in SKILL_IDS]
    ys = [positions[sid][1] for sid in SKILL_IDS]
    colors = [_mastery_color(float(mastery.get(sid, 0.0))) for sid in SKILL_IDS]
    ring_colors = []
    ring_widths = []
    for sid in SKILL_IDS:
        if root_cause and sid == root_cause:
            ring_colors.append("#dc2626")
            ring_widths.append(4)
        elif active and sid == active:
            ring_colors.append("#2563eb")
            ring_widths.append(4)
        else:
            ring_colors.append("#ffffff")
            ring_widths.append(1.5)

    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers+text",
            marker=dict(
                size=30,
                color=colors,
                line=dict(color=ring_colors, width=ring_widths),
            ),
            text=[SKILL_SHORT[sid] for sid in SKILL_IDS],
            textposition="bottom center",
            textfont=dict(size=10),
            customdata=[
                [qb.skill_name(sid), float(mastery.get(sid, 0.0))] for sid in SKILL_IDS
            ],
            hovertemplate="%{customdata[0]}<br>Mastery %{customdata[1]:.0%}<extra></extra>",
            showlegend=False,
        )
    )

    max_level = max(levels.values()) if levels else 0
    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=30, b=10),
        template="plotly_white",
        xaxis=dict(visible=False, range=[-0.8, max_level * 1.7 + 0.9]),
        yaxis=dict(visible=False, range=[-1.5, 1.8]),
        showlegend=False,
    )
    return fig


def heatmap_figure(matrix: pd.DataFrame) -> go.Figure:
    z = matrix.to_numpy(dtype=float)
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=list(matrix.columns),
            y=list(matrix.index),
            colorscale="RdYlGn",
            zmin=0,
            zmax=1,
            colorbar=dict(title="Mastery", tickformat=".0%"),
            text=[[f"{value:.0%}" for value in row] for row in z],
            texttemplate="%{text}",
            textfont={"size": 9},
            hovertemplate="%{y}<br>%{x}: %{z:.0%}<extra></extra>",
        )
    )
    fig.update_layout(
        height=max(380, 26 * len(matrix.index) + 90),
        margin=dict(l=10, r=10, t=40, b=10),
        template="plotly_white",
        xaxis=dict(side="top", tickangle=-25),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def roc_figure(bundle: dict) -> go.Figure:
    roc = bundle["roc"]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=roc["fpr"],
            y=roc["tpr"],
            mode="lines",
            name=f"Risk model (AUC {roc['auc']:.2f})",
            line=dict(color="#2563eb", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Chance",
            line=dict(color="#9ca3af", dash="dash"),
        )
    )
    fig.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=30, b=10),
        template="plotly_white",
        xaxis_title="False positive rate",
        yaxis_title="True positive rate",
        legend=dict(orientation="h", yanchor="bottom", y=-0.3),
    )
    return fig


def confusion_figure(bundle: dict) -> go.Figure:
    matrix = bundle["confusion"]
    fig = go.Figure(
        go.Heatmap(
            z=matrix,
            x=["Predicted on track", "Predicted at risk"],
            y=["Actually on track", "Actually at risk"],
            colorscale="Blues",
            showscale=False,
            text=[[str(value) for value in row] for row in matrix],
            texttemplate="%{text}",
            textfont={"size": 22},
            hovertemplate="%{y} / %{x}: %{z}<extra></extra>",
        )
    )
    fig.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=30, b=10),
        template="plotly_white",
        yaxis=dict(autorange="reversed"),
    )
    return fig


def earliness_figure(bundle: dict) -> go.Figure:
    info = bundle["earliness"]
    n_items = int(info["n_items"])
    positions = info["positions"]
    counts = [positions.count(k) for k in range(1, n_items + 1)]
    fig = go.Figure(
        go.Bar(
            x=list(range(1, n_items + 1)),
            y=counts,
            marker_color="#0d9488",
            hovertemplate="Question %{x}: %{y} students<extra></extra>",
        )
    )
    mean_value = info.get("mean_questions")
    if mean_value is not None and not (isinstance(mean_value, float) and np.isnan(mean_value)):
        fig.add_vline(
            x=float(mean_value),
            line_dash="dash",
            line_color="#dc2626",
            annotation_text=f"mean {mean_value:.1f}",
            annotation_position="top",
        )
    fig.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=40, b=10),
        template="plotly_white",
        xaxis_title="Question number of first at-risk flag",
        yaxis_title="At-risk students",
        bargap=0.15,
    )
    return fig


def importance_figure(bundle: dict) -> go.Figure:
    items = sorted(bundle["importances"].items(), key=lambda item: item[1])
    labels = [risk.FEATURE_LABELS.get(name, name) for name, _ in items]
    values = [value for _, value in items]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color="#6366f1",
            text=[f"{value:.2f}" for value in values],
            textposition="outside",
        )
    )
    fig.update_layout(
        height=360,
        margin=dict(l=10, r=30, t=30, b=10),
        template="plotly_white",
        xaxis_title="RandomForest importance",
        xaxis=dict(range=[0, max(values + [0.01]) * 1.25]),
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar() -> None:
    with st.sidebar:
        st.title("📐 Maths Gap Lab")
        st.caption("Adaptive diagnostics for first-year engineering maths")
        st.divider()

        name = st.text_input(
            "Student name or ID",
            key="sidebar_name",
            placeholder="e.g. Asha-01",
        )
        if st.button("Use this student", use_container_width=True):
            if name.strip():
                student_id = db.get_or_create_student(name.strip())
                st.session_state.student_id = student_id
                st.session_state.current_question = None
                st.session_state.feedback = None
                st.rerun()
            else:
                st.warning("Type a name or ID first.")

        student = _active_student()
        if student:
            answered = db.count_responses(int(student["id"]))
            st.success(f"Active: {student['name']}")
            st.caption(f"{answered} question(s) recorded")
        else:
            st.info("No active student yet.")

        st.divider()

        with st.expander("About the model"):
            st.write(
                "Questions are chosen by maximum Fisher information from a 2PL IRT "
                "model. Mastery uses Bayesian Knowledge Tracing per skill. Risk uses "
                "a RandomForest trained on a synthetic cohort."
            )


# ---------------------------------------------------------------------------
# Tab 1 - Student quiz
# ---------------------------------------------------------------------------
def _render_feedback(feedback: dict) -> None:
    question = feedback["question"]
    if feedback["correct"]:
        st.success("Correct - nicely done.")
    else:
        correct_text = question["options"][int(question["answer_index"])]
        st.error(f"Not quite. The correct answer is **{correct_text}**.")

    st.markdown(f"**Question:** {question['question']}")
    explanation = str(question.get("explanation") or "").strip()
    if explanation:
        st.info(f"**Why:** {explanation}")
    if feedback.get("ai_explanation"):
        st.markdown("**Tutor note**")
        st.write(feedback["ai_explanation"])

    st.caption(
        f"Mastery for {qb.skill_name(question['skill'])} is now "
        f"{feedback['new_mastery']:.0%}."
    )

    if st.button("Next question", type="primary"):
        st.session_state.feedback = None
        st.session_state.current_question = None
        st.rerun()


def _render_progress(student_id: int) -> None:
    responses = db.get_responses(student_id)
    mastery = bkt.get_mastery(student_id)
    left, right = st.columns([3, 2])
    with left:
        st.plotly_chart(
            mastery_bar_figure(mastery), use_container_width=True, key="quiz_mastery"
        )
    with right:
        st.markdown("#### Recent answers")
        if not responses:
            st.caption("No answers yet - the first one starts your map.")
        else:
            rows = [
                {
                    "Skill": SKILL_SHORT.get(row["skill"], row["skill"]),
                    "Result": "Correct" if row["correct"] else "Wrong",
                    "Seconds": round(float(row["time_taken"]), 1),
                    "Hints": int(row["hints_used"]),
                }
                for row in responses[-8:][::-1]
            ]
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def _render_extra_generator(skill_id: str | None) -> None:
    with st.expander("Need more practice? Generate extra questions"):
        index = SKILL_IDS.index(skill_id) if skill_id in SKILL_IDS else 0
        col1, col2, col3 = st.columns(3)
        target = col1.selectbox(
            "Skill",
            SKILL_IDS,
            index=index,
            format_func=lambda sid: SKILL_SHORT.get(sid, sid),
            key="gen_skill",
        )
        difficulty = col2.selectbox("Difficulty", list(config.DIFFICULTY_TO_B), index=1, key="gen_difficulty")
        count = col3.slider("How many", 1, 3, 2, key="gen_count")

        if st.button("Generate", key="gen_button"):
            with st.spinner("Writing questions..."):
                new_items = qb.generate_questions_with_groq(target, difficulty, count)
            st.session_state.extra_questions.extend(new_items)
            st.success(f"Added {len(new_items)} question(s).")


def render_quiz() -> None:
    st.subheader("Student quiz")
    student = _active_student()
    if not student:
        st.info("Enter a student name or ID in the sidebar to begin.")
        return

    student_id = int(student["id"])
    responses = db.get_responses(student_id)
    mastery = bkt.get_mastery(student_id)
    theta, se = irt.estimate_ability(responses)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Questions answered", len(responses), help="Seed bank has 40 items.")
    col2.metric("Ability (theta)", f"{theta:+.2f}", help="MAP estimate, N(0,1) prior.")
    col3.metric("Ability percentile", f"{irt.ability_percentile(theta):.0f}th")
    col4.metric("Mean mastery", f"{np.mean(list(mastery.values())):.0%}", help="Across all ten skills.")

    st.caption(f"Standard error of theta: {se:.2f} - it shrinks as more answers arrive.")
    st.divider()

    feedback = st.session_state.get("feedback")
    if feedback:
        _render_feedback(feedback)
        st.divider()
        _render_progress(student_id)
        return

    question = st.session_state.get("current_question")
    if question is None:
        question, target_skill = qb.pick_next_question(
            student_id, mastery, responses, st.session_state.get("extra_questions")
        )
        if question is None:
            st.success(
                "You have worked through every available seed question. "
                "Generate fresh practice below or come back tomorrow."
            )
            _render_extra_generator(None)
            _render_progress(student_id)
            return
        st.session_state.current_question = question
        st.session_state.q_start = time.time()
        st.session_state.hint_used = False
        st.session_state.q_target = target_skill

    question = st.session_state.current_question
    skill = question["skill"]

    st.markdown(
        f"**Skill:** {qb.skill_name(skill)} &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"**Difficulty:** b = {float(question['difficulty']):+.1f} &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"**Answer {len(responses) + 1}**"
    )
    st.markdown(f"### {question['question']}")

    labels = [f"{chr(65 + i)}. {option}" for i, option in enumerate(question["options"])]
    choice = st.radio(
        "Choose one option",
        labels,
        index=None,
        key=f"choice_{question['id']}",
        label_visibility="collapsed",
    )

    hint_col, info_col = st.columns([1, 4])
    if hint_col.button("Show hint", key=f"hint_{question['id']}"):
        st.session_state.hint_used = True
    if st.session_state.get("hint_used"):
        info_col.info(f"Hint: {question.get('hint') or 'Write the rule you need before computing.'}")

    if st.button("Submit answer", type="primary", key=f"submit_{question['id']}"):
        if choice is None:
            st.warning("Pick one option first.")
        else:
            choice_index = labels.index(choice)
            correct = choice_index == int(question["answer_index"])
            elapsed = round(
                max(0.6, time.time() - float(st.session_state.get("q_start") or time.time())),
                2,
            )
            db.add_response(
                student_id,
                question["id"],
                skill,
                correct,
                elapsed,
                1 if st.session_state.get("hint_used") else 0,
            )
            new_mastery = bkt.update(student_id, skill, correct)

            ai_explanation = None
            if not correct:
                with st.spinner("Writing a short tutor note..."):
                    ai_explanation = llm.explain_mistake(
                        question,
                        question["options"][choice_index],
                        question["options"][int(question["answer_index"])],
                    )

            st.session_state.feedback = {
                "question": question,
                "correct": correct,
                "new_mastery": new_mastery,
                "ai_explanation": ai_explanation,
            }
            st.session_state.current_question = None
            st.session_state.hint_used = False
            st.rerun()

    st.divider()
    _render_progress(student_id)
    st.divider()
    _render_extra_generator(skill)


# ---------------------------------------------------------------------------
# Tab 2 - My report
# ---------------------------------------------------------------------------
def _build_student_summary(
    student: dict,
    mastery: dict[str, float],
    theta: float,
    risk_info: dict | None,
    root_info: dict | None,
    responses: list[dict],
) -> dict:
    weak = sorted(mastery.items(), key=lambda item: item[1])[:3]
    summary = {
        "student": {
            "name": student["name"],
            "questions_answered": len(responses),
            "ability_theta": round(theta, 2),
            "ability_percentile": round(irt.ability_percentile(theta)),
        },
        "mastery": {SKILL_SHORT[sid]: round(mastery.get(sid, 0.0), 2) for sid in SKILL_IDS},
        "weakest_skills": [skill for skill, _ in weak],
        "weakest_skills_named": [
            {"skill": qb.skill_name(skill), "mastery": round(value, 2)}
            for skill, value in weak
        ],
        "root_cause": root_info["skill"] if root_info else None,
        "root_cause_name": qb.skill_name(root_info["skill"]) if root_info else None,
    }
    if risk_info:
        summary["risk"] = {
            "probability": round(risk_info["risk"], 3),
            "band": risk_info["label"],
            "top_factors": [
                {"feature": factor["label"], "value": factor["value"]}
                for factor in risk_info["top_factors"]
            ],
        }
    return summary


def render_plan(plan: dict) -> None:
    st.markdown(f"**Root-cause skill:** {qb.skill_name(plan['root_cause_skill'])}")
    st.write(plan["why"])

    st.markdown("#### Three targeted exercises")
    columns = st.columns(3)
    for column, exercise in zip(columns, plan["exercises"]):
        with column:
            st.markdown(f"**{qb.skill_name(exercise['skill'])}**")
            st.write(exercise["task"])
            if exercise.get("hint"):
                st.caption(f"Hint: {exercise['hint']}")

    example = plan["worked_example"]
    with st.expander("Worked example", expanded=True):
        st.markdown(f"**{qb.skill_name(example['skill'])}**")
        st.write(example["problem"])
        for index, step in enumerate(example["steps"], start=1):
            st.markdown(f"{index}. {step}")
        st.success(f"Answer: {example['answer']}")


def render_report() -> None:
    st.subheader("My report")
    students = db.list_students()
    if not students:
        st.info("No students yet. Create one in the sidebar, or load the demo class in the Teacher Dashboard.")
        return

    labels = {f"{s['name']} (#{s['id']})": int(s["id"]) for s in students}
    options = list(labels)
    current = st.session_state.get("student_id")
    default_index = next(
        (i for i, label in enumerate(options) if labels[label] == current), 0
    )
    choice = st.selectbox("Student", options, index=default_index, key="report_student")
    student_id = labels[choice]
    student = db.get_student(student_id)
    if student is None:
        st.error("That student no longer exists.")
        return

    responses = db.get_responses(student_id)
    mastery = bkt.get_mastery(student_id)
    theta, se = irt.estimate_ability(responses)

    try:
        bundle = _model_bundle()
    except Exception as exc:  # training should almost never fail, but be explicit
        st.error(f"Risk model unavailable: {exc}")
        bundle = None

    risk_info = risk.predict_risk(student_id, bundle=bundle) if bundle else None
    observed = sorted(
        {row["skill"] for row in responses}, key=lambda sid: mastery.get(sid, 0.5)
    )
    target_skill = observed[0] if observed else min(mastery, key=mastery.get)
    root_info = risk.find_root_cause(target_skill, mastery)

    if not responses:
        st.warning("No answers yet - this report becomes meaningful after about five questions.")

    left, right = st.columns([1, 2])
    with left:
        if risk_info:
            st.plotly_chart(
                risk_gauge_figure(risk_info["risk"]),
                use_container_width=True,
                key="risk_gauge",
            )
            st.markdown(f"**Band:** {risk_info['label']}")
            st.caption(
                f"Based on {risk_info['n_responses']} responses. "
                "A probability, not a verdict."
            )
            st.markdown("**Top drivers**")
            for factor in risk_info["top_factors"]:
                st.markdown(
                    f"- {factor['label']}: {factor['value']} ({factor['direction']} risk)"
                )
        else:
            st.metric("Risk", "n/a")
        st.metric("Ability (theta)", f"{theta:+.2f}", help=f"SE {se:.2f}")

    with right:
        st.plotly_chart(
            mastery_radar_figure(mastery), use_container_width=True, key="mastery_radar"
        )

    st.divider()
    graph_col, detail_col = st.columns([3, 2])
    with graph_col:
        st.markdown("#### Prerequisite map")
        st.caption("Red ring = root-cause prerequisite, blue ring = current focus.")
        st.plotly_chart(
            prereq_figure(
                mastery,
                root_cause=root_info["skill"] if root_info else None,
                active=target_skill,
            ),
            use_container_width=True,
            key="prereq_graph",
        )
    with detail_col:
        st.markdown("#### Where to start")
        st.write(
            f"Current focus: **{qb.skill_name(target_skill)}** "
            f"({mastery.get(target_skill, 0.0):.0%} mastery)."
        )
        if root_info:
            if root_info["found"]:
                st.error(
                    f"Root-cause gap: **{root_info['name']}** "
                    f"({root_info['mastery']:.0%}). {root_info['reason'].capitalize()}."
                )
            else:
                st.info(
                    f"Weakest prerequisite: **{root_info['name']}** "
                    f"({root_info['mastery']:.0%}). {root_info['reason'].capitalize()}."
                )
            if len(root_info["path"]) > 1:
                st.caption(
                    "Chain: "
                    + " -> ".join(SKILL_SHORT.get(sid, sid) for sid in root_info["path"])
                )

    st.divider()
    st.markdown("#### Personalised study plan")
    summary = _build_student_summary(
        student, mastery, theta, risk_info, root_info, responses
    )
    plan_key = f"plan_{student_id}"
    if st.button("Generate personalised plan", type="primary", key=f"plan_btn_{student_id}"):
        with st.spinner("Building a plan..."):
            st.session_state[plan_key] = llm.personalized_plan(summary)

    plan = st.session_state.get(plan_key)
    if plan:
        render_plan(plan)
    else:
        st.caption("Generate a plan to get a root-cause skill, three exercises and a worked example.")


# ---------------------------------------------------------------------------
# Tab 3 - Teacher dashboard
# ---------------------------------------------------------------------------
def _load_demo_class(count: int) -> int:
    db.delete_students_like("Demo Student%")
    st.session_state.student_id = None
    records = simulator.generate_demo_class(n=count)
    start = datetime.now()
    for record in records:
        student_id = db.create_student(record["name"])
        for index, response in enumerate(record["responses"]):
            db.add_response(
                student_id,
                response["question_id"],
                response["skill"],
                response["correct"],
                response["time_taken"],
                response["hints_used"],
                timestamp=(start + timedelta(minutes=3 * index)).isoformat(timespec="seconds"),
            )
        for skill, value in record["mastery"].items():
            db.save_mastery_snapshot(
                student_id,
                skill,
                value,
                timestamp=(start + timedelta(minutes=3 * len(record["responses"]))).isoformat(timespec="seconds"),
            )
    return len(records)


def render_teacher() -> None:
    st.subheader("Teacher dashboard")
    action_col, spacer = st.columns([1.5, 2.5])
    if action_col.button("Load synthetic demo class", type="primary"):
        with st.spinner("Simulating 30 students and writing their responses..."):
            count = _load_demo_class(config.DEMO_CLASS_SIZE)
        st.session_state.class_insight = None
        st.success(f"Loaded {count} demo students.")
        st.rerun()

    students = db.list_students()
    if not students:
        st.info(
            "No students yet. Load the synthetic demo class above, or add a student "
            "in the sidebar and answer a few questions."
        )
        return

    try:
        bundle = _model_bundle()
    except Exception as exc:
        st.error(f"Risk model unavailable: {exc}")
        return

    latest = db.get_all_mastery()
    names = [student["name"] for student in students]
    defaults = {sid: bkt.params_for(sid)["p_init"] for sid in SKILL_IDS}
    matrix = pd.DataFrame(
        [
            [latest.get((name, sid), defaults[sid]) for sid in SKILL_IDS]
            for name in names
        ],
        index=names,
        columns=SKILL_IDS,
    )
    display_matrix = matrix.rename(columns=SKILL_SHORT)

    st.markdown("#### Class mastery heatmap")
    st.plotly_chart(
        heatmap_figure(display_matrix), use_container_width=True, key="class_heatmap"
    )

    rows = []
    root_id_by_student: dict[str, str] = {}
    for student in students:
        student_id = int(student["id"])
        try:
            info = risk.predict_risk(student_id, bundle=bundle)
        except Exception:
            continue
        root = info.get("root_cause")
        if root:
            root_id_by_student[student["name"]] = root["skill"]
        rows.append(
            {
                "Student": student["name"],
                "Risk %": round(100 * info["risk"], 1),
                "Band": info["label"],
                "Avg mastery %": round(
                    100 * float(matrix.loc[student["name"]].mean()), 1
                ),
                "Weakest skill": qb.skill_short(info["weakest_skill"])
                if info["weakest_skill"]
                else "-",
                "Root cause": root["name"] if root else "-",
                "Answers": info["n_responses"],
            }
        )

    table = pd.DataFrame(rows).sort_values("Risk %", ascending=False)
    st.markdown("#### Students needing support first")
    st.caption("Sorted by predicted risk. Teacher review required before any action.")
    st.dataframe(
        table,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Risk %": st.column_config.ProgressColumn(
                "Risk %", min_value=0.0, max_value=100.0, format="%.0f"
            ),
            "Avg mastery %": st.column_config.ProgressColumn(
                "Avg mastery %", min_value=0.0, max_value=100.0, format="%.0f"
            ),
        },
    )

    st.divider()
    st.markdown("#### Class insight")
    column_means = matrix.mean().sort_values()
    weakest = [
        {
            "skill": skill,
            "name": qb.skill_name(skill),
            "mastery": float(value),
        }
        for skill, value in column_means.head(3).items()
    ]
    at_risk_rows = table[table["Risk %"] >= 50].head(8)
    stats = {
        "n_students": len(students),
        "n_at_risk": int((table["Risk %"] >= 50).sum()),
        "class_avg_mastery": round(float(matrix.to_numpy().mean()), 3),
        "weakest_skills": weakest,
        "at_risk": [
            {
                "name": row["Student"],
                "risk": round(float(row["Risk %"]) / 100, 2),
                "weakest_skill": row["Weakest skill"],
                "root_cause": root_id_by_student.get(row["Student"]),
            }
            for _, row in at_risk_rows.iterrows()
        ],
    }

    if st.button("Generate class summary", key="class_summary_button"):
        with st.spinner("Reading the class snapshot..."):
            st.session_state.class_insight = llm.teacher_summary(stats)

    if st.session_state.get("class_insight"):
        st.markdown(
            st.session_state.class_insight.replace("\n", "  \n")
        )
        st.caption("AI summary of learning gaps - teacher review required.")
    else:
        st.caption("Generate a five-line summary of the shared gaps and next lesson.")


# ---------------------------------------------------------------------------
# Tab 4 - Model evaluation
# ---------------------------------------------------------------------------
def render_evaluation() -> None:
    st.subheader("Model evaluation")
    try:
        bundle = _model_bundle()
    except Exception as exc:
        st.error(f"Could not load the risk model: {exc}")
        return

    metrics = bundle["metrics"]
    baseline = bundle["baseline"]

    st.caption(
        f"Trained {bundle['trained_at']} on {bundle['n_train_students']} synthetic "
        f"students ({bundle['n_train']} snapshots), tested on {bundle['n_test_students']}. "
        f"Hidden label: mean final mastery < 0.50 ({bundle['at_risk_rate']:.0%} at risk)."
    )

    comparison = pd.DataFrame(
        [
            {
                "Model": "RandomForest (7 features)",
                "Accuracy": round(metrics["accuracy"], 3),
                "Precision": round(metrics["precision"], 3),
                "Recall": round(metrics["recall"], 3),
                "F1": round(metrics["f1"], 3),
                "ROC-AUC": round(metrics["roc_auc"], 3),
            },
            {
                "Model": "Baseline: recent accuracy < 50%",
                "Accuracy": round(baseline["accuracy"], 3),
                "Precision": round(baseline["precision"], 3),
                "Recall": round(baseline["recall"], 3),
                "F1": round(baseline["f1"], 3),
                "ROC-AUC": round(baseline["roc_auc"], 3),
            },
        ]
    )
    st.dataframe(comparison, hide_index=True, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### ROC curve")
        st.plotly_chart(roc_figure(bundle), use_container_width=True, key="eval_roc")
    with col2:
        st.markdown("#### Confusion matrix")
        st.plotly_chart(confusion_figure(bundle), use_container_width=True, key="eval_cm")

    earliness = bundle["earliness"]
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("#### Earliness")
        st.plotly_chart(earliness_figure(bundle), use_container_width=True, key="eval_early")
        st.caption(
            f"Flagged {earliness['flagged_fraction']:.0%} of at-risk test students after "
            f"a mean of {earliness['mean_questions']:.1f} of {earliness['n_items']} questions."
        )
    with col4:
        st.markdown("#### What the model relies on")
        st.plotly_chart(importance_figure(bundle), use_container_width=True, key="eval_imp")

    st.info(
        "Baseline comparison: the model needs to beat 'recent accuracy below 50%' "
        "while flagging students earlier in the session. Earliness counts the first "
        "question at which predicted risk crosses 50%."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    _init_state()
    st.title("Adaptive Maths Gap Lab")
    st.caption(
        "First-year engineering mathematics - adaptive practice, early gap detection "
        "and personalised next steps."
    )
    _disclaimer()
    render_sidebar()

    tab_quiz, tab_report, tab_teacher, tab_eval = st.tabs(
        ["Student Quiz", "My Report", "Teacher Dashboard", "Model Evaluation"]
    )
    with tab_quiz:
        render_quiz()
    with tab_report:
        render_report()
    with tab_teacher:
        render_teacher()
    with tab_eval:
        render_evaluation()


if __name__ == "__main__":
    main()
