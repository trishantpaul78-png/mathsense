"""
All Groq API traffic lives in this module.

Design rules
------------
* Every public function returns a usable value even when GROQ_API_KEY is
  missing, the network is down, the call times out, or the model returns
  malformed JSON. The local fallbacks are deterministic and honest.
* Prompts carry a hard guardrail: this tool identifies LEARNING GAPS only.
  It never diagnoses or names clinical conditions (dyscalculia, ADHD, ...)
  and never labels the learner.
* Model name and timeout come from config.py.
"""

import json
import logging
import re

import config

logger = logging.getLogger(__name__)

try:  # The app must import cleanly even if the groq package is absent.
    from groq import Groq

    _GROQ_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment dependent
    Groq = None
    _GROQ_IMPORT_ERROR = exc

SYSTEM_GUARDRAIL = (
    "You are a supportive mathematics teaching assistant inside a learning-analytics "
    "tool for first-year engineering students. You identify LEARNING GAPS only. "
    "Never diagnose, name, or hint at clinical conditions such as dyscalculia, "
    "dyslexia, ADHD, or any learning disorder. Never label the student (do not use "
    "'weak', 'slow', 'failing', 'bad at maths', 'learning disabled'). "
    "Talk about specific skills, what to practise next, and small concrete steps. "
    "Use encouraging, plain, non-judgemental language. Keep maths notation ASCII."
)


# ---------------------------------------------------------------------------
# Low-level transport
# ---------------------------------------------------------------------------
def is_available() -> bool:
    """True when a key is configured and the SDK imported."""
    return bool(config.GROQ_API_KEY) and Groq is not None


def status_message() -> str:
    if Groq is None:
        return f"groq package unavailable ({_GROQ_IMPORT_ERROR})"
    if not config.GROQ_API_KEY:
        return "no GROQ_API_KEY set - using local fallbacks"
    return f"Groq ready ({active_model()})"


_client = None

# Runtime model resolution: the first model that answers successfully is reused
# for the rest of the session; models Groq reports as unavailable are retired.
_working_model: str | None = None
_retired_models: set[str] = set()


def _get_client():
    global _client
    if _client is None:
        _client = Groq(api_key=config.GROQ_API_KEY, timeout=config.GROQ_TIMEOUT_SECONDS)
    return _client


def active_model() -> str:
    """The model actually in use right now (working model, else the configured one)."""
    return _working_model or config.GROQ_MODEL


def _candidate_models() -> list[str]:
    """Configured model first, then the fallback chain; retired models removed."""
    chain = [config.GROQ_MODEL, *getattr(config, "GROQ_MODEL_FALLBACKS", [])]
    ordered, seen = [], set()
    for name in chain:
        name = (name or "").strip()
        if name and name not in seen and name not in _retired_models:
            seen.add(name)
            ordered.append(name)
    if _working_model and _working_model in ordered:
        ordered.remove(_working_model)
        ordered.insert(0, _working_model)
    return ordered


def _is_model_error(exc: Exception) -> bool:
    """True for 'this account/model cannot be used' rather than a transient fault."""
    text = str(exc).lower()
    return any(
        marker in text
        for marker in ("model_not_found", "does not exist", "do not have access", "decommissioned")
    )


def _request(model: str, messages: list[dict], max_tokens: int, temperature: float,
             json_mode: bool):
    """One raw completion. Returns (text, finish_reason)."""
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": config.GROQ_TIMEOUT_SECONDS,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    # Reasoning models (gpt-oss) spend part of the budget thinking; the short
    # replies this app asks for get truncated without this hint.
    if getattr(config, "GROQ_REASONING_EFFORT", "") and "gpt-oss" in model:
        kwargs["reasoning_effort"] = config.GROQ_REASONING_EFFORT
    response = _get_client().chat.completions.create(**kwargs)
    choice = response.choices[0]
    return (choice.message.content or "").strip(), getattr(choice, "finish_reason", None)


def _call_model(model: str, messages: list[dict], max_tokens: int, temperature: float,
                json_mode: bool) -> str | None:
    """Call once; if the answer was cut off by the token limit, retry bigger."""
    text, finish = _request(model, messages, max_tokens, temperature, json_mode)
    if finish == "length":
        retry_budget = min(max(max_tokens * 3, 900), 4000)
        logger.info("Answer hit the token cap (%s), retrying with %s", model, retry_budget)
        text, finish = _request(model, messages, retry_budget, temperature, json_mode)
        if finish == "length":
            logger.warning("Answer still truncated at %s tokens (%s)", retry_budget, model)
            return None
    return text or None


def _chat(
    prompt: str,
    system: str = SYSTEM_GUARDRAIL,
    max_tokens: int | None = None,
    temperature: float = 0.4,
    json_mode: bool = False,
) -> str | None:
    """One chat completion. Returns None on any failure (never raises)."""
    global _working_model
    if not is_available():
        return None

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    tokens = max_tokens or config.GROQ_MAX_TOKENS

    for model in _candidate_models():
        try:
            text = _call_model(model, messages, tokens, temperature, json_mode)
            if text:
                _working_model = model
                return text
        except Exception as exc:
            if _is_model_error(exc):
                logger.warning("Groq model %s unavailable, trying next: %s", model, exc)
                _retired_models.add(model)
                if _working_model == model:
                    _working_model = None
                continue
            logger.warning("Groq call failed (%s): %s", model, exc)

        if json_mode:
            # Some models/accounts reject response_format; retry once in plain mode.
            try:
                text = _call_model(model, messages, tokens, temperature, False)
                if text:
                    _working_model = model
                    return text
            except Exception as exc:
                logger.warning("Groq retry failed (%s): %s", model, exc)
                if _is_model_error(exc):
                    _retired_models.add(model)
                    if _working_model == model:
                        _working_model = None
                    continue
        return None

    return None


def _extract_json(text: str | None) -> str | None:
    """Pull the first JSON object/array out of a model reply (handles fences)."""
    if not text:
        return None
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    starts = [i for i in (cleaned.find("{"), cleaned.find("[")) if i != -1]
    if not starts:
        return None
    start = min(starts)
    end_curly = cleaned.rfind("}")
    end_square = cleaned.rfind("]")
    end = max(end_curly, end_square)
    if end <= start:
        return None
    return cleaned[start : end + 1]


def chat_json(
    prompt: str,
    max_tokens: int | None = None,
    temperature: float = 0.6,
    system: str = SYSTEM_GUARDRAIL,
) -> str | None:
    """Ask for strict JSON and return the JSON text (or None)."""
    return _extract_json(
        _chat(
            prompt,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            json_mode=True,
        )
    )


# ---------------------------------------------------------------------------
# Skill helpers (lazy import avoids a question_bank <-> llm import cycle)
# ---------------------------------------------------------------------------
def _skill_menu() -> tuple[list[str], dict[str, str]]:
    import question_bank

    skills = question_bank.load_skills()
    ids = [s["id"] for s in skills]
    names = {s["id"]: s["name"] for s in skills}
    return ids, names


def _normalise_skill(value) -> str | None:
    """Map an id, full name or short name returned by the model to a skill id."""
    if not value:
        return None
    text = str(value).strip().lower()
    ids, names = _skill_menu()
    for skill_id in ids:
        if text == skill_id.lower():
            return skill_id
    for skill_id, full_name in names.items():
        if text == full_name.lower():
            return skill_id
    import question_bank

    for skill in question_bank.load_skills():
        if text == str(skill.get("short", "")).lower():
            return skill["id"]
    # Fuzzy last resort: substring either way.
    for skill_id, full_name in names.items():
        if text and (text in full_name.lower() or full_name.lower() in text):
            return skill_id
    return None


# ---------------------------------------------------------------------------
# 1) Mistake explanation
# ---------------------------------------------------------------------------
def explain_mistake(question: dict, student_answer: str, correct_answer: str) -> str:
    """
    Short, friendly explanation of one wrong answer.

    `question` is a question-bank dict; answers are option texts.
    Falls back to the seeded explanation when the API is unavailable.
    """
    question_text = str(question.get("question", "")).strip()
    skill = str(question.get("skill", "")).replace("_", " ")
    options = question.get("options") or []

    prompt = f"""A first-year engineering student answered this multiple-choice question.

Skill: {skill}
Question: {question_text}
Options: {json.dumps(options)}
Student chose: {student_answer}
Correct answer: {correct_answer}

Write 2-4 sentences (max 90 words) that:
1. say plainly which idea the student's choice missed,
2. give the key rule or formula in one line,
3. end with one concrete next step (a small practice action).
Warm and specific. No labels about the student. No clinical language."""

    text = _chat(prompt, max_tokens=700, temperature=0.35)
    if text:
        return text.strip()

    seeded = str(question.get("explanation", "")).strip()
    message = f"Not quite - the correct answer is {correct_answer}."
    if seeded:
        message += f" {seeded}"
    return (
        message
        + " Try one similar question and say the rule out loud before you start; "
        "that usually locks the idea in."
    )


# ---------------------------------------------------------------------------
# 2) Personalised study plan (strict JSON)
# ---------------------------------------------------------------------------
PLAN_SKILLS_NOTE = "Use the skill ids exactly as listed."


def _validate_plan(data, allowed: list[str]) -> dict | None:
    """Normalise + validate the plan JSON. Returns None if it is unusable."""
    if not isinstance(data, dict):
        return None

    root = _normalise_skill(data.get("root_cause_skill") or data.get("root_cause"))
    if root is None or root not in allowed:
        return None

    why = str(data.get("why") or "").strip()
    if not why:
        return None

    exercises_in = data.get("exercises")
    if not isinstance(exercises_in, list) or len(exercises_in) < 3:
        return None
    exercises = []
    for item in exercises_in[:3]:
        if isinstance(item, str):
            item = {"task": item}
        if not isinstance(item, dict):
            return None
        task = str(item.get("task") or item.get("question") or "").strip()
        if not task:
            return None
        exercises.append(
            {
                "skill": _normalise_skill(item.get("skill")) or root,
                "task": task,
                "hint": str(item.get("hint") or "").strip(),
            }
        )

    example = data.get("worked_example")
    if not isinstance(example, dict):
        return None
    problem = str(example.get("problem") or "").strip()
    steps_in = example.get("steps")
    if not problem or not isinstance(steps_in, list) or not steps_in:
        return None
    steps = [str(s).strip() for s in steps_in if str(s).strip()]
    if not steps:
        return None
    answer = str(example.get("answer") or "").strip()
    if not answer:
        return None

    try:
        minutes = int(float(data.get("estimated_time_minutes", 40)))
    except (TypeError, ValueError):
        minutes = 40
    minutes = max(15, min(120, minutes))

    return {
        "root_cause_skill": root,
        "why": why,
        "exercises": exercises,
        "worked_example": {
            "skill": _normalise_skill(example.get("skill")) or root,
            "problem": problem,
            "steps": steps,
            "answer": answer,
        },
        "estimated_time_minutes": minutes,
    }


def _fallback_plan(summary: dict) -> dict:
    """Deterministic plan built from the local question bank (no AI)."""
    import question_bank

    allowed = question_bank.skill_ids()
    root = _normalise_skill(
        summary.get("root_cause") or summary.get("root_cause_skill")
    )
    if root not in allowed:
        weak = summary.get("weakest_skills") or []
        root = _normalise_skill(weak[0]) if weak else allowed[0]
    if root not in allowed:
        root = allowed[0]

    pool = question_bank.questions_for_skill(root)
    exercises = []
    for question in pool[:3]:
        exercises.append(
            {
                "skill": root,
                "task": question["question"],
                "hint": question.get("hint", "Write the rule you need before computing."),
            }
        )
    while len(exercises) < 3:
        exercises.append(
            {
                "skill": root,
                "task": "Re-derive the main formula for this skill from first principles, "
                "then apply it to one number of your choice.",
                "hint": "Say each step out loud; stop at the first step you cannot justify.",
            }
        )

    example_source = pool[0] if pool else None
    if example_source:
        worked = {
            "skill": root,
            "problem": example_source["question"],
            "steps": [example_source.get("explanation", "Apply the rule step by step.")],
            "answer": example_source["options"][int(example_source["answer_index"])],
        }
    else:
        worked = {
            "skill": root,
            "problem": f"Work one standard exercise on {root.replace('_', ' ')}.",
            "steps": ["Write the definition.", "Substitute the given values.", "Simplify."],
            "answer": "Compare with your lecture notes.",
        }

    name = question_bank.skill_name(root)
    plan = {
        "root_cause_skill": root,
        "why": (
            f"Your recent work points to {name} as the earliest gap. "
            "Strengthening it makes the later topics much lighter."
        ),
        "exercises": exercises[:3],
        "worked_example": worked,
        "estimated_time_minutes": 40,
    }
    return plan


def personalized_plan(student_summary_json) -> dict:
    """
    Build a JSON study plan: root cause, 3 exercises, 1 worked example, time.

    Returns a dict with keys root_cause_skill, why, exercises, worked_example,
    estimated_time_minutes, plus 'source' ('groq' or 'local-fallback').
    """
    if isinstance(student_summary_json, str):
        try:
            summary = json.loads(student_summary_json)
        except json.JSONDecodeError:
            summary = {}
    else:
        summary = dict(student_summary_json or {})

    allowed, names = _skill_menu()
    skill_lines = "\n".join(f"- {sid}: {names[sid]}" for sid in allowed)

    prompt = f"""Here is a compact learning-gap summary for one first-year engineering student.

{json.dumps(summary, indent=2)[:4000]}

Available skills (use the id exactly):
{skill_lines}

Return STRICT JSON only, no prose, with this exact shape:
{{
  "root_cause_skill": "<one skill id from the list>",
  "why": "<2 sentences, encouraging, about the gap and why it matters>",
  "exercises": [
    {{"skill": "<skill id>", "task": "<one clear exercise>", "hint": "<one short hint>"}},
    {{"skill": "<skill id>", "task": "<one clear exercise>", "hint": "<one short hint>"}},
    {{"skill": "<skill id>", "task": "<one clear exercise>", "hint": "<one short hint>"}}
  ],
  "worked_example": {{
    "skill": "<skill id>",
    "problem": "<a short worked problem>",
    "steps": ["step 1", "step 2", "step 3"],
    "answer": "<final answer>"
  }},
  "estimated_time_minutes": 40
}}

Rules: exactly 3 exercises; realistic first-year engineering maths; no diagnosis or
clinical language; do not label the student; concrete numbers where possible."""

    raw = chat_json(prompt, max_tokens=1400, temperature=0.5)
    plan = None
    if raw:
        try:
            plan = _validate_plan(json.loads(raw), allowed)
        except json.JSONDecodeError:
            logger.warning("Plan response was not valid JSON")

    if plan is None:
        plan = _fallback_plan(summary)
        plan["source"] = "local-fallback"
    else:
        plan["source"] = "groq"
    return plan


# ---------------------------------------------------------------------------
# 3) Teacher class insight
# ---------------------------------------------------------------------------
def teacher_summary(class_stats_json) -> str:
    """Five-line class insight. Always returns five usable lines."""
    if isinstance(class_stats_json, str):
        try:
            stats = json.loads(class_stats_json)
        except json.JSONDecodeError:
            stats = {}
    else:
        stats = dict(class_stats_json or {})

    prompt = f"""Class snapshot for a first-year engineering mathematics cohort:

{json.dumps(stats, indent=2)[:4000]}

Write exactly 5 short lines (max 22 words each), plain text, no numbering,
no markdown bullets. Cover: overall picture, the main shared gap, which
prerequisite to revisit, one grouping suggestion, and one concrete next lesson.
Encouraging and non-labelling: talk about skills and gaps, never about
student character or clinical conditions."""

    text = _chat(prompt, max_tokens=800, temperature=0.4)
    if text:
        lines = [ln.strip(" -*\t") for ln in text.splitlines() if ln.strip()]
        if len(lines) >= 5:
            return "\n".join(lines[:5])

    return _fallback_teacher_summary(stats)


def _fallback_teacher_summary(stats: dict) -> str:
    import question_bank

    n_students = int(stats.get("n_students") or 0)
    n_risk = int(stats.get("n_at_risk") or 0)
    avg = float(stats.get("class_avg_mastery") or 0.0)
    weakest = stats.get("weakest_skills") or []

    if weakest and isinstance(weakest[0], dict):
        main = weakest[0]
        main_name = question_bank.skill_name(main.get("skill", "")) or "the core skills"
        main_value = float(main.get("mastery") or 0.0)
    else:
        main_name = question_bank.skill_name(str(weakest[0])) if weakest else "the core skills"
        main_value = 0.0

    root_counts: dict[str, int] = {}
    for row in stats.get("at_risk") or []:
        root = row.get("root_cause") if isinstance(row, dict) else None
        if root:
            root_counts[root] = root_counts.get(root, 0) + 1
    common_root = max(root_counts, key=root_counts.get) if root_counts else None
    root_name = question_bank.skill_name(common_root) if common_root else main_name

    share = (n_risk / n_students) if n_students else 0.0
    lines = [
        f"{n_students} students are in this snapshot, with mean mastery {avg:.0%} across the ten skills.",
        f"{n_risk} students ({share:.0%}) would benefit from early support in the next session.",
        f"The most shared gap is {main_name} ({main_value:.0%} mastery), which feeds several later topics.",
        f"Gaps most often trace back to {root_name}, so a short prerequisite revisit is worthwhile.",
        "Suggested next step: a 10-minute prerequisite warm-up, then paired practice on the shared gap.",
    ]
    return "\n".join(lines)
