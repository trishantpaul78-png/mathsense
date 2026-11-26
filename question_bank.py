"""
Question bank: 40 seeded MCQs plus Groq-generated items.

Responsibilities
----------------
* Load data/skills.json (skills + prerequisite DAG helpers).
* Load data/questions.json (the seed bank used offline and as a fallback).
* generate_questions_with_groq(skill, difficulty_label, n) - asks Groq for
  STRICT JSON, validates every item, retries once, then falls back to seeds.
* Adaptive selection: maximum IRT information inside the weakest skill that
  still has unseen items.
"""

import json
import random
import uuid
from functools import lru_cache

import config
import irt
import llm


# ---------------------------------------------------------------------------
# Skills / prerequisite DAG
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_skills() -> list[dict]:
    with open(config.SKILLS_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)["skills"]


def skill_ids() -> list[str]:
    return [skill["id"] for skill in load_skills()]


def skill_name(skill_id: str) -> str:
    for skill in load_skills():
        if skill["id"] == skill_id:
            return skill["name"]
    return str(skill_id).replace("_", " ").title()


def skill_short(skill_id: str) -> str:
    for skill in load_skills():
        if skill["id"] == skill_id:
            return skill.get("short", skill["name"])
    return str(skill_id)


def skill_by_id(skill_id: str) -> dict | None:
    for skill in load_skills():
        if skill["id"] == skill_id:
            return skill
    return None


def prerequisites(skill_id: str) -> list[str]:
    skill = skill_by_id(skill_id)
    return list(skill.get("prerequisites", [])) if skill else []


@lru_cache(maxsize=1)
def prerequisite_edges() -> tuple[tuple[str, str], ...]:
    """(prerequisite, dependent) pairs derived from the skills file."""
    edges = []
    for skill in load_skills():
        for parent in skill.get("prerequisites", []):
            edges.append((parent, skill["id"]))
    return tuple(edges)


@lru_cache(maxsize=1)
def skill_levels() -> dict[str, int]:
    """Longest-path depth from a root; used to lay out the DAG."""
    levels: dict[str, int] = {}
    visiting: set[str] = set()

    def level(skill_id: str) -> int:
        if skill_id in levels:
            return levels[skill_id]
        if skill_id in visiting:  # defensive: treat a malformed cycle as a root
            return 0
        visiting.add(skill_id)
        parents = prerequisites(skill_id)
        levels[skill_id] = 0 if not parents else 1 + max(level(p) for p in parents)
        visiting.discard(skill_id)
        return levels[skill_id]

    for sid in skill_ids():
        level(sid)
    return levels


def ancestors(skill_id: str) -> set[str]:
    """All transitive prerequisites of a skill (excluding itself)."""
    found: set[str] = set()
    stack = list(prerequisites(skill_id))
    while stack:
        node = stack.pop()
        if node in found:
            continue
        found.add(node)
        stack.extend(prerequisites(node))
    return found


def path_between(start: str, goal: str) -> list[str]:
    """Shortest prerequisite-chain path from start to goal (inclusive)."""
    if start == goal:
        return [start]
    children: dict[str, list[str]] = {}
    for parent, child in prerequisite_edges():
        children.setdefault(parent, []).append(child)

    queue: list[list[str]] = [[start]]
    seen = {start}
    while queue:
        path = queue.pop(0)
        node = path[-1]
        if node == goal:
            return path
        for child in children.get(node, []):
            if child not in seen:
                seen.add(child)
                queue.append(path + [child])
    return [start, goal]


# ---------------------------------------------------------------------------
# Seed questions
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_questions() -> list[dict]:
    with open(config.QUESTIONS_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)["questions"]


@lru_cache(maxsize=1)
def _question_index() -> dict[str, dict]:
    return {question["id"]: question for question in load_questions()}


def get_question(question_id: str) -> dict | None:
    return _question_index().get(question_id)


def questions_for_skill(skill_id: str) -> list[dict]:
    return [q for q in load_questions() if q["skill"] == skill_id]


# ---------------------------------------------------------------------------
# Groq generation with validation + retry + fallback
# ---------------------------------------------------------------------------
def _validate_item(item: dict, skill: str, b_default: float) -> dict | None:
    if not isinstance(item, dict):
        return None

    text = str(item.get("question") or item.get("stem") or "").strip()
    if len(text) < 8:
        return None

    options = item.get("options")
    if not isinstance(options, list) or len(options) != 4:
        return None
    options = [str(option).strip() for option in options]
    if any(not option for option in options) or len(set(options)) != 4:
        return None

    try:
        answer_index = int(item.get("answer_index", item.get("correct_index", -1)))
    except (TypeError, ValueError):
        return None
    if answer_index not in range(4):
        return None

    try:
        difficulty = float(item.get("difficulty", item.get("b", b_default)))
    except (TypeError, ValueError):
        difficulty = b_default
    difficulty = max(-2.0, min(2.0, difficulty))

    try:
        discrimination = float(item.get("discrimination", item.get("a", 1.2)))
    except (TypeError, ValueError):
        discrimination = 1.2
    discrimination = max(0.5, min(2.0, discrimination))

    return {
        "id": f"gen_{skill}_{uuid.uuid4().hex[:8]}",
        "skill": skill,
        "question": text,
        "options": options,
        "answer_index": answer_index,
        "difficulty": round(difficulty, 2),
        "discrimination": round(discrimination, 2),
        "explanation": str(item.get("explanation") or "").strip(),
        "hint": str(item.get("hint") or "").strip(),
    }


def _parse_generated(raw_json: str, skill: str, b_default: float) -> list[dict]:
    data = json.loads(raw_json)
    if isinstance(data, dict):
        data = data.get("questions") or data.get("items") or [data]
    if not isinstance(data, list):
        return []

    validated = []
    for item in data:
        parsed = _validate_item(item, skill, b_default)
        if parsed:
            validated.append(parsed)
    return validated


def _seed_fallback(skill: str, n: int) -> list[dict]:
    pool = questions_for_skill(skill) or load_questions()
    count = max(1, min(int(n), len(pool)))
    picks = random.sample(pool, count)
    return [dict(question, _source="seed") for question in picks]


def generate_questions_with_groq(skill: str, difficulty_label: str = "medium", n: int = 3) -> list[dict]:
    """
    Generate `n` MCQs for a skill with Groq.

    Strict JSON is requested and every item is validated (4 options, in-range
    answer index, numeric a/b). One retry happens on malformed output; after
    that the seed bank is returned so the quiz never stalls.
    """
    n = max(1, int(n))
    b_default = config.DIFFICULTY_TO_B.get(str(difficulty_label).lower(), 0.0)
    skill_label = skill_name(skill)
    skill_row = skill_by_id(skill) or {}
    keywords = ", ".join(skill_row.get("keywords", []))

    prompt = f"""Write {n} multiple-choice questions on {skill_label} for first-year
engineering students. Difficulty target: {difficulty_label} (IRT b about {b_default}).
Keywords to stay within: {keywords or skill_label}.

Return STRICT JSON only:
{{
  "questions": [
    {{
      "question": "text",
      "options": ["A", "B", "C", "D"],
      "answer_index": 0,
      "difficulty": {b_default},
      "discrimination": 1.2,
      "explanation": "why the answer is right, 1-2 sentences",
      "hint": "one short nudge"
    }}
  ]
}}

Rules: exactly 4 options, exactly one correct, answer_index in 0..3,
difficulty in [-2, 2], discrimination in [0.5, 2], all maths ASCII,
no duplicate options, no references to clinical conditions."""

    if llm.is_available():
        for attempt in (1, 2):
            raw = llm.chat_json(prompt, max_tokens=1800, temperature=0.7)
            if not raw:
                continue
            try:
                parsed = _parse_generated(raw, skill, b_default)
            except (json.JSONDecodeError, TypeError, ValueError):
                parsed = []
            if parsed:
                for question in parsed:
                    question["_source"] = "groq"
                return parsed[:n]

    return _seed_fallback(skill, n)


# ---------------------------------------------------------------------------
# Adaptive selection
# ---------------------------------------------------------------------------
def pick_next_question(
    student_id: int,
    mastery: dict[str, float],
    responses: list[dict],
    extra_questions: list[dict] | None = None,
) -> tuple[dict | None, str | None]:
    """
    Choose the next item.

    Strategy: order skills weakest-first, and inside the first skill that still
    has unseen items pick the maximum-information item at the current theta.
    Extra (AI-generated) questions participate in the same pool.
    """
    del student_id  # kept for a stable API; all state comes from the arguments
    asked = {str(row.get("question_id")) for row in responses}
    pool = list(load_questions()) + list(extra_questions or [])
    theta, _ = irt.estimate_ability(responses)

    ordered_skills = sorted(
        skill_ids(), key=lambda sid: float(mastery.get(sid, 0.5))
    )
    for skill in ordered_skills:
        candidates = [
            question
            for question in pool
            if question["skill"] == skill and question["id"] not in asked
        ]
        if candidates:
            return irt.select_next_question(theta, candidates, asked), skill
    return None, None


def remaining_counts(asked_ids: set[str], extra_questions: list[dict] | None = None) -> dict[str, int]:
    """Unseen seed questions per skill (teacher-facing diagnostic)."""
    counts = {skill: 0 for skill in skill_ids()}
    for question in load_questions():
        if question["id"] not in asked_ids:
            counts[question["skill"]] = counts.get(question["skill"], 0) + 1
    for question in extra_questions or []:
        counts[question["skill"]] = counts.get(question["skill"], 0) + 1
    return counts
