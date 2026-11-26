# Adaptive Maths Gap Lab

An adaptive, AI-assisted system for **early identification of learning gaps** and
**personalised intervention** in first-year engineering mathematics.

It combines four pieces:

1. **2PL Item Response Theory** to estimate ability and pick the most informative
   next question.
2. **Bayesian Knowledge Tracing** to maintain a mastery probability per skill.
3. A **synthetic-cohort risk classifier** (RandomForest) that flags students likely
   to fall behind, with a **prerequisite-DAG root-cause finder**.
4. **Groq-hosted LLM** (`openai/gpt-oss-120b` by default) for mistake explanations, study plans and class
   summaries - always optional, always with a local fallback.

> **This tool identifies learning gaps, not clinical conditions. Teacher review required.**
> It is not a diagnostic instrument and must never be used to label a student or to
> suggest a learning disorder.

---

## 1. What is included

| Module | Purpose |
|---|---|
| `app.py` | Streamlit UI: Student Quiz, My Report, Teacher Dashboard, Model Evaluation |
| `config.py` | Paths, Groq model/timeout, BKT priors, thresholds, simulation sizes |
| `db.py` | SQLite tables and helpers: `students`, `responses`, `mastery_snapshots` |
| `irt.py` | 2PL probability, MAP/Newton ability estimation, item information, selection |
| `bkt.py` | BKT update equations, per-skill parameters, persisted mastery history |
| `risk.py` | Seven risk features, RandomForest inference, top-3 drivers, root-cause walk |
| `simulator.py` | 200 synthetic students, train/test split, metrics, earliness, baseline |
| `llm.py` | Every Groq call, prompt guardrails, JSON validation, graceful fallbacks |
| `question_bank.py` | 40 seeded MCQs, Groq generation with retry, adaptive picking |
| `data/skills.json` | 10 skills and the prerequisite DAG |
| `data/questions.json` | 40 MCQs, 4 per skill, tagged with `a` and `b` |

The app creates `data/app.db` on first run and `models/risk_model.pkl` on first
training (or automatically on first app load if the file is missing).

---

## 2. Assumptions (read this first)

These are deliberate MVP choices, not claims about real students:

1. **Risk target.** "At risk" means *predicted probability of ending the topic with
   mean mastery below 0.50*. In the simulator this label is computed from hidden
   final mastery. In the app it is a probability, not a verdict.
2. **One simulated session = 20 questions.** The seed bank has 40 items; a real
   student can answer all 40, after which the UI offers AI-generated extras.
3. **BKT parameters are demo priors**, not calibrated on a real cohort:
   `p_init=0.45`, `p_learn=0.15`, `p_slip=0.10`, `p_guess=0.20`, with small
   per-topic overrides in `config.py`.
4. **Ability prior is N(0, 1)** and theta is clipped to [-4, 4]. MAP estimation is
   used instead of pure MLE so a new student gets a sensible estimate after one item.
5. **"Earliest prerequisite"** = the failing ancestor with the smallest DAG depth;
   ties are broken by lower mastery. If no ancestor is below 0.50, the weakest
   ancestor is shown with `found=False`.
6. **Question difficulties/discriminations are authored**, not empirically calibrated.
   `b` in [-2, 2], `a` in [0.5, 2]. Treat them as sensible seed values.
7. **The LLM is never part of scoring.** Groq only writes explanations, plans and
   summaries. If `GROQ_API_KEY` is missing, timeouts, or returns bad JSON, the app
   uses deterministic local fallbacks and remains fully usable.
8. **Single-user local MVP.** SQLite lives on the machine; names are stored in
   plain text. Do not enter real student PII in a demo.
9. **Subject scope** is the 10 first-year engineering maths skills in
   `data/skills.json`; question content is English and ASCII maths notation.

---

## 3. Setup

### 3.1 Python

Python **3.10+** (tested on 3.13).

```bash
pip install -r requirements.txt
```

Optional, but recommended so the project does not touch your global Python:

```bash
python -m venv .venv
# Windows (Git Bash / PowerShell):
.venv/Scripts/python.exe -m pip install -r requirements.txt
# macOS / Linux:
source .venv/bin/activate && pip install -r requirements.txt
```

### 3.2 Free Groq API key (optional but recommended)

1. Open <https://console.groq.com> and sign up (free, no card required).
2. Go to **API Keys** -> **Create API Key**.
3. Copy the key.
4. Copy `.env.example` to `.env` and set:

```bash
GROQ_API_KEY=gsk_...
```

`config.py` loads `.env` with `python-dotenv`. Default model:
`openai/gpt-oss-120b` (override with `GROQ_MODEL`; `GROQ_MODEL_FALLBACKS` is tried automatically if
that model is not enabled on your account). The key stays on the server
side; it is never sent to the browser. Do not commit `.env`.

If you skip this step everything works: the quiz serves seed questions, explanations
use the seeded solution notes, plans are built locally, and the class insight is
computed from the class statistics.

### 3.3 Run

```bash
python simulator.py      # trains + evaluates, writes models/risk_model.pkl
streamlit run app.py     # opens http://localhost:8501
```

The app trains the model automatically if `models/risk_model.pkl` is missing, so
`python simulator.py` is only needed to see the metrics in the terminal or to refresh
the saved model.

---

## 4. Architecture

```
project/
  app.py                 Streamlit UI (4 tabs)
  config.py              settings + .env loading
  db.py                  SQLite access (no ORM)
  irt.py                 2PL / MAP ability / information / item selection
  bkt.py                 Bayesian Knowledge Tracing per skill
  risk.py                features, classifier, root-cause finder
  simulator.py           synthetic cohort + evaluation entry point
  llm.py                 all Groq calls + fallbacks
  question_bank.py       seed bank, generation, adaptive picking
  data/
    skills.json          10 skills + prerequisite edges
    questions.json       40 seeded MCQs
    app.db               created at runtime
  models/
    risk_model.pkl       created by simulator.py or first app run
  .env.example
  requirements.txt
  README.md
```

Data flow:

```
skills.json ──> question_bank / prerequisite DAG ──> app, risk
questions.json ──> adaptive picker ──> response log (SQLite)
response log ──> IRT theta        ──> next item choice
             └─> BKT mastery     ──> mastery chart / class heatmap
             └─> 7 features      ──> RandomForest risk ──> top drivers
mastery ──> DAG walk backwards ──> root-cause skill
summary JSON ──> Groq plan / class insight (or local fallback)
```

### Skill DAG

```
Functions -> Limits -> Derivatives -> Applications of Derivatives
                    \-> Integrals
Matrices  -> Determinants -> Linear Systems -> Eigenvalues
Derivatives -> Partial Derivatives
```

10 skills, 8 edges. Levels: Functions/Matrices 0, Limits/Determinants 1,
Derivatives/Integrals/Linear Systems 2, Applications/Eigenvalues/Partial Derivatives 3.

### Database

| Table | Columns |
|---|---|
| `students` | `id`, `name` (unique), `created_at` |
| `responses` | `id`, `student_id`, `question_id`, `skill`, `correct`, `time_taken`, `hints_used`, `timestamp` |
| `mastery_snapshots` | `id`, `student_id`, `skill`, `mastery`, `timestamp` |

Every answer writes a response **and** a mastery snapshot, so mastery trajectories are
recoverable per student per skill.

---

## 5. The math used

### 5.1 2PL Item Response Theory

```
P(correct | theta, a, b) = 1 / (1 + exp(-a (theta - b)))
```

`a` = discrimination, `b` = difficulty, `theta` = ability.

**Item information:**

```
I(theta) = a^2 * P(theta) * (1 - P(theta))
```

**MAP ability with an N(0,1) prior, solved by Newton-Raphson:**

```
L(theta)   = sum_i [u_i log P_i + (1-u_i) log(1-P_i)] - (theta - mu)^2 / (2 sigma^2)
L'(theta)  = sum_i a_i (u_i - P_i) - (theta - mu) / sigma^2
L''(theta) = -sum_i a_i^2 P_i (1-P_i) - 1 / sigma^2

theta_new = theta - L'(theta) / L''(theta)
SE        = sqrt(-1 / L''(theta))
```

The next question is the unasked item of the **weakest skill that still has unseen
items**, chosen by maximum information at the current theta. If a skill is exhausted,
selection moves to the next weakest skill.

### 5.2 Bayesian Knowledge Tracing

For each skill, `P(L)` is the probability the skill is learned.

```
correct:  P(L | correct) = P(L)(1 - s) / [P(L)(1 - s) + (1 - P(L)) g]
wrong:    P(L | wrong)   = P(L) s     / [P(L) s     + (1 - P(L))(1 - g)]

P(L_next) = P(L | obs) + (1 - P(L | obs)) * p_learn
```

`s` = slip, `g` = guess, `p_learn` = transition probability. Values are clamped to
`(0, 1)` to avoid numerical dead ends.

### 5.3 Risk features (computed from the response log only)

| Feature | Definition |
|---|---|
| `recent_accuracy` | mean correctness over the last 5 answers |
| `accuracy_trend` | least-squares slope of correctness over the last 10 answers |
| `rt_z` | `(mean response time - 18) / 9`, clipped to [-4, 4] |
| `hint_rate` | mean hints per question, clipped to [0, 4] |
| `wrong_streak` | trailing run of wrong answers, capped at 10 |
| `avg_mastery` | mean BKT mastery over the skills attempted |
| `skills_below_04` | attempted skills with mastery < 0.40 |

Model: `RandomForestClassifier` (240 trees, max depth 6, balanced class weights).
The top-3 drivers are `importance * standardised feature value`, signed - a simple
local attribution, not SHAP.

### 5.4 Simulator and metrics

Each synthetic student has hidden `theta ~ N(0,1)`, a personal learning rate
(0.02-0.22), slip (0.05-0.18), guess (0.15-0.30), per-skill initial mastery driven by
theta, and one deliberately blocked prerequisite. They answer 20 questions weighted
toward weak skills; observed correctness mixes the 2PL probability with slip/guess.

Hidden label: `mean final mastery < 0.50`.

Training uses two snapshots per student (full session + a random prefix) so early
prediction is in-distribution. The train/test split is **by student** to avoid label
leakage. Reported metrics: accuracy, precision, recall, F1, ROC-AUC, plus:

* **Baseline**: flag when recent accuracy < 50%.
* **Earliness**: for each at-risk test student, the first question (from Q3) at which
  predicted risk crosses 0.50; reported as a distribution and a mean.

Reference run (`python simulator.py`, seed 7, 200 students, scikit-learn 1.9.1):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| RandomForest (7 features) | 0.92 | 0.857 | 0.947 | 0.900 | 0.980 |
| Baseline: recent accuracy < 50% | 0.82 | 0.708 | 0.895 | 0.791 | 0.907 |

Earliness: 100% of at-risk test students flagged after a mean of 3.3 of 20 questions.

These numbers describe **synthetic** students. They are not evidence of real-world
validity.

---

## 6. Five-minute demo script

**0:00 - Start**
`streamlit run app.py`. Point at the orange disclaimer banner: learning gaps, not
clinical conditions.

**0:20 - Student Quiz**
Type `Asha-01` in the sidebar -> *Use this student*. A question appears (eigenvalues
first is normal - the weakest skill with unseen items is targeted). Click *Show hint*,
answer deliberately wrong once, and show the instant seeded/AI explanation. Answer
one or two more; point out theta, the percentile and the mastery bars moving.

**1:40 - My Report**
Switch to *My Report*. Show the radar, the risk gauge with its band, the top three
drivers, and the prerequisite graph. The red ring is the root-cause prerequisite;
the blue ring is the active skill. Click *Generate personalised plan* - root-cause
skill, three exercises, worked example, estimated time.

**3:00 - Teacher Dashboard**
Click *Load synthetic demo class* (30 students). Show the students x skills heatmap,
the risk-ranked table (root causes included), then *Generate class summary with Groq*.
Mention it works without a key (local 5-line fallback).

**4:00 - Model Evaluation**
Show the model-vs-baseline table, ROC curve, confusion matrix, earliness histogram
and feature importances. Key line: "it beats the accuracy baseline while flagging
at-risk students around question 3 of 20."

**4:40 - Close**
Restate the ethics boundary: decision support only, teacher review required, synthetic
validation only.

---

## 7. Limitations and ethics

* **Not a diagnostic tool.** It does not detect, screen for, or rule out dyscalculia,
  dyslexia, ADHD or any other condition. It estimates gaps in specific maths skills.
  A qualified teacher must review every recommendation.
* **Synthetic validation only.** The classifier is trained and evaluated on simulated
  learners. Its metrics are optimistic by construction and will not transfer to a real
  cohort without recalibration and external validation.
* **No demographic or clinical inputs.** The model deliberately uses only behavioural
  learning data (accuracy, timing, hints, mastery).
* **Risk is probabilistic and provisional.** Early-session estimates (1-5 questions)
  are noisy. The UI labels the number of responses and the band, not a verdict.
* **Feedback loops.** Any early-warning system can become self-fulfilling if teachers
  treat flagged students differently. Use flags to change support, not expectations.
* **LLM risk.** Explanations/plans can be wrong or awkward. Prompts forbid clinical
  language and labelling, but outputs are not guaranteed. The fallback path is
  deterministic and uses seeded solution notes. Do not send student names or sensitive
  data to the LLM - only skill/mastery summaries are sent.
* **Privacy.** Names and response logs are stored unencrypted in local SQLite. Use
  pseudonyms. Delete `data/app.db` to wipe all data.
* **Small seed bank.** 40 authored items cannot cover a curriculum. Difficulty and
  discrimination are authored, not calibrated. The AI generator helps with volume but
  its output should be reviewed before classroom use.
* **No accessibility audit, authentication or multi-tenant isolation** in this MVP.

---

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: streamlit` | Activate the venv, or `pip install -r requirements.txt` with the same interpreter you run. |
| App says "no GROQ_API_KEY" | Copy `.env.example` to `.env`, add the key, restart Streamlit. |
| `models/risk_model.pkl` missing | The app retrains on load; or run `python simulator.py`. |
| Port 8501 busy | `streamlit run app.py --server.port 8502` (loopback only). |
| Want a clean database | Stop the app and delete `data/app.db`. |
| Want to retrain the risk model | Delete `models/risk_model.pkl` or run `python simulator.py`. |
