# Adaptive Maths Gap Lab

A Streamlit app that tries to spot gaps in first-year engineering maths early, and
then help close them.

Questions are picked with 2PL item response theory, mastery is tracked per skill with
Bayesian knowledge tracing, and a RandomForest trained on a synthetic cohort flags
students who look likely to fall behind. Explanations, study plans and class summaries
come from an LLM hosted on Groq (`openai/gpt-oss-120b` by default). That part is
optional, and every AI call has a local fallback behind it.

One boundary before anything else. The app reports gaps in specific maths skills. It
does not screen for or diagnose dyscalculia, dyslexia, ADHD or any other condition, and
it should not be used to label a student. A teacher should review every recommendation
before acting on it.

---

## 1. What's in the repo

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

The app creates `data/app.db` on first run. `models/risk_model.pkl` is written the first
time you run `simulator.py`, or on the first app load if the file is missing.

---

## 2. Assumptions

Worth knowing before you read any of the numbers below.

1. **Risk target.** "At risk" means the student is predicted to finish the topic with
   mean mastery below 0.50. In the simulator that label comes from hidden final mastery.
   In the app the same thing is shown as a probability, not a verdict.
2. **One simulated session is 20 questions.** The seed bank has 40 items, so a real
   student can answer all of them; after that the UI offers AI-generated extras.
3. **The BKT parameters are demo priors**, not values fitted to a real cohort:
   `p_init=0.45`, `p_learn=0.15`, `p_slip=0.10`, `p_guess=0.20`, with small per-topic
   overrides in `config.py`.
4. **Ability is assumed N(0, 1)** and theta is clipped to [-4, 4]. Estimation uses MAP
   rather than plain MLE so a new student gets a sensible number after one item.
5. **"Earliest prerequisite"** is the failing ancestor with the smallest DAG depth, and
   ties go to the lower mastery. If no ancestor is below 0.50, the weakest one is shown
   with `found=False`.
6. **Difficulties and discriminations were written by hand**, not calibrated against
   responses. `b` sits in [-2, 2] and `a` in [0.5, 2]; treat both as seed values.
7. **The LLM never touches scoring.** Groq only writes explanations, plans and
   summaries. With no `GROQ_API_KEY`, on a timeout, or on bad JSON, the app falls back to
   deterministic local text and stays usable.
8. **This is a single-user local MVP.** SQLite lives on your machine and names are
   stored in plain text, so don't put real student data into a demo.
9. **Scope** is the 10 first-year engineering maths skills in `data/skills.json`.
   Question text is in English with ASCII maths notation.

---

## 3. Setup

### 3.1 Python

Python 3.10 or newer (tested on 3.13).

```bash
pip install -r requirements.txt
```

A virtual environment is optional, but it keeps the project off your global Python:

```bash
python -m venv .venv
# Windows (Git Bash / PowerShell):
.venv/Scripts/python.exe -m pip install -r requirements.txt
# macOS / Linux:
source .venv/bin/activate && pip install -r requirements.txt
```

### 3.2 A free Groq API key

Only needed if you want generated explanations. Everything works without it.

1. Sign up at <https://console.groq.com>. It's free and no card is required.
2. Open **API Keys** and create a key.
3. Copy `.env.example` to `.env` and set `GROQ_API_KEY=gsk_...`.

`config.py` loads `.env` through `python-dotenv`. The default model is
`openai/gpt-oss-120b`; override it with `GROQ_MODEL`, and `GROQ_MODEL_FALLBACKS` is
tried automatically if that model isn't enabled on your account. The key stays on the
server side and is never sent to the browser. Keep `.env` out of git.

If you skip this step, the quiz serves the seed questions, explanations use the seeded
solution notes, plans are built locally, and the class insight is computed from the
class statistics.

### 3.3 Run

```bash
python simulator.py      # trains + evaluates, writes models/risk_model.pkl
streamlit run app.py     # opens http://localhost:8501
```

The app retrains on load if `models/risk_model.pkl` is missing, so `python simulator.py`
is only needed when you want the metrics in the terminal or a refreshed saved model.

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

Each answer writes a response and a mastery snapshot, which is what makes mastery
trajectories recoverable per student and per skill.

---

## 5. The maths

### 5.1 2PL item response theory

```
P(correct | theta, a, b) = 1 / (1 + exp(-a (theta - b)))
```

`a` is discrimination, `b` is difficulty and `theta` is ability.

Item information:

```
I(theta) = a^2 * P(theta) * (1 - P(theta))
```

MAP ability with an N(0,1) prior, solved by Newton-Raphson:

```
L(theta)   = sum_i [u_i log P_i + (1-u_i) log(1-P_i)] - (theta - mu)^2 / (2 sigma^2)
L'(theta)  = sum_i a_i (u_i - P_i) - (theta - mu) / sigma^2
L''(theta) = -sum_i a_i^2 P_i (1-P_i) - 1 / sigma^2

theta_new = theta - L'(theta) / L''(theta)
SE        = sqrt(-1 / L''(theta))
```

The next question is the unasked item of the weakest skill that still has unseen items,
chosen by maximum information at the current theta. Once a skill runs out of items,
selection moves to the next weakest one.

### 5.2 Bayesian knowledge tracing

For each skill, `P(L)` is the probability that the skill is learned.

```
correct:  P(L | correct) = P(L)(1 - s) / [P(L)(1 - s) + (1 - P(L)) g]
wrong:    P(L | wrong)   = P(L) s     / [P(L) s     + (1 - P(L))(1 - g)]

P(L_next) = P(L | obs) + (1 - P(L | obs)) * p_learn
```

`s` is slip, `g` is guess and `p_learn` is the transition probability. Values are
clamped to (0, 1) so the updates can't dead-end numerically.

### 5.3 Risk features (response log only)

| Feature | Definition |
|---|---|
| `recent_accuracy` | mean correctness over the last 5 answers |
| `accuracy_trend` | least-squares slope of correctness over the last 10 answers |
| `rt_z` | `(mean response time - 18) / 9`, clipped to [-4, 4] |
| `hint_rate` | mean hints per question, clipped to [0, 4] |
| `wrong_streak` | trailing run of wrong answers, capped at 10 |
| `avg_mastery` | mean BKT mastery over the skills attempted |
| `skills_below_04` | attempted skills with mastery < 0.40 |

The model is a `RandomForestClassifier` with 240 trees, max depth 6 and balanced class
weights. The top three drivers are `importance * standardised feature value`, signed.
That is a simple local attribution, not SHAP.

### 5.4 Simulator and metrics

Each synthetic student gets a hidden `theta ~ N(0,1)`, a personal learning rate
(0.02-0.22), slip (0.05-0.18), guess (0.15-0.30), per-skill initial mastery driven by
theta, and one deliberately blocked prerequisite. They answer 20 questions weighted
toward weak skills, and the observed correctness mixes the 2PL probability with
slip and guess.

The hidden label is `mean final mastery < 0.50`.

Training uses two snapshots per student (full session, plus a random prefix) so that
early prediction is in-distribution. The train/test split is by student, which keeps
labels from leaking. Reported metrics are accuracy, precision, recall, F1 and ROC-AUC,
plus two extras:

* **Baseline**: flag when recent accuracy is under 50%.
* **Earliness**: for each at-risk test student, the first question (from Q3) where
  predicted risk crosses 0.50, reported as a distribution and a mean.

Reference run (`python simulator.py`, seed 7, 200 students, scikit-learn 1.9.1):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| RandomForest (7 features) | 0.92 | 0.857 | 0.947 | 0.900 | 0.980 |
| Baseline: recent accuracy < 50% | 0.82 | 0.708 | 0.895 | 0.791 | 0.907 |

Earliness: 100% of at-risk test students were flagged after a mean of 3.3 of 20
questions.

All of this describes synthetic students. It is not evidence that the model works on a
real cohort.

---

## 6. Demo walkthrough

Roughly five minutes, if you want a script for showing this to someone.

**Start.** Run `streamlit run app.py` and point at the orange disclaimer banner: learning
gaps, not clinical conditions.

**Student quiz.** Type `Asha-01` in the sidebar and click *Use this student*. A question
appears (eigenvalues first is normal, since the weakest skill with unseen items is
targeted). Click *Show hint*, answer one deliberately wrong, and show the explanation
that comes back. Answer one or two more and point out theta, the percentile and the
mastery bars moving.

**My report.** Switch to *My Report*. The radar, the risk gauge with its band, the top
three drivers and the prerequisite graph are all there. The red ring is the root-cause
prerequisite and the blue ring is the active skill. Click *Generate personalised plan*
to get the root-cause skill, three exercises, a worked example and an estimated time.

**Teacher dashboard.** Click *Load synthetic demo class* (30 students). The students x
skills heatmap and the risk-ranked table (with root causes) come up; then try *Generate
class summary with Groq*. It also works without a key, using the five-line local
fallback.

**Model evaluation.** The model-vs-baseline table, ROC curve, confusion matrix,
earliness histogram and feature importances are on this tab. The line worth saying out
loud: it beats the accuracy baseline while flagging at-risk students around question 3
of 20.

**Close.** Repeat the ethics boundary. Decision support only, teacher review required,
synthetic validation only.

---

## 7. Limitations and ethics

* **Not a diagnostic tool.** It does not detect, screen for or rule out dyscalculia,
  dyslexia, ADHD or any other condition. It estimates gaps in specific maths skills, and
  a qualified teacher should review every recommendation.
* **Synthetic validation only.** The classifier is trained and evaluated on simulated
  learners, so its metrics are optimistic by construction. They will not transfer to a
  real cohort without recalibration and external validation.
* **No demographic or clinical inputs.** The model uses behavioural learning data only:
  accuracy, timing, hints and mastery.
* **Risk is probabilistic and provisional.** Estimates from the first few questions are
  noisy. The UI shows the number of responses and the band rather than a verdict.
* **Feedback loops.** Any early-warning system can become self-fulfilling if teachers
  treat flagged students differently. Flags should change the support a student gets,
  not the expectations placed on them.
* **LLM risk.** Explanations and plans can be wrong or awkwardly worded. The prompts
  forbid clinical language and labelling, but outputs aren't guaranteed. The fallback
  path is deterministic and uses the seeded solution notes. Only skill and mastery
  summaries go to the LLM; student names and sensitive data should not be sent.
* **Privacy.** Names and response logs sit unencrypted in local SQLite. Use pseudonyms.
  Delete `data/app.db` to wipe everything.
* **Small seed bank.** 40 authored items can't cover a curriculum, and the difficulty and
  discrimination values are authored rather than calibrated. The AI generator helps with
  volume, but review its output before classroom use.
* **No accessibility audit, authentication or multi-tenant isolation** in this MVP.

---

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: streamlit` | Activate the venv, or `pip install -r requirements.txt` with the interpreter you actually run. |
| App says there is no `GROQ_API_KEY` | Copy `.env.example` to `.env`, add the key, restart Streamlit. |
| `models/risk_model.pkl` missing | The app retrains on load, or run `python simulator.py`. |
| Port 8501 is busy | `streamlit run app.py --server.port 8502` (loopback only). |
| You want a clean database | Stop the app and delete `data/app.db`. |
| You want to retrain the risk model | Delete `models/risk_model.pkl` or run `python simulator.py`. |
