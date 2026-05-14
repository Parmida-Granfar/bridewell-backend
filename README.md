# Bridewell AI Metrics Dashboard

A Django-based backend for classroom analytics and cognitive-load monitoring. This project computes student metrics from chat logs, supports passport-aware accommodations, and exposes teacher-facing summary routes.

## Features

- `cognitive-load`: per-student cognitive load scoring using NLP complexity, confusion cues, and response delay.
- `topic-wrestling`: extracts topics students are struggling with through confusion and explanation request patterns.
- `behavior-mix`: class-level breakdown of student behaviour types.
- `engagement-timeline`: bucketed engagement scores over time.
- `class-summary`: aggregate metrics for a class with passport summary insights.
- Passport-aware scoring using `StudentPassport` records.
- Supports importing student logs and passport documents via Django management commands.

## Architecture

- `bridewell_project/` — Django project settings and URL routing.
- `bridewell_api/` — main application implementing models, serializers, views, NLP logic, and imports.
- `bridewell_api/models.py` — `ChatMessage` and `StudentPassport` database models.
- `bridewell_api/nlp_utils.py` — NLP metric and scoring functions.
- `bridewell_api/views.py` — DRF API views exposing metric routes.
- `bridewell_api/serializers.py` — DRF serializers validating API responses.
- `bridewell_api/management/commands/` — import utilities and demo population scripts.

## Installation

1. Clone the repository or place code in a fresh folder.
2. Create a Python virtual environment and activate it:

```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies (adjust as needed):

```bash
pip install django djangorestframework spacy
python -m spacy download en_core_web_sm
```

4. Apply migrations:

```bash
./venv/bin/python manage.py migrate
```

5. Run the server:

```bash
./venv/bin/python manage.py runserver
```

## API Routes

All metric routes are mounted under `/api/v1/metrics/`.

### `GET /api/v1/metrics/cognitive-load/`

Returns a list of student cognitive load scores.

Query params:
- `live` (optional, default `true`) — use `false` to compute over all stored messages instead of the last 60 minutes.

Response example:

```json
[
  {"student_id": "TOM", "score": 0.82},
  {"student_id": "PRIYA", "score": 0.64}
]
```

### `GET /api/v1/metrics/topic-wrestling/`

Returns the top topics students are struggling with.

Query params:
- `live` (optional, default `true`)
- `top_n` (optional, default `10`)

Response example:

```json
[
  {"topic": "fractions", "count": 12},
  {"topic": "division", "count": 8}
]
```

### `GET /api/v1/metrics/class-summary/`

Returns a teacher-facing overview of the class, including:
- cognitive load summary
- topic wrestling
- behavior mix
- engagement overview
- passport summary

Query params:
- `live` (optional, default `true`)

### `GET /api/v1/metrics/engagement-timeline/`

Returns engagement time series data.

Query params:
- `window_minutes` (optional, default `30`)
- `bucket_minutes` (optional, default `2`)
- `live` (optional, default `true`)

Response example:

```json
[
  {"time": "14:00", "score": 0.85},
  {"time": "14:02", "score": 0.72}
]
```

### `GET /api/v1/metrics/behavior-mix/`

Returns counts for student behaviour categories.

Response example:

```json
{
  "deep_questions": 4,
  "scaffolded": 7,
  "off_topic": 1,
  "answer_seeking": 2
}
```

### `GET /api/v1/metrics/chat-summary/<student_id>/`

Returns a summary and signals for an individual student chat.

### `GET /api/v1/metrics/pair-ups/`

Returns pair-up suggestions for student collaboration.

### `GET /api/v1/metrics/learning-preferences/<student_id>/`

Returns a student’s learning preferences summary.

## Metrics Behavior

### Cognitive load

Computed from:
- average sentence length
- subordinate clause ratio
- confusion/explanation signal frequency
- assistant-to-student response delay
- passport accommodations adjustments

The score is normalized to `0.0–1.0`.

### Topic wrestling

Identifies student struggle text using:
- confusion patterns like `confused`, `don't understand`, `help`
- explanation requests like `explain`, `walk me through`

Then extracts noun-phrase topics using spaCy, with an optional scikit-learn TF-IDF fallback if no topics are found.

### Behavior mix

Classifies student messages into:
- deep questions
- scaffolded help requests
- off-topic comments
- answer-seeking queries

### Engagement timeline

Buckets student messages over time and computes a score from:
- message activity density
- proportion of student words to total words

## Data Ingestion

### Import student logs

```bash
./venv/bin/python manage.py import_studentlogs path/to/studentlogs.json
```

### Import passports

```bash
./venv/bin/python manage.py import_passport path/to/passport.docx
```

### Populate demo chat messages

```bash
./venv/bin/python manage.py populate_chats
```

## Notes

- The app uses `db.sqlite3` by default for data storage.
- `live=true` means only the last 60 minutes of chat data is used.
- Use `live=false` to compute metrics from all available messages.
- Do not commit `db.sqlite3`, `venv/`, or imported data directories.

## GitHub Push Guidance

Make sure `.gitignore` excludes:
- `venv/`
- `db.sqlite3`
- `studentlogs/`
- `passport/`
- `mnt/`
- `__pycache__/`
- `.DS_Store`

Then commit only the application code and push to your repository.
