# Blabby

AI IELTS Speaking Practice — and, since Sprint Reading-1, IELTS Academic Reading.

## Reading Module (v1)

IELTS Academic Reading practice. One AI-generated passage (700–900 words) plus 9 questions (3 MCQ, 3 True/False/Not Given, 3 Heading Matching) per attempt.

- Free: 1 attempt / day (UTC calendar day)
- Pro: unlimited
- Band stored separately as `profiles.user_band_reading` — does not touch the existing Speaking `user_band`

### Endpoints

| Method | Path | Notes |
|---|---|---|
| `POST` | `/reading/passage/generate` | Generate + persist passage and 9 questions. Pre-emptively checks quota. |
| `POST` | `/reading/attempt/start` | Open an `in_progress` attempt row. Authoritative quota gate. Idempotent: returns existing in_progress attempt for the same passage. |
| `POST` | `/reading/attempt/submit` | Score answers, update `user_band_reading`, reveal correct answers + explanations + evidence quotes. |
| `POST` | `/reading/attempt/abandon` | Mark an in_progress attempt as `abandoned`. Abandoned attempts do not count toward quota. |
| `GET`  | `/reading/attempt/{id}` | Re-fetch a submitted attempt's full reveal. |
| `GET`  | `/reading/history` | Caller's most recent submitted attempts. |
| `GET`  | `/reading/quota` | `{used_today, limit, remaining, is_pro, should_upgrade}`. For Pro users `limit` and `remaining` are `null`. |
| `POST` | `/vocab/lookup` | Single-word definition (≤25 words) via Claude, in-process LRU cache (500 entries). |
| `POST` | `/api/vocabulary/save_word` | Free-form word save. Lazily creates a sparse `vocabulary_items` row if the word isn't in the catalog. Tags `user_vocabulary.source='reading'`. |

### Frontend

`frontend/app/reading.html` — single-file page (inline CSS + JS), three view states: Landing → Examination → Reckoning. Click any word in the passage for a definition popover + Save-to-lexicon action.

### Tests

Backend tests live in `backend/tests/`. The suite runs on a clean checkout with
**no `.env` and no provider credentials at all** — `tests/conftest.py` blanks the
Supabase vars and `setdefault`s dummy `GROQ_API_KEY` / `ANTHROPIC_API_KEY`, which
is required because `main.py` constructs both provider clients at import time.

```sh
cd blabby/backend
python3 -m venv venv                                              # one-time
./venv/bin/pip install -r requirements.txt -r requirements-dev.txt  # one-time
./venv/bin/pytest tests -q
```

Expected on a clean environment: **99 passed, 10 skipped, 0 failed**.

The 10 skips are credential-gated integration tests and are expected to stay
skipped locally and in CI. To run them, set the following against a **staging**
environment — never production:

| Test module | Required env vars |
|---|---|
| `test_reading_e2e.py` (8 tests) | `READING_E2E_BASE_URL`, `READING_E2E_USER_TOKEN` |
| `test_reading_e2e.py` (1 Pro test) | additionally `READING_E2E_PRO_USER_TOKEN` |
| `test_supabase_p1_permissions.py` (1 test) | `SUPABASE_SECURITY_TEST_URL`, `SUPABASE_SECURITY_TEST_ANON_KEY`, `SUPABASE_SECURITY_TEST_SERVICE_KEY`, `SUPABASE_SECURITY_TEST_USER_A_TOKEN`, `SUPABASE_SECURITY_TEST_USER_B_TOKEN` |

### CI

`.github/workflows/test.yml` runs the command above on every push and pull
request. It holds no secrets, so the 10 credential-gated tests stay skipped
there by design.

**Python version:** CI pins **3.11** (`python-version: '3.11'` — minor pinned,
patch floats so the runner picks up interpreter security fixes). The local
`backend/venv` is 3.10.0 and the suite passes on both; the Render production
runtime is not declared anywhere in this repo, so 3.11 is a deliberate choice
rather than a mirror of production. If Render's runtime is ever pinned
explicitly, align this value to it.
