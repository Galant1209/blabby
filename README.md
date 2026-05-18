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

Backend tests live in `backend/tests/`. Pure-validator tests run without any credentials; e2e integration tests skip unless `READING_E2E_BASE_URL` and `READING_E2E_USER_TOKEN` are set (see `tests/test_reading_e2e.py` for full env-var doc).

```sh
cd blabby/backend
./venv/bin/pip install -r requirements-dev.txt   # one-time
./venv/bin/pytest tests -v
```
