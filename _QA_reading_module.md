# Reading Module — Manual QA

## E2E test users

Two dedicated test users must exist in Supabase auth. They must be separate
accounts — sharing them flaps the quota-blocked test.

- `e2e-free@blabby.test` — free tier, **never** granted Pro. Token goes in
  `READING_E2E_USER_TOKEN`.
- `e2e-pro@blabby.test` — permanently granted Pro via the admin tool
  (`profiles.is_pro_grant = true`). Token goes in
  `READING_E2E_PRO_USER_TOKEN`. The e2e suite NEVER mutates this user's
  pro status — provisioning is manual and one-time.

Both should be reset (today's attempts cleared) before each full e2e run.
Run this SQL in the Supabase SQL editor:

```sql
-- Reset the FREE e2e user's today-of-UTC attempts. Replace the UUID
-- placeholder with the actual id from auth.users.
delete from public.reading_attempts
where user_id = '<E2E_FREE_USER_UUID>'
  and started_at >= date_trunc('day', now() at time zone 'utc');

-- The PRO user has no quota so does not need resetting, but if you want
-- a clean history view do the same for them:
-- delete from public.reading_attempts
-- where user_id = '<E2E_PRO_USER_UUID>'
--   and started_at >= date_trunc('day', now() at time zone 'utc');
```

Optional helper script: drop a `scripts/reset_e2e_attempts.sh` that wraps
the SQL above via `psql` for one-command resets. Not part of v1 — flagged.



## Brand language audit
- [ ] Every visible string passes the "parchment test" — would it be acceptable on a 19th-century examination notice?
- [ ] No exclamation marks except where the register genuinely warrants
- [ ] No emoji (✓ ✗ are typographic marks, acceptable)
- [ ] British spelling throughout (organise, behaviour, judgement, etc.)

## Quota
- [ ] Free user, 0 attempts today: "Summon" works
- [ ] Free user, 1 submitted attempt today: "Summon" returns paywall (redirects to `/upgrade.html?source=reading_quota`)
- [ ] Free user, 1 abandoned attempt today (via the Abandon button → `POST /reading/attempt/abandon`): "Summon" still works
- [ ] Free user, 1 in_progress attempt that was left via navigation (no abandon call): still counts toward quota
- [ ] Pro user: unlimited; allowance line is hidden; never redirected to `/upgrade`
- [ ] `GET /reading/quota` returns `{used_today, limit, remaining, is_pro, should_upgrade}`; `limit` and `remaining` are `null` for Pro users

## Generation quality (run 5 passages, eyeball each)
- [ ] Passage word count within 600–1000
- [ ] No US spelling
- [ ] All 9 evidence_quotes are literal substrings of the passage body
- [ ] At least 2 of the 3 TFNG questions cover distinct True/False/Not Given values
- [ ] Headings list contains 5 entries, 3 correct + 2 distractors
- [ ] Heading-matching questions all share the same 5-heading bank
- [ ] MCQ distractors are plausible (not laughably wrong)

## Critical paths
- [ ] Submit with all answers → results page renders with score, band, per-question reveal
- [ ] Submit with some unanswered → "Submit for Assessment" stays disabled until all 9 answered
- [ ] Network drop during submit → answers preserved in `localStorage` (`reading_answers_${attempt_id}`), retry banner appears with working "Try again"
- [ ] Vocab save from passage → entry appears in `/vocabulary.html` lexicon page; word in passage gets `.is-saved` highlight
- [ ] Vocab popover shows real definition (from `/vocab/lookup`) within ~2 s, falls back to "Definition unavailable." on failure
- [ ] "Review the Passage" highlights evidence quotes in the passage view above the reckoning
- [ ] "Return to the Hall" navigates back to `/index.html`
- [ ] Closing line on reckoning is forward-looking ("Tomorrow's passage will be set when the sun next rises.")

## Mobile
- [ ] Two-column layout collapses cleanly under 900 px
- [ ] Word-tap popovers don't get clipped at viewport edges (clamped left/right)
- [ ] Sticky exam footer stays accessible above iOS Safari's bottom bar

## Tracking (open browser console; check `window.analytics.track` fires)
- [ ] `reading_page_viewed`
- [ ] `reading_summon_clicked`
- [ ] `reading_passage_rendered`
- [ ] `reading_attempt_started`
- [ ] `reading_word_clicked` (on first word tap)
- [ ] `reading_word_saved` (on successful save)
- [ ] `reading_question_answered` (on each radio/dropdown change)
- [ ] `reading_attempt_submitted`
- [ ] `reading_paywall_hit` (when 403 fires on a 2nd attempt)
- [ ] `reading_review_passage_clicked`
- [ ] `reading_attempt_abandoned`
- [ ] `reading_hub_clicked` (from `index.html` skill-entry)

## Server-side log spot-checks
- [ ] `[READING_PASSAGE_GENERATED]` log line on every successful generate
- [ ] `[READING_ATTEMPT_STARTED]`, `[READING_ATTEMPT_SUBMITTED]`, `[READING_ATTEMPT_ABANDONED]` on respective actions
- [ ] `[READING_QUOTA_BLOCKED]` when paywall triggers
- [ ] `[VOCAB_SAVE_WORD]` / `[VOCAB_SAVE_WORD_DUP]` on save_word calls

## Data integrity
- [ ] On submit, `reading_attempts.score`, `total`, `band_estimate`, `submitted_at`, `status='submitted'` are populated
- [ ] `reading_answers` has 9 rows linked to the attempt
- [ ] `profiles.user_band_reading` is updated using the 80/20 weighted MA (mirror of Speaking)
- [ ] Sparse `vocabulary_items` rows created via `/save_word` have `word` populated and `zh_meaning=''`; surface in `vocabulary.html` shows them as placeholder cards

## Failure modes to verify
- [ ] LLM returns malformed JSON for passage → retry twice → eventually 500 with `passage_generation_failed`
- [ ] LLM returns questions with fabricated evidence → retry twice → 500 with `questions_generation_failed`; orphan passage rolled back
- [ ] User submits an `attempt_id` that isn't theirs → 404
- [ ] User submits an already-submitted attempt → 409
- [ ] Abandon called on an already-submitted attempt → 409
