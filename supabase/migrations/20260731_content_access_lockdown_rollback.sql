-- ############################################################################
-- ##  WARNING — THIS ROLLBACK RE-OPENS THE CONTENT TABLES TO THE PUBLIC     ##
-- ##                                                                        ##
-- ##  Running this file makes questions (71 rows), reading_passages (32)    ##
-- ##  and writing_questions (95) readable by the anon role again.           ##
-- ##                                                                        ##
-- ##  The anon key is published in frontend/app/config.js:3 and this repo   ##
-- ##  is public. "anon" therefore means anyone with a browser: the entire   ##
-- ##  Part 1 question bank, every Reading passage and every Writing prompt  ##
-- ##  become downloadable by a stranger, without an account.                ##
-- ##                                                                        ##
-- ##  Nothing in the product needs this. Verified 2026-07-30: the frontend  ##
-- ##  issues zero queries against these tables, and all 153 backend call    ##
-- ##  sites use the service_role client, which is unaffected by the         ##
-- ##  lockdown. If something broke after the lockdown, it broke because it  ##
-- ##  was bypassing FastAPI — fix that caller instead of running this.      ##
-- ##                                                                        ##
-- ##  Run this ONLY to restore the pre-2026-07-31 state deliberately.       ##
-- ############################################################################
--
-- ROLLBACK for 20260731_content_access_lockdown.sql
--
-- Sections mirror the forward file and are INDEPENDENT — run only what you
-- need, in reverse order (§2 → §1) to revert everything.
--
-- No gate. Unlike the entitlement migrations, this restores a known previous
-- state and cannot revoke anything anyone paid for. The danger here is not
-- subtle breakage, it is that the file does exactly what it says — hence the
-- banner instead of a preflight.
--
-- reading_questions is absent from this file for the same reason it is absent
-- from the forward file: the forward migration never touched it. Do not add it
-- here. Its answer columns are protected by the column-level GRANT installed by
-- 20260714_p1_rls_and_reading_answers.sql:68-71, and a "symmetry" edit that
-- revoked that grant would delete a working defence — see the forward file's
-- header for the 2026-07-30 audit misreading that nearly did exactly that.


-- ── §2 rollback — restore the SELECT grants ─────────────────────────────────
-- Only SELECT is restored, deliberately. The forward file revoked ALL, but
-- anon/authenticated never legitimately held INSERT/UPDATE/DELETE on these
-- tables; handing write access back would exceed "undo".

GRANT SELECT ON TABLE public.questions          TO anon, authenticated;
GRANT SELECT ON TABLE public.reading_passages   TO anon, authenticated;
GRANT SELECT ON TABLE public.writing_questions  TO anon, authenticated;


-- ── §1 rollback — disable row level security ────────────────────────────────
-- With RLS off, the restored grants above take effect immediately and
-- unconditionally. This is the statement that actually re-opens the data.

ALTER TABLE public.questions          DISABLE ROW LEVEL SECURITY;
ALTER TABLE public.reading_passages   DISABLE ROW LEVEL SECURITY;
ALTER TABLE public.writing_questions  DISABLE ROW LEVEL SECURITY;


-- Restore the pre-lockdown comments (the forward file replaced them with the
-- zero-policy rationale, which would now be untrue).
COMMENT ON TABLE public.questions         IS NULL;
COMMENT ON TABLE public.reading_passages  IS NULL;
COMMENT ON TABLE public.writing_questions IS NULL;


-- ── verification after rollback ─────────────────────────────────────────────
-- 1. BEGIN; SET LOCAL ROLE anon;
--    SELECT count(*) FROM public.questions;   → 71 (re-exposed, as intended)
--    ROLLBACK;
--
-- 2. reading_questions must STILL be protected — the rollback must not have
--    touched it:
--    BEGIN; SET LOCAL ROLE authenticated;
--    SELECT count(*) FROM reading_questions WHERE correct_answer IS NOT NULL;
--      → ERROR 42501
--    ROLLBACK;
