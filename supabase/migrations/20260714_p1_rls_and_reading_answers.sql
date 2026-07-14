-- P1 containment: user-owned RLS and Reading answer secrecy.
-- Forward-only, idempotent, and data-preserving. Do not apply to production
-- until the staging permission matrix passes and a backup/rollback plan exists.

begin;

-- -------------------------------------------------------------------------
-- subscriptions: clients may read only their own non-payment entitlement
-- summary. All writes remain backend/service-role only.
-- -------------------------------------------------------------------------
alter table public.subscriptions enable row level security;

drop policy if exists subscriptions_select_own on public.subscriptions;
create policy subscriptions_select_own
    on public.subscriptions
    for select
    to authenticated
    using (auth.uid() = user_id);

revoke all on table public.subscriptions from anon, authenticated;
grant select (id, plan, status, started_at, expires_at)
    on table public.subscriptions to authenticated;

-- -------------------------------------------------------------------------
-- diagnosis_cache: owner-scoped cache. The composite behavior is anchored by
-- the existing user_id primary key, so one user cannot collide with another.
-- -------------------------------------------------------------------------
alter table public.diagnosis_cache enable row level security;

drop policy if exists diagnosis_cache_select_own on public.diagnosis_cache;
drop policy if exists diagnosis_cache_insert_own on public.diagnosis_cache;
drop policy if exists diagnosis_cache_update_own on public.diagnosis_cache;
drop policy if exists diagnosis_cache_delete_own on public.diagnosis_cache;

create policy diagnosis_cache_select_own
    on public.diagnosis_cache for select to authenticated
    using (auth.uid() = user_id);
create policy diagnosis_cache_insert_own
    on public.diagnosis_cache for insert to authenticated
    with check (auth.uid() = user_id);
create policy diagnosis_cache_update_own
    on public.diagnosis_cache for update to authenticated
    using (auth.uid() = user_id)
    with check (auth.uid() = user_id);
create policy diagnosis_cache_delete_own
    on public.diagnosis_cache for delete to authenticated
    using (auth.uid() = user_id);

revoke all on table public.diagnosis_cache from anon, authenticated;
grant select, insert, update, delete
    on table public.diagnosis_cache to authenticated;

-- -------------------------------------------------------------------------
-- Reading questions: the base table contains answer material and is backend
-- only. Authenticated clients may select only prompt/options columns. PostgreSQL
-- column privileges make SELECT * and any correct_answer/explanation/evidence
-- request fail even though the row policy permits shared question prompts.
-- -------------------------------------------------------------------------
alter table public.reading_questions enable row level security;

drop policy if exists reading_questions_prompt_select on public.reading_questions;
create policy reading_questions_prompt_select
    on public.reading_questions
    for select
    to authenticated
    using (true);

revoke all on table public.reading_questions from anon, authenticated;
grant select (
    id, passage_id, question_type, question_text, options, order_idx, created_at
) on table public.reading_questions to authenticated;

commit;
