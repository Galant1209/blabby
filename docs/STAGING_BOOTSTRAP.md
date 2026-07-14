# Blabby staging bootstrap

This checklist creates an isolated validation environment. Do not substitute
production credentials, projects, databases, domains, or user accounts.

## 1. Supabase staging project

- Create a dedicated Supabase project for staging.
- Record its project ref as `EXPECTED_SUPABASE_PROJECT_REF` and confirm that
  `SUPABASE_URL` resolves to `<project-ref>.supabase.co`.
- Keep staging data disposable and separate from production backups.
- Before migration, record row counts, RLS status, policies, grants, relevant
  functions/views, orphan ownership records, and long-running transactions.

## 2. Test users

- Create two confirmed, disposable users: User A and User B.
- Generate independent access tokens for the permission matrix.
- Seed only the minimum owner-scoped rows needed to test cross-user denial.

## 3. Render staging service

- Create a separate backend service pointing only to the staging Supabase project.
- Set `APP_ENV=staging` and the staging project ref in
  `EXPECTED_SUPABASE_PROJECT_REF`.
- Disable production domains and production secrets.
- Record the full start command, Uvicorn/Gunicorn worker count,
  `WEB_CONCURRENCY`, instance count, autoscaling min/max, and instance memory.

## 4. Vercel staging project

- Create or identify a separate frontend project with a staging-only domain.
- Configure it to call the Render staging backend and staging Supabase project.
- Do not alias the staging deployment to a production domain.

## 5. Secrets storage

- Store backend staging secrets in the Render staging service secret store.
- Store frontend staging configuration in the Vercel staging project.
- Store permission-test credentials in the staging CI secret store or an
  ephemeral local environment that is excluded from Git.
- Copy `.env.staging.example` to an ignored local file; never fill secrets into
  the tracked example.

## 6. Credential isolation

- Never use a production service-role key, API key, JWT, project ref, or user.
- Before any connection, compare `EXPECTED_SUPABASE_PROJECT_REF` with the host
  in `SUPABASE_URL`. A mismatch is a hard stop.
- Do not print secret values in logs, screenshots, test output, or tickets.

## 7. Migration baseline and permission matrix

Run these commands only after all variables are confirmed as staging-only:

```sh
# Capture the SQL/catalog baseline with the approved staging database tooling.
# Save row counts, RLS flags, policies, grants, functions/views, orphans, and
# long-running transactions before applying any migration.

psql "$STAGING_DATABASE_URL" \
  --set ON_ERROR_STOP=1 \
  --file supabase/migrations/20260714_p1_rls_and_reading_answers.sql

# Run a second time to prove idempotency, then compare the baseline row counts.
psql "$STAGING_DATABASE_URL" \
  --set ON_ERROR_STOP=1 \
  --file supabase/migrations/20260714_p1_rls_and_reading_answers.sql

perl -e 'alarm 60; exec @ARGV' \
  backend/venv/bin/pytest backend/tests/test_supabase_p1_permissions.py -q
```

The permission test must run without skips using all five
`SUPABASE_SECURITY_TEST_*` variables.

## 8. Render topology record

| Field | Staging value | Evidence/date |
|---|---|---|
| Full start command | | |
| Gunicorn/Uvicorn workers | | |
| `WEB_CONCURRENCY` | | |
| Instance count | | |
| Autoscaling enabled | | |
| Autoscaling min/max | | |
| Instance memory | | |

Process-local Part 2 concurrency is temporarily acceptable only for one worker,
one instance, and autoscaling off. Otherwise a distributed lease is required.

## 9. Browser and audio validation matrix

| Check | Browser/input | Expected result | Actual/evidence |
|---|---|---|---|
| Health | Direct staging backend | 200 | |
| Reading complete flow | Chrome and Safari | Complete without answer pre-exposure | |
| Part 2 WebM | Chrome recording | Accepted and evaluated | |
| Part 2 MP4/M4A | Safari recording | Accepted and evaluated | |
| Oversized multipart | Over request limit | Safe JSON 413 before endpoint | |
| Malformed/truncated audio | Invalid WebM/MP4 | Consistent 4xx, no provider call | |
| Writing SVG payload | Security corpus | No executable DOM or external request | |
| Part 2 transcript payload | Security corpus | Rendered as text | |
| Parallel uploads | 5 and 10 near-15 MiB requests | Bounded memory and concurrency | |
| Disconnect/timeout | Interrupted upload/provider | Temporary files cleaned | |
| Network inspection | All payload cases | No malicious external request | |

## 10. GO / NO-GO criteria

GO requires all of the following:

- Render topology is evidenced; distributed lease is present if topology is
  not single-worker/single-instance with autoscaling off.
- Migration succeeds twice, row counts remain unchanged, and catalog state
  matches the intended grants and policies.
- The anon/User A/User B/service-role matrix runs without skips and passes.
- Every browser/audio validation row passes with retained evidence.
- No production credential or endpoint appears in staging configuration.
- No unresolved P1 security blocker remains.

Any missing evidence, skipped permission test, credential ambiguity, data
change, cross-user access, answer exposure, executable payload, provider call
after rejection, or unbounded memory/concurrency observation is NO-GO.
