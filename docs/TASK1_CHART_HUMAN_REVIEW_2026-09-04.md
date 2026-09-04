# Task 1 Chart Human Review — 2026-09-04

## Scope and safety boundary

This round adds the smallest admin human-review loop for `writing_questions`
Task 1 chart rows. Review metadata is separate from `is_pregenerated`: saving
`approved` means **Approved for engineering reactivation**, not serving approval,
and no review action reactivates a question. No production migration was
applied, no production data was written, and no writing submission data is
returned by the review API.

## Historical truth

The exact six design issues named in the 2026-09-03 handoff could not be
recovered from the canonical repository, Git history, checked-in docs, visible
handoff attachments, or available rollout summaries. They are intentionally
not reconstructed from inference. This remains a Round G evidence blocker.

The following facts are independently recoverable from
`docs/PIE_RETIRED_MULTIPERIOD_20260714.md` and the current code:

- Eleven valid multi-period pie questions still exist.
- The only historical serving-field change was `is_pregenerated=false`.
- The current `PieChartData` contract contains one `labels`/`values` set.
- `parse_legacy_chart_description()` returns `None` for the multi-period pipe
  shape, so the student path falls back to text rather than drawing a false
  single pie.

## Renderer truth

| Task 1 subtype | Current source of preview | Truth for review |
|---|---|---|
| `pie_chart` single period | Shared deterministic canvas renderer | Supported when the stored shape parses into `PieChartData` |
| `pie_chart` multi-period | Same parser and renderer contract | Unsupported; no fake single-pie preview |
| `bar_chart`, `line_graph`, `table` | Sanitized stored SVG | Existing legacy image path; review UI reuses the same shared image adapter |
| `process`, `map` | Not in `TASK1_SERVED_SUBTYPES` | Frozen pending a real spatial renderer |

The shared renderer is `frontend/app/task1-chart-renderer.js`; the student
Writing page and admin review page load that same file.

## Eleven-question reproduction and recommendation

Every row below was reconstructed from the checked-in retirement document and
passed through the actual backend `parse_legacy_chart_description()` function.
All eleven returned `None`, giving 11/11 `renderer_unsupported` results. The
review recommendation is `needs_fix` for engineering follow-up; keep the
serving flag unchanged until a multi-period renderer is separately shipped and
verified.

| # | Question ID | Stored shape | Render | Issue | Recommendation |
|---:|---|---|---|---|---|
| 1 | `491a28c2-4bfa-47f3-8e3b-484695df73d2` | Energy Source · 2010 · 2023 | NO | `renderer_unsupported` | `needs_fix`; no reactivation |
| 2 | `3d8d7bc5-915f-4ca2-b343-7fab5461d4fc` | Energy Source · 2010 · 2023 | NO | `renderer_unsupported` | `needs_fix`; no reactivation |
| 3 | `ab70b2fd-b96b-435b-81ca-ee7864be2c10` | Energy Source · 2000 · 2010 · 2020 | NO | `renderer_unsupported` | `needs_fix`; no reactivation |
| 4 | `6b3c1090-fca5-48c3-886f-92d25195fb69` | Energy Source · 2010 · 2023 | NO | `renderer_unsupported` | `needs_fix`; no reactivation |
| 5 | `85f9e3b7-8695-467c-9cf5-402324f7e6da` | Energy Source · 2010 · 2023 | NO | `renderer_unsupported` | `needs_fix`; no reactivation |
| 6 | `642acea1-339d-4f75-a0f0-1eb25afdd32a` | Energy Source · 2010 · 2023 | NO | `renderer_unsupported` | `needs_fix`; no reactivation |
| 7 | `2e0a374b-68d2-458f-adca-d5a4a45da0c7` | Sector · 2010 · 2023 | NO | `renderer_unsupported` | `needs_fix`; no reactivation |
| 8 | `0727e5e7-2749-4a35-8922-92ad99181a2f` | Energy Source · 2010 · 2023 | NO | `renderer_unsupported` | `needs_fix`; no reactivation |
| 9 | `ed6a48ea-a177-4661-929c-df51b97e523e` | Sector · 2010 · 2023 | NO | `renderer_unsupported` | `needs_fix`; no reactivation |
| 10 | `ae202b45-7ad9-4622-8132-b0bd749ebdf6` | Energy Source · 2010 · 2023 | NO | `renderer_unsupported` | `needs_fix`; no reactivation |
| 11 | `a4665e22-f250-42c7-9f2f-4c6a77ea2d9c` | Energy Source · 2010 · 2023 | NO | `renderer_unsupported` | `needs_fix`; no reactivation |

## Human-review model

The migration adds `review_status` (default `pending`), `review_issue`,
`review_note`, `reviewed_at`, and `reviewed_by`. Allowed status values are
`pending`, `approved`, `needs_fix`, and `retired`. Allowed issue categories are
high-level only: `renderer_unsupported`, `data_shape_invalid`,
`misleading_visual`, `label_collision`, `unreadable`, `content_issue`, and
`other`.

The admin API is:

- `GET /admin/writing/task1-review?review_status=...`
- `PATCH /admin/writing/task1-review/{question_id}`

The PATCH body accepts only the three review fields. The server derives both
audit fields. `is_pregenerated`, prompt, chart source, and all submission data
are outside the update contract.

## Verification status

- Focused backend/frontend/schema contracts: PASS.
- Browser desktop fixture review: PASS; 11 rows visible, source/preview/status
  controls usable, and a local Needs-fix save removed one row from Pending while
  preserving `not_pool_eligible`.
- Responsive contract: PASS by source (`@media (max-width: 800px)` stacks the
  review list/detail) and narrow-width interaction smoke. The connected Chrome
  driver did not expose a true device viewport override, so this is not claimed
  as a device-emulation pass.
- Production migration apply: NOT DONE by design.
- Production review write: NOT DONE by design.
- Exact 09-03 six-issue recovery: BLOCKED; authoritative source unavailable.
- Production-backed browser review: NOT DONE; requires the schema to be
  separately deployed and authorized.
