# Task 1 Chart Human Review — 2026-09-04

## Scope and safety boundary

This round adds the smallest admin human-review loop for `writing_questions`
Task 1 chart rows. Review metadata is separate from `is_pregenerated`: saving
`approved` means **Approved for engineering reactivation**, not serving approval,
and no review action reactivates a question. No production migration was
applied, no production data was written, and no writing submission data is
returned by the review API.

## Historical provenance

The 2026-09-03 handoff named an additional set of “six design issues”, but the
original list was not found after exhaustive recovery across the canonical
repository, Git history, checked-in docs, visible handoff attachments, and
available rollout summaries. The exact six are therefore **UNRECOVERABLE
HISTORICAL PROVENANCE**. No inference or fabricated replacement taxonomy was
made, and this provenance gap is non-blocking for the current review system.

The recoverable July evidence is independently consistent with the current
repository behavior. These are **verifiable historical facts**, not the lost
09-03 six issues:

1. The in-pool pie set contained 16 questions.
2. Five questions rendered as actual deterministic pies.
3. Eleven questions fell back to text.
4. The fallback coverage was approximately 69% of in-pool pies.
5. `parse_legacy_chart_description()` supports a single pie with two columns;
   multi-period shapes such as `Energy Source | 2000 | 2010 | 2020` are not
   supported and consequently fall back to text.
6. Those 11 questions were soft-retired by changing only
   `is_pregenerated=false`; the action was reversible, and no multi-period pie
   renderer was shipped.

The exact 11 IDs, their source descriptions, and the serving-field truth are
independently recoverable from `docs/PIE_RETIRED_MULTIPERIOD_20260714.md`.

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
  driver did not expose a true device viewport override; true mobile device
  review was unavailable and is non-blocking under the reconciled gate.
- Production migration apply: NOT DONE by design.
- Production review write: NOT DONE by design.
- Exact 09-03 six-issue recovery: UNRECOVERABLE PROVENANCE GAP; not fabricated
  and non-blocking.
- Production-backed browser review: NOT DONE; requires the schema to be
  separately deployed and authorized.

## Round G gate

**ROUND G GATE: PASS — HISTORICAL PROVENANCE GAP DOCUMENTED, NOT FABRICATED**

The current taxonomy is grounded in renderer capability and the 11/11 local
reproduction, not attributed to the unrecoverable 09-03 list. No question was
reactivated, no production migration or write was performed, and the change
remains local pending Galant review/push.


## Human Review Sample — 2026-09-05

### Evidence boundary and preflight

Round O baseline, reverified in the canonical checkout: HEAD `968f5950ee68ff28d6ff77a2937156b77ccbe210`; origin/main `88eada001f89597eb7721ea425c6fd4af23edde3`; ahead/behind **8/0**, clean. No application code, renderer, migration or production state was changed. The historical sections above are preserved verbatim, including **UNRECOVERABLE HISTORICAL PROVENANCE** for the lost six design issues.

**Round O content gate: BLOCKED — the three original prompts and their unit/population context were not recovered.** The rubric, source-data analysis and local UI exercise below are completed, but the content dispositions are provisional. They must not be presented as three fully validated IELTS questions or operator approval of production rows. This is an evidence gap distinct from the older six-issue provenance gap; exact task instructions are directly necessary for this review.

Current tracked source and `git log --all -S <first canonical UUID>` were checked. Matches were the retirement document (`4eea975`, `c3e30b1`), review infrastructure (`562b88e`) and rollout documentation (`d6ee879`). The retirement document has IDs and compressed value tables, not original prompts, exact raw description bytes, geography/population or explicit units. Existing `test_task1_review.py` assigns synthetic household-expenditure/energy prompts and different data to real IDs; those are API fixtures, **not canonical content evidence**. No prompt was borrowed from those tests or invented from the chart heading. A path to an existing original-prompt snapshot was requested; none was available when this evidence was written. No production request was used to fill the gap.

### TASK1_RETIRED_11

The set was parsed programmatically from the retirement document's numbered Markdown table, not retyped. Ordered equality was asserted against both its embedded rollback-SQL UUIDs and the original Round G review table: **11/11 exact matches; 11 unique IDs**. Source SHA-256: `ed1c2acf0503de1817954e6f1bd3c583893798059e8eb2c58e6b1c809ad93f1a`.

The compressed arrows/slashes were expanded into category-by-period pipe tables without dropping values, relabeling categories or changing periods. These are **lossless reconstructions of the documented values**, not a claim to have recovered byte-identical stored `chart_description` or current production data. All reconstructions were passed through actual `parse_legacy_chart_description()` and `_task1_review_row()`.

| # | Question ID | Periods × categories | Column sums | Renderer result |
|---:|---|---|---|---|
| 1 | `491a28c2-4bfa-47f3-8e3b-484695df73d2` | 2 × 5 | 100 / 100 | `renderer_unsupported` |
| 2 | `3d8d7bc5-915f-4ca2-b343-7fab5461d4fc` | 2 × 5 | 100 / 100 | `renderer_unsupported` |
| 3 | `ab70b2fd-b96b-435b-81ca-ee7864be2c10` | 3 × 5 | 100 / 100 / 100 | `renderer_unsupported` |
| 4 | `6b3c1090-fca5-48c3-886f-92d25195fb69` | 2 × 4 | 100 / 100 | `renderer_unsupported` |
| 5 | `85f9e3b7-8695-467c-9cf5-402324f7e6da` | 2 × 5 | 100 / 100 | `renderer_unsupported` |
| 6 | `642acea1-339d-4f75-a0f0-1eb25afdd32a` | 2 × 5 | 100 / 100 | `renderer_unsupported` |
| 7 | `2e0a374b-68d2-458f-adca-d5a4a45da0c7` | 2 × 4 | 100 / 100 | `renderer_unsupported` |
| 8 | `0727e5e7-2749-4a35-8922-92ad99181a2f` | 2 × 5 | 100 / 100 | `renderer_unsupported` |
| 9 | `ed6a48ea-a177-4661-929c-df51b97e523e` | 2 × 4 | 100 / 100 | `renderer_unsupported` |
| 10 | `ae202b45-7ad9-4622-8132-b0bd749ebdf6` | 2 × 5 | 100 / 100 | `renderer_unsupported` |
| 11 | `a4665e22-f250-42c7-9f2f-4c6a77ea2d9c` | 2 × 5 | 100 / 100 | `renderer_unsupported` |

For **each of the 11**: historical retirement document says `is_pregenerated=false`; local reconstruction also uses false. Current production serving state is **NOT RE-VERIFIED**. Original prompt and stored SVG are **UNKNOWN**; chart family is historically `pie_chart`. Local review metadata starts at `pending` with null issue/note/audit fields, explicitly a test setup rather than historical production review metadata. No fabricated timestamps or usage counts are treated as source truth. The review API's default used_count=0 is only adapter behavior.

Mechanical inventory only, not a bulk content review: **7 two-period/five-category**, **3 two-period/four-category**, **1 three-period/five-category**. Header families are Energy Source (9) and Sector (2). The three sizes describe one category×period matrix grammar, not eleven distinct parser problems. Arithmetic totals are 100 per period, but that alone does not prove mutually exclusive categories, percentage units or a valid task prompt.

### 1. Review rubric

This is a local editorial decision rubric, not an official IELTS score. Apply it to the original prompt, declared measurement/unit, full data and faithful visual together; do not substitute parser failure for content judgment.

| Criterion | Required question / gate |
|---|---|
| A — IELTS validity | Is the original task instruction intelligible and consistent with the visual? Are subject, population, periods and unit available? Are there sufficient supported comparisons for a 150+ word response? Missing prompt/context means UNKNOWN, not PASS |
| B — Data integrity | Validate every value and period sum; category/period labels complete; shared denominator established; categories mutually exclusive; units consistent. Record arithmetic integrity separately from semantic integrity |
| C — Visual recoverability | Preserve every category, value, period and relationship. Determine whether the existing single-chart shape can encode them, whether static panels would be faithful, or whether new rendering is necessary. No silent period deletion, category merger or invented visual |
| D — Learning value | Identify actual ranking, crossover, contrast, proportional change and overview opportunities. A complex matrix without distinctive comparisons does not automatically deserve engineering |
| E — Cost | LOW: bounded manual asset/metadata preparation after source validation. MEDIUM: reviewed multi-panel asset, accessibility and release integration. HIGH: unresolved semantics, reauthoring, or reusable renderer/schema/parser work. Costs are relative judgments, not delivery estimates |
| F — Disposition | KEEP_RETIRED: hold while unsafe/unknown or low value; distinguish temporary hold from permanent retirement. DATA_FIX_CANDIDATE: existing supported shape after 100% semantic preservation. STATIC_FALLBACK_CANDIDATE: faithful panels possible after content gates. ENGINEERING_FIX_CANDIDATE: high-value recurring shape clears the engineering threshold |

Confidence is HIGH for directly reproduced technical/numerical findings; MEDIUM/LOW for conclusions depending on missing prompt context. `approved` remains engineering-review metadata, **never permission to serve**. No state transition may set `is_pregenerated=true`.

### 2–3. Selected questions and rationale

| Sample | Question | Why selected |
|---|---|---|
| A | `2e0a374b-68d2-458f-adca-d5a4a45da0c7` | Two periods/four sector categories: smallest clear composition candidate, with a leading-category crossover; among the two Sector rows this one has the larger Residential gain (5 vs 4). Closest to bounded visual recovery |
| B | `ab70b2fd-b96b-435b-81ca-ee7864be2c10` | Only three-period row, five categories; intermediate observations carry information lost by collapsing to endpoints. Tests whether extra complexity buys learning value |
| C | `6b3c1090-fca5-48c3-886f-92d25195fb69` | Two periods/four categories yet Electricity sits alongside Natural Gas/Oil/Renewable. Same simple geometry as A, different semantic risk; not automatically invalid or permanently retired |

Selection was deterministic from the documented shape and values. No dispositions were assigned to the remaining eight.

### 4. Sample A — Sector, 2010 / 2023

Original prompt: **NOT RECOVERED**. Reconstructed documented source:

| Sector | 2010 | 2023 |
|---|---:|---:|
| Industrial | 32 | 28 |
| Residential | 26 | 31 |
| Transport | 25 | 22 |
| Commercial | 17 | 19 |
| Total | 100 | 100 |

1. **Expected visual semantics:** if the missing prompt confirms shares of one common whole, the student should see two separately labeled pies, 2010 and 2023, with equal visual size, consistent category colors, all four labels and the original unit/context. We cannot supply a country or claim these are energy shares from `Sector` alone.
2. **Text fallback loss:** numbers survive, but instant part-to-whole/rank comparison is lost; students must reconstruct the overview mentally. The documented crossover is Industrial 32→28 versus Residential 26→31. The gap changes from Industrial +6 to Residential +3.
3. **Renderer limitation:** three pipe columns versus the parser's exact two-column contract; `PieChartData` and the shared renderer accept one flat values array. A single pie cannot retain two separate denominators and year labels.
4. **Faithful solution without new renderer:** a manually authored, reviewed two-panel static asset is geometrically possible. It must preserve labels/values, provide an equivalent accessible table and be bound to a source hash; source prompt/unit recovery comes first. No asset was authored this round.
5. **Can data correction save it?** No numerical correction is indicated. Rewriting arrows as a pipe matrix helps inspection but remains unsupported. Splitting into two separate questions or keeping only one year is not a data fix.
6. **Would a data change alter the question?** Removing a year or merging categories removes the crossover comparison. Recasting the task as a table/chart subtype changes the requested visual task; do not call that lossless normalization.
7. **Worth saving?** Provisionally yes: a compact four-category comparison offers a useful overview, ranking change, gains/losses and percentage-point language, assuming units are confirmed. Exact task-instruction validity and 150-word sufficiency remain UNKNOWN without the prompt.

Disposition: **STATIC_FALLBACK_CANDIDATE (provisional)**. Engineering cost: **LOW** for this two-panel asset after validation; shared asset/accessibility integration is MEDIUM. Confidence: **MEDIUM** for recovery recommendation; HIGH for arithmetic/unsupported reproduction. Keep serving false.

### 5. Sample B — Energy Source, 2000 / 2010 / 2020

Original prompt: **NOT RECOVERED**. Reconstructed documented source:

| Energy Source | 2000 | 2010 | 2020 |
|---|---:|---:|---:|
| Coal | 32 | 24 | 12 |
| Natural Gas | 28 | 31 | 28 |
| Renewable | 8 | 12 | 35 |
| Nuclear | 18 | 18 | 15 |
| Oil | 14 | 15 | 10 |
| Total | 100 | 100 | 100 |

1. **Expected visual semantics:** three distinct period compositions, preserving all five category identities. If the source confirms percentages, three same-sized pies with common color mapping and one accessible table would retain the stated comparisons. Population and whether this means production/consumption/electricity generation remain unknown.
2. **Text fallback loss:** the rise of Renewable and decline of Coal are less immediately visible; intermediate peak/plateau evidence is easy to overlook. Renewable 8→12→35 replaces Coal as the largest category; Coal falls 32→24→12.
3. **Renderer limitation:** four columns encode three values per category, outside both the two-column parser and one-array client schema. Folding 15 observations into one pie creates a combined total of 300 and invents a cross-period whole; normalizing that to 100 also misstates every period's composition.
4. **Faithful non-renderer solution:** a reviewed three-panel static asset is plausible, with readable labels and an accessible matrix. It requires more layout/accessibility verification than A. Mobile stacking must preserve period labels and reading order; no asset implementation here.
5. **Can data correction save it?** The documented column sums are already 100. No lossless normalization to the current single-pie model exists.
6. **Would changing data alter the task?** Dropping 2010 hides Natural Gas 28→31→28 and Oil 14→15→10; it also hides Nuclear's initial 18→18 plateau. Endpoints alone falsely simplify the available narrative. Averaging years or merging categories changes the task.
7. **Worth saving?** Provisionally yes: unlike complexity for its own sake, this row supports leader change, accelerating Renewable growth and non-monotonic contrasts. This is the strongest learning-value candidate of the three, conditional on source context/units and a valid prompt. `Renewable` boundaries must be confirmed, not assumed from arithmetic alone.

Disposition: **STATIC_FALLBACK_CANDIDATE (provisional)**. Engineering cost: **MEDIUM**. Confidence: **MEDIUM** for recovery recommendation; HIGH for the numerical comparisons and inability of the current renderer to retain them. No renderer implementation or serving approval.

### 6. Sample C — mixed Energy Source labels, 2010 / 2023

Original prompt: **NOT RECOVERED**. Reconstructed documented source:

| Energy Source | 2010 | 2023 |
|---|---:|---:|
| Natural Gas | 42 | 38 |
| Electricity | 35 | 44 |
| Oil | 15 | 12 |
| Renewable | 8 | 6 |
| Total | 100 | 100 |

1. **Expected visual semantics:** mathematically two four-slice compositions; semantically a pie is legitimate only after the missing source establishes non-overlapping categories and one denominator. Electricity is an energy carrier, while gas/oil and Renewable can describe sources. A final-energy-use definition could make these categories exclusive; a generation-source interpretation might not. The original prompt could settle this.
2. **Text fallback loss:** the rank reversal (Gas ahead by 7, then Electricity ahead by 6) is less salient. Text also cannot cure category ambiguity; showing a polished pie would falsely imply a verified partition if the categories overlap.
3. **Renderer limitation:** the same three-column/two-period structural failure as A. This is not evidence of corrupt numeric data and is not the reason for the stronger content hold.
4. **Faithful non-renderer solution:** only conditional on a recovered definition of Electricity versus Renewable. A static image can draw the values accurately yet still be a visual lie about the whole. Do not author it while that meaning is unresolved.
5. **Can data correction save it?** No supported correction can be established. If the original source contains an omitted explanatory note, restoring it verbatim may clarify the task, but that is an unproven recovery path. Inventing a new category name is not a fix.
6. **Would changing data alter the task?** Renaming Electricity to Nuclear, removing Renewable, assigning assumed subcategories or redistributing percentages changes categories/data and therefore the question. Reject all of these.
7. **Worth saving?** Defer. Potential comparison value is MEDIUM, but evidence recovery has priority over visual work. A is presently a cleaner use of bounded authoring effort. The evidence does **not** justify declaring C intrinsically worthless or permanently retired.

Disposition: **KEEP_RETIRED (temporary evidence hold, not permanent rejection)**. Engineering cost: **HIGH** for safe recovery if semantic reconstruction/reauthoring is needed; drawing two pies alone would be cheap but insufficient. Confidence: **HIGH** in holding for clarification, LOW in any claim that the original question itself is invalid. Serving remains false.

### 7. Decision tables

All IELTS validity cells remain UNKNOWN pending original prompts. This limitation is not replaced by the retirement document's broad historical description of the rows as valid pies.

| Question | IELTS validity | Data integrity | Recoverability | Learning value | Cost | Disposition |
|---|---|---|---|---|---|---|
| A — `2e0a374b-68d2-458f-adca-d5a4a45da0c7` | UNKNOWN; plausible comparison task | 100/100; unit/population unknown | Conditional two-panel static | HIGH, provisional | LOW | STATIC_FALLBACK_CANDIDATE, provisional |
| B — `ab70b2fd-b96b-435b-81ca-ee7864be2c10` | UNKNOWN; plausible richer comparison | 100/100/100; source definitions unknown | Conditional three-panel static | HIGH, provisional | MEDIUM | STATIC_FALLBACK_CANDIDATE, provisional |
| C — `6b3c1090-fca5-48c3-886f-92d25195fb69` | UNKNOWN; semantic clarification needed | 100/100; category exclusivity unresolved | No safe asset approval yet | MEDIUM potential | HIGH | KEEP_RETIRED, temporary hold |

| Disposition | Sample count |
|---|---:|
| KEEP_RETIRED | 1 temporary hold |
| DATA_FIX_CANDIDATE | 0 |
| STATIC_FALLBACK_CANDIDATE | 2 provisional |
| ENGINEERING_FIX_CANDIDATE | 0 |

### 8. Static fallback decision

**Worth considering a small reviewed static pilot after source recovery; NO IMPLEMENTATION THIS ROUND.** Two or three static panels could faithfully express the matrices without implementing a reusable multi-period runtime renderer. Advantages are deterministic content and low runtime complexity. However, a static image is not automatically approved just because it can be drawn.

Required future controls: question ID + immutable source hash + asset version; exact labels/units/values/period correspondence; accessible data table and meaningful image description; legibility at student viewport sizes; cache/version invalidation; review whenever source changes; independent visual checking. Manual authoring, asset storage/delivery and reviewer workload are real costs. The existing pie review adapter ignores pie SVG and only accepts parsed pie data, so **uploading an SVG alone is not a working fallback integration**. Even a static approach requires a separately scoped serving/preview/accessibility decision; no runtime shortcut was added.

### 9. Data-fix decision

**Zero confirmed DATA_FIX_CANDIDATE samples.** All three retain multiple independent period vectors after lossless delimiter normalization. The existing model has no period dimension. Deleting a year, averaging years, flattening into 200/300 total slices, merging categories, changing subtype/instruction, or inventing missing context does not preserve semantics 100%. C's unit/taxonomy might be recoverable from a genuine original source, but it cannot be supplied by inference. Never ship a visual lie.

### 10. Engineering threshold

**Do not reopen multi-period renderer implementation on current evidence.** Shape uniformity is more promising than a five-unrelated-formats scenario: all eleven are category×period matrices, in three sizes. But eight are not content-reviewed, all original prompts remain unavailable, semantic category validity is not proven by totals, and apparent topical repetition may reduce distinct learning value. Historical 11/16 coverage is not sufficient justification.

Proposed gates for a separately authorized future engineering decision (policy proposal, not measured results):

1. At least **6 distinct high-value questions** in this retired set pass complete original-prompt/unit/data/semantic review; similarity is assessed for educational duplication, not counted as six independent benefits automatically.
2. A common source schema retains 2–3 explicitly labeled periods, 2–8 consistent categories, one documented unit/denominator and complete values; no question-specific relabeling or hidden special cases.
3. Parser/schema examples and negative cases are stable; inconsistent totals or ambiguous partitions fail closed.
4. Side-by-side/stacked panels, consistent categories, accessibility and student/admin preview parity are specified; visual fidelity tests cover all supported sizes before serving consideration.
5. Reusable rendering's benefit exceeds reviewed-static asset costs and maintenance. Quantified eligible demand, not the raw count of parser failures, drives the comparison.

These gates are not yet met. A static pilot after evidence closure can inform costs without authorizing a renderer now.

### 11. Review notes and local Admin UX

The existing `admin.html` and `task1-chart-renderer.js` were served unchanged through a loopback-only preview. External SDK/config/font loads were removed only from the served test response; a CSP denied network connections, and fetch/auth were in-memory fixtures. A conspicuous LOCAL REVIEW FIXTURE banner disclosed missing original prompts. The UI correctly showed `(no prompt)`; it did not show invented task text. Pie previews remained explicit unsupported fallbacks, with no canvas or image pretending to render these matrices.

Browser interaction covered all three source previews, status/issue/note controls, Save review and updated queue/readback. A/B simulated `needs_fix` with `renderer_unsupported`; C simulated `retired` with `content_issue`. The C note explicitly says HOLD, not permanent rejection. Each row remained `not_pool_eligible`. Full-page screenshots were visually inspected:

- [Sample A local review](evidence/task1_review_samples_20260905/sample-a.png)
- [Sample B local review](evidence/task1_review_samples_20260905/sample-b.png)
- [Sample C local review](evidence/task1_review_samples_20260905/sample-c.png)

No Admin presentation fix was needed: the operator can inspect source, detect unsupported rendering, edit metadata and see serving separation. Missing content is a source-evidence blocker, not a UI rendering failure. Notes are scrollable in the existing textarea; complete saved notes follow.

**Sample A — `2e0a374b-68d2-458f-adca-d5a4a45da0c7`:**

> PROVISIONAL: two four-sector compositions, 2010/2023; totals 100 each. Preserve both periods and category labels. Consider two faithful static panels only after original prompt/unit/population recovery. Single-pie collapse changes the task. Keep serving false.

**Sample B — `ab70b2fd-b96b-435b-81ca-ee7864be2c10`:**

> PROVISIONAL: three five-source compositions, 2000/2010/2020; totals 100 each. Coal decline and Renewable rise require all three panels; dropping 2010 hides non-monotonic series. Static candidate after original prompt/units recovery. Existing single-pie renderer cannot express this. Keep serving false.

**Sample C — `6b3c1090-fca5-48c3-886f-92d25195fb69`:**

> HOLD, not permanent rejection: Gas/Electricity/Oil/Renewable totals are 100 in both years, but carrier/source labels may overlap. Original prompt and denominator are missing; establish mutually exclusive categories before any static asset. Renaming Electricity or merging categories is not a lossless fix. Keep serving false.

### 12. Remaining eight and verification

**Do not bulk-review or relabel the remaining eight.** Their only Round O processing was mechanical ID/shape/parser verification. Keep historical serving retirement and unknown production review state distinct. Next action, requiring a separate user-directed continuation: recover the three selected questions' original prompt, complete description, units/denominator and available original visual from an existing trustworthy snapshot, or a later authorized read-only source after access is available. Close these three evidence gaps first; only then decide whether to review the remaining eight with this rubric. Do not start them now.

Executed local evidence:

- Canonical IDs: **11/11**, exact ordered agreement among retirement table, embedded SQL list and original Round G table. No SQL from the document was executed.
- Actual backend parser/preview adapter: **11/11 renderer_unsupported**, chart_data null; every documented period total 100.
- Focused suite: `test_task1_review.py`, `test_frontend_task1_review_contracts.py`, `test_writing_pie_chart.py` — **38 passed**, 5 existing deprecation warnings, Python 3.11 / Node20. Includes shared renderer contracts and the approved-does-not-reactivate regression. **Full pytest deliberately not run** for this docs/evidence-only change.
- Existing Human Review UI: **3/3 local previews and 3/3 in-memory save/readbacks** exercised in Chromium via agent-browser; retained source tables, explicit fallback, issue labels, metadata and serving separation.
- Same three captured PATCH bodies replayed through the real FastAPI endpoint with the existing hermetic in-memory database fixture: **3/3 PASS**; only review/audit fields changed, prompt/chart/serving fields unchanged, remaining eight fixture rows still pending and unmodified. This is not a disposable PostgreSQL or production write.
- `git diff --check`: PASS. Original human-review document prefix verified byte-for-byte preserved; application and migration files unchanged.

**Final scope result:** canonical recovery/reproduction and review workflow verification PASS; three source-data assessments documented; complete human content review remains **BLOCKED — original prompt and unit/population context unavailable**. No claim of human operator approval or production metadata update is made. The prior production SQL/Render SHA/recoverability blockers remain unchanged. Production DB writes = 0; production review PATCH = 0; migration apply = 0; deployment = 0; push = NO.


## Source Context Recovery — 2026-09-05

### 1. Scope and new finding

Round O-R baseline: HEAD `dcd08bac24293aa0ca7c13d25d44d82ea1ed7743`, origin/main `88eada001f89597eb7721ea425c6fd4af23edde3`, ahead/behind **9/0**, working tree clean; reverified in `/Users/yichengchiu/dev/Blabby/blabby`.

**All three original prompts and complete stored chart descriptions were recovered from historical project tool results.** This supersedes Round O's limited search result that the prompts were not available there. It does not rewrite that historical assessment, turn the reconstructed fixtures into original sources, or resolve the separate **UNRECOVERABLE HISTORICAL PROVENANCE** of the 09-03 six design issues.

The remaining content gate is narrower: the recovered per-row prompt/table fields do not explicitly state a unit, original SVG contents were not recovered, and C still lacks an exclusivity definition for Electricity versus Renewable Energy. The historical renderer suggests percentage-style display but is not a per-row unit declaration. All three therefore remain **KEEP_RETIRED_PROVENANCE_INSUFFICIENT** under this round's strict confirmation gate. This is not proof that the source was permanently lost or that the questions are intrinsically invalid.

Only this appendix and the [minimal source archive](evidence/task1_review_samples_20260905/source-context-recovery.json) are new. No renderer, static fallback, source data, question text, application code or migration was changed. No production query or PATCH was issued.

### 2. Search coverage and method

| Search surface | What was actually searched | Result / limitation |
|---|---|---|
| Canonical current tree | 190 tracked files plus hidden/ignored text sources; UUIDs, exact numeric/category sequences and recovered distinctive prompt strings; docs, SQL, JSON/CSV, tests, seeds, import/generation scripts and reports | Only existing retirement/review/readiness docs and synthetic API test matched; no original question bank export |
| Reachable Git history | 466 commits; `git log -S/-G`, `git show`, historical `git grep` at `071695b`, `c3e30b1`, `4eea975`, `562b88e`; `git cat-file --batch` scan of 771 historical text blobs | 9 matching blobs across the same four documentation/test paths; no original prompt in Git. Deleted JSON/CSV/SQL/XLSX path search found only unrelated `frontend/app/config.json`, not a deleted question dataset |
| Desktop rescue | Filesystem text search under `/Users/yichengchiu/Desktop/Blabby`, including available worktrees/reports; no rescue Git command or writes | Retirement document only; no additional original source. Search completed within the 35-second bound, no metadata repair |
| Nearby archives / Downloads | Canonical parent contains only canonical repo; filesystem search of `/Users/yichengchiu/Downloads/blabby`; 16 Blabby-named text handoff/context files in Downloads | Downloads rescue again contains the retirement document; 16 handoffs had no sample/value match |
| Local project operation logs | Four Blabby Claude project directories, plus available July Codex sessions; exact UUIDs and distinctive value/phrase matching | Two Claude logs carry relevant evidence; one contains original rows, the other later metadata. Prior scratchpad path cited in the log no longer exists |
| Attachments / external banks | Existing attachment tree, including 333 text attachments, exact IDs and distinctive numeric patterns | Only current recovery request/attachment metadata matched IDs; no source CSV/XLSX bank identified. A remote external bank/backup or live original SVG is **EXTERNAL SOURCE NOT AVAILABLE**, not claimed searched. Unrelated personal sheets/course material were not treated as source data |

Text scans excluded Git metadata, dependency/runtime directories and env/credential files. Source strings, not broad private-data exports, drove the searches. Reachable Git history is not a proof about unreachable/pruned objects. Lack of a local SVG match is not proof the database no longer holds it.

### 3. Recovered evidence chain

The archive contains only the three target rows and their relevant metadata, with source paths, line numbers, UTC timestamps, matching request/result tool IDs and SHA-256 hashes of exact archived log lines. No session-wide export, credentials, submissions or other question rows are copied.

| Evidence | Archived source | Meaning |
|---|---|---|
| S1 | `~/.claude/projects/-Users-yichengchiu-Desktop-Blabby/80b8cb73-c3f7-478f-a62c-507ea8ff0396.jsonl`, request line 1416 / result 1417, 2026-07-14 05:59:24.421 / 05:59:28.920 UTC | Successful Supabase `execute_sql` SELECT of `id,prompt,chart_description,created_at` from in-pool task1/pie_chart rows in project `mkwywkwruyqzdhuzwnoa`. Request tool ID equals response tool_use_id; these are actual historical tool-result values, not an assistant summary |
| S2 | Same log, request 1479 / result 1480, 2026-07-14 06:04:07.859 / 06:04:13.993 UTC | Historical soft-retirement operation returned the three IDs with subtype pie_chart and `is_pregenerated=false`. O-R inspected this record as text; its UPDATE was **not executed** |
| S3 | `~/.claude/projects/-Users-yichengchiu-Desktop-Blabby/df81760a-b94c-4dde-bf69-d93a65eda394.jsonl`, request 47 / result 48, 2026-09-03 13:04:21.314 / 13:04:26.039 UTC | Successful historical SELECT of subtype, serving flag, used_count, created_at, SVG length, prompt length and description-null flag |

S1's subsequent scratchpad copy (same July log lines 1432/1436) is derivative evidence, not an independent source; its original `/private/tmp/claude-501/.../scratchpad/inpool_pies.json` is unavailable now. S3 proves historical nonempty SVG storage, **not recovery of SVG contents or present-day availability**. The S3 prompt lengths match the recovered prompts, but equal length alone is not a proof of unchanged prompt bytes between July and September.

### 4. SAMPLE_IDENTITY_TABLE

| Field | A | B | C |
|---|---|---|---|
| id | `2e0a374b-68d2-458f-adca-d5a4a45da0c7` | `ab70b2fd-b96b-435b-81ca-ee7864be2c10` | `6b3c1090-fca5-48c3-886f-92d25195fb69` |
| task type / subtype | task1 / pie_chart, S1 filter and S2/S3 | Same | Same |
| prompt/question | Exact prompt RECOVERED, S1 | Exact prompt RECOVERED, S1 | Exact prompt RECOVERED, S1 |
| chart_description / raw data | Exact stored pipe table recovered | Exact stored pipe table recovered | Exact stored pipe table recovered |
| period labels | 2010, 2023 | 2000, 2010, 2020 | 2010, 2023 |
| categories | Industrial, Residential, Transport, Commercial | Coal, Natural Gas, **Renewable Energy**, Nuclear, Oil | Natural Gas, Electricity, Oil, **Renewable Energy** |
| title / subtitle | Original SVG title/subtitles NOT RECOVERED; source prompt provides context, not byte-exact visual title | Same | Same |
| unit | PARTIAL: % intended by historical renderer; no explicit unit in S1 prompt/table | Same | Same |
| base/context | Energy consumption by sector, a developed country, each stated year | Energy consumption by source, a European country, three stated years over 20 years | Household energy consumption by source, a developed country, each stated year |
| created_at, UTC (S1/S3 agree) | 2026-07-07 06:12:31.064086+00 | 2026-07-07 06:07:09.451709+00 | 2026-07-07 06:07:54.458501+00 |
| updated_at | No such dedicated column in reviewed schema; actual row last modification UNKNOWN | Same | Same |
| historical serving | true at S1 query predicate; false S2 and S3 | Same | Same |
| used_count at S3 | 1 | 1 | 1 |
| SVG / prompt lengths at S3 | 6647 / 211 characters | 10879 / 277 | 7104 / 221 |
| description at S3 | Non-null | Non-null | Non-null |
| source / generation metadata | No row-specific model, prompt-template version, source citation, schema version or generation/import batch ID recovered | Same | Same |
| review metadata | Round O test-only pending→needs_fix; production metadata unverified | Round O test-only pending→needs_fix | Round O test-only pending→retired |

The retirement summary shortened **Renewable Energy** to **Renewable** in B/C. The original spelling is now retained in the archive; original values/periods match the documented matrix exactly. This is a documentation/provenance correction, not a database or source-data edit. A's stored description matches the reconstructed matrix without that abbreviation.

### 5. PROVENANCE TIMELINE and Git sanity

| Event | Evidence and what it does / does not prove |
|---|---|
| 2026-06-18 original schema | `20260618040300_create_writing_module_tables.sql` records `prompt text NOT NULL`, chart_description and created_at. Companion `20260618045448...` adds chart_svg, is_pregenerated and used_count. Files were later reconstructed from the ledger; their Git creation is not the table's creation event |
| 2026-06-18 generator | `f55d0a7` introduces `_writing_question_prompt`; it generates prompt plus chart_description. This is pipeline evidence, not sample creation attribution |
| 2026-07-07 multi-period generation | `679fd8b` contains the 1/2/3/4-period generation contract; `da96816` adds subtype replenishment; `68398da` rejects imageless Task 1; `071695b` is the latest inspected reachable revision before the samples' created_at timestamps. None is proven as the exact deployed generation revision for these rows |
| 2026-07-07 row creation timestamps | B 06:07:09Z, C 06:07:54Z, A 06:12:31Z, recovered from S1/S3. These are row timestamps, not Git commits or model run IDs |
| 2026-07-14 rendering change | `80b46d3` introduces deterministic single-pie parsing; explains unsupported multi-period behavior, not a content-authoring event |
| 2026-07-14 first full local source observation | S1 at 05:59:28.920Z contains original prompt/description. Exact initial model response/insert event and any prior edits are UNKNOWN |
| 2026-07-14 retirement event | S2 at 06:04:13.993Z confirms serving=false; no question-text rewrite is attributed to this operation |
| First observed Git source, all A/B/C | Canonical ancestry: `4eea97527521650bc3df554a63e6286de1063549`, 14:05:51+08. All-ref search also finds `c3e30b17ffafcbaf48b15879c111ccf3bc55f656`; both retirement files have blob `a0a86cbd5663c72172fc49059363e60f82c32b44`. These are documentation commits, not database insertion/retirement transactions |
| Last modified commit, all A/B/C | Actual source-row LAST MODIFIED COMMIT: **UNKNOWN / no row-to-Git version mapping**. Retirement document has no later value edit; Round G/O documentation additions are not question edits |
| 2026-09-03 latest recovered metadata | S3: all three still soft-retired, nonempty SVG and prompt. This is not a current production readback |

For every sample: FIRST OBSERVED COMMIT = canonical retirement-document commit above; RETIREMENT COMMIT = documentary `4eea975`, with the actual database event separately identified by S2; SOURCE FILE = original S1 operation log plus this scoped archive; source-row LAST MODIFIED COMMIT = UNKNOWN. No cherry-pick, checkout mutation, history repair or rescue-repository Git operation was used.

### 6. Generator/schema findings

The historical contract at `071695b:backend/main.py` is concrete:

- `_writing_question_prompt()` requests JSON containing **prompt + chart_description**. Multi-period pie instructions require the same categories across period columns and column sums between 95 and 105; examples use a category-by-year table. They do not request a dedicated unit, denominator, source citation or batch/version object.
- `_pregenerate_task1_subtype()` calls `claude-haiku-4-5-20251001`, strips JSON fences, checks prompt presence, passes the prompt/table into `generate_chart_svg()`, then persists task_type, subtype, prompt, chart_description, chart_svg, serving flag and used_count. SVG generation uses `claude-sonnet-4-6`. **These are code-level model settings, not verified per-row generation attribution.**
- `generate_chart_svg()` derives a title from prompt when none is supplied and requests percentage-style pie labels (`% label`). It does not create independently verified measurement metadata; the original SVG text would be needed to recover each actual displayed label/title.
- The database's non-null prompt column and S1 results directly disprove **SOURCE NEVER PERSISTED** for these prompts. Context was partly persisted as free text, not simply discarded into numeric values.
- The reviewed legacy schema has no dedicated unit, population/base, source/provenance, model/version, raw-response hash or import/generation batch field. It allows SVG storage, which S3 confirms was nonempty. Lack of structured fields does not prove that every missing detail was absent from that SVG.
- Today's single-pie path (`80b46d3` onward) explicitly carries chart title/unit and serializes unit into `Value (...)`; its optional `context` is not passed through `_pie_artifacts_from_response()`. This newer behavior cannot be used to backfill or certify the three older multi-period rows.

**Pipeline classification:** prompts/descriptions = **SOURCE STILL RECOVERABLE**, now recovered; original row-level unit/visual semantics and exact generator attribution = **UNKNOWN / partially recovered**, not proven lost during generation. Dedicated structured provenance fields were not part of the examined persistence contract. No claim that source context was universally never persisted is justified.

### 7. Sample A source recovery

Question: `2e0a374b-68d2-458f-adca-d5a4a45da0c7`. **Original prompt: RECOVERED**, exact S1 text:

> The pie charts below show the distribution of energy consumption by sector in a developed country in 2010 and 2023. Summarise the information by describing the main features, and make comparisons where relevant.

Exact stored chart description:

```text
Sector | 2010 | 2023
Industrial | 32 | 28
Residential | 26 | 31
Transport | 25 | 22
Commercial | 17 | 19
```

- Task instruction: describes main features and makes relevant comparisons, explicitly recovered above. Standalone prompt omits a 150-word sentence; the app's existing Task 1 writing UI displays 150 as its minimum target, so prompt wording alone is not the entire instruction surface.
- Chart title/subtitle: no original SVG text recovered; do not invent a visual title from the first sentence and label it original.
- Unit: **PARTIAL**. A 100-based proportional energy-consumption display is strongly indicated by the generator and matrix, but neither stored prompt nor header says `%`. Exact displayed units are not recovered.
- Population/base: one unspecified developed country's energy consumption, grouped by sector in 2010 and 2023. Each year is a separate composition; no absolute energy quantity, primary/final-energy accounting convention or geographic identity is provided. An unnamed country itself is not disqualifying for an IELTS-style generated task.
- Categories: Industrial, Residential, Transport, Commercial. Exclusivity **PLAUSIBLE**, consistent with ordinary sector grouping; exact boundary/accounting definitions are not recorded, so do not claim independently CONFIRMED taxonomy.
- Source: S1 exact query result; S2 retirement; S3 later metadata. Country and values are exercise context, not verified real-world statistics or a published dataset citation.
- Confidence: **HIGH** in recovered prompt/data/base; **MEDIUM** in intended percentage interpretation; insufficient for original visual-label confirmation.
- IELTS validity: **INSUFFICIENT_EVIDENCE** for the complete task under the O-R unit/visual gate. Wording, comparison targets and periods are appropriate; source units/visual presentation remain unverified. This is not INVALID.
- Final disposition: **KEEP_RETIRED_PROVENANCE_INSUFFICIENT**. Earlier static candidacy remains a potential engineering option, not CONFIRMED.

### 8. Sample B source recovery

Question: `ab70b2fd-b96b-435b-81ca-ee7864be2c10`. **Original prompt: RECOVERED**, exact S1 text:

> The pie charts below show the distribution of energy consumption by source in a European country across three decades: 2000, 2010, and 2020. Summarize the information by describing the main changes in energy sources over the 20-year period, and make comparisons where relevant.

Exact stored chart description:

```text
Energy Source | 2000 | 2010 | 2020
Coal | 32 | 24 | 12
Natural Gas | 28 | 31 | 28
Renewable Energy | 8 | 12 | 35
Nuclear | 18 | 18 | 15
Oil | 14 | 15 | 10
```

- Task instruction: main changes over the **20-year period**, relevant comparisons. The phrase “across three decades: 2000, 2010, and 2020” names three decadal reference years, not evidence for a 30-year interval or decade-wide averages. Clarifying it to “in 2000, 2010 and 2020” would be an editorial proposal, not an original source quote; no text edit made.
- Chart title/subtitles: original SVG text unavailable. Recovered period labels are 2000, 2010, 2020; retain all three.
- Unit: **PARTIAL**, same missing explicit per-row unit declaration as A. Percent display is intended by pipeline but not independently recovered from actual SVG labels.
- Population/base: energy consumption by source in **one European country** at three time points. The wording supports the same geographic referent over time and comparison of separate annual compositions, not three countries/populations. It does not name a fixed population cohort, future projections or forecast series. No documentary basis to claim real measured historical data or a named country exists.
- Categories: Coal, Natural Gas, Renewable Energy, Nuclear, Oil; exclusivity **PLAUSIBLE** for a source mix, with no explicit boundary/energy-accounting definition. This is not merely the question's period count.
- Source: S1 exact prompt/description; the original Renewable Energy label restores the abbreviated retirement summary. S3 stores an SVG length but no SVG content in that query result.
- Confidence: **HIGH** in source recovery and the same-country/three-time-point reading; **MEDIUM** in intended unit; actual-vs-synthetic data provenance remains unverified.
- IELTS validity: **INSUFFICIENT_EVIDENCE** for full validation. Rich comparisons and a reasonable task instruction are recovered; the minor “three decades” wording can be flagged separately, but does not solve the unit/visual gate.
- Final disposition: **KEEP_RETIRED_PROVENANCE_INSUFFICIENT**, not CONFIRMED static/engineering candidacy.

### 9. Sample C source recovery

Question: `6b3c1090-fca5-48c3-886f-92d25195fb69`. **Original prompt: RECOVERED**, exact S1 text:

> The pie charts below show the distribution of household energy consumption by source in a developed country in 2010 and 2023. Summarise the information by describing the main features, and make comparisons where relevant.

Exact stored chart description:

```text
Energy Source | 2010 | 2023
Natural Gas | 42 | 38
Electricity | 35 | 44
Oil | 15 | 12
Renewable Energy | 8 | 6
```

- Task instruction: main features and relevant comparisons, with explicit **household energy consumption** context in one developed country in 2010 and 2023.
- Chart title/subtitles: original SVG not recovered. Do not reinterpret Electricity as a sector or rename it.
- Unit: **PARTIAL**; percentage-style display is suggested, not recovered as an explicit per-row label.
- Population/base: household energy consumption by source, not national electricity-generation composition. This narrows the Round O ambiguity materially. No household count, named country, energy accounting basis or absolute total is specified.
- Categories: Natural Gas, Electricity, Oil, Renewable Energy. The prompt does **not** define Electricity as its generation source; it is listed as a supplied energy category within household consumption. Renewable Energy may mean direct household renewables, or may overlap electricity generated from renewables. Neither interpretation is explicitly selected by the source.
- Category exclusivity: **UNKNOWN**, not CONFIRMED and not proven CONFLICT. The restored household framing makes a mutually exclusive final-energy interpretation possible but does not establish it. No label/data rewrite may be used as evidence.
- Source: S1 exact context and full category label; S3 nonempty SVG metadata. No source note defining grid electricity versus direct renewables was recovered.
- Confidence: **HIGH** in household context recovery and necessity of a hold; **LOW** in any assertion that the categories are inherently invalid.
- IELTS validity: **INSUFFICIENT_EVIDENCE**. Clear years and arithmetic cannot resolve hidden category assumptions.
- Final disposition: **KEEP_RETIRED_PROVENANCE_INSUFFICIENT**. Do not save this question through an assumed redefinition or static visual.

### 10. Validity, final dispositions and static gate

| Sample | Original prompt | Explicit per-row unit / original visual | Base/context | Category semantics | IELTS classification | Final disposition |
|---|---|---|---|---|---|---|
| A | RECOVERED | PARTIAL / NOT RECOVERED | Developed country's annual energy consumption, by sector | PLAUSIBLE | INSUFFICIENT_EVIDENCE | KEEP_RETIRED_PROVENANCE_INSUFFICIENT |
| B | RECOVERED | PARTIAL / NOT RECOVERED | One European country's consumption by source at three time points | PLAUSIBLE | INSUFFICIENT_EVIDENCE | KEEP_RETIRED_PROVENANCE_INSUFFICIENT |
| C | RECOVERED | PARTIAL / NOT RECOVERED | Household consumption in one developed country | UNKNOWN exclusivity | INSUFFICIENT_EVIDENCE | KEEP_RETIRED_PROVENANCE_INSUFFICIENT |

A/B have plausible static geometry and recovered comparison context; C needs both source and category clarification. **None clears all static fallback conditions**, especially original unit/visual recovery; no candidate receives a CONFIRMED suffix. No source is declared INVALID or permanently discarded on this evidence. Review decisions here are metadata/documentary holds only; historical/local serving remains false, current production untouched.

### 11. Exact provenance-gap classification

| Sample / field | Classification | Evidence and exact remaining gap |
|---|---|---|
| A/B/C prompt + raw table + created_at | **E — SOURCE RECOVERED** | S1 exact archived DB result, linked to request/tool/project; values agree with canonical retirement record |
| A/B/C complete task context | **D — SOURCE PARTIALLY RECOVERED** | Base and periods recovered; original per-row displayed units/title/subtitles unavailable |
| A/B/C original SVG | **C — SOURCE MAY EXIST EXTERNALLY BUT NOT AVAILABLE** | S3 records lengths 6647/10879/7104, so SVG was stored in September; the query did not return its contents. Current database/backups were not accessed |
| C category exclusivity | **D — SOURCE PARTIALLY RECOVERED** | Household base recovered, but no definition separating Electricity and Renewable Energy |
| Dedicated unit/model/source/batch fields | **A — SOURCE NEVER PERSISTED**, narrowly at structured-field-contract level | No such dedicated fields in the reviewed original schema/inserts; do not extend this label to all prose/SVG context |
| Source deletion/history loss | **B — NOT ESTABLISHED** | No deleted source dataset found in reachable history; positive historical storage evidence forbids a claim of confirmed permanent loss |

The remaining missing artifact is specific: an existing original `chart_svg`/full source record or corresponding source-authoring specification for these exact three IDs, plus C's accounting definition if the SVG does not supply it. A later authorized read-only retrieval would be a new evidence source; do not rerun broad Git searches as though the original prompts were still absent. Do not use unavailable production read access as permission to alter schema or request credentials in chat.

### 12. Future content provenance requirement — documentation only

The demonstrated design weakness is incomplete **structured** provenance, not failure to save prompt text. A future contract should require: stable question ID; exact prompt/instruction; chart title and period subtitles; explicit unit; measure and per-period denominator/base; category definitions/exclusivity; periods and actual/forecast status; immutable structured values; synthetic/generated versus cited real-data source; source citation or generation model/version and prompt-template version; generator/schema version and run/batch ID; raw-response/source hash; created_at and versioned content modification metadata; original visual/asset hash if used; review state/audit separate from serving state. Optional generation context should not be silently lost during persistence.

No migration or new field was implemented. Unknown older fields must remain unknown; this contract is future backlog, not a retrospective metadata fabrication exercise.

### 13. Local review notes

Updated documentary notes are saved in the archive. No local database or production PATCH was necessary this round; Round O screenshots remain historical and were not edited to imply newly recovered content had been displayed there.

**Sample A:**

> Original prompt/table recovered from archival Supabase SELECT S1, 2026-07-14, UUID 2e0a374b-68d2-458f-adca-d5a4a45da0c7. Original table has no explicit unit; archived SVG content is unavailable. Keep retired for provenance insufficiency; no source edit or serving approval.

**Sample B:**

> Original prompt/table recovered from archival Supabase SELECT S1, 2026-07-14, UUID ab70b2fd-b96b-435b-81ca-ee7864be2c10. Original table has no explicit unit; archived SVG content is unavailable. Keep retired for provenance insufficiency; no source edit or serving approval.

**Sample C:**

> Original prompt/table recovered from archival Supabase SELECT S1, 2026-07-14, UUID 6b3c1090-fca5-48c3-886f-92d25195fb69. Original table has no explicit unit; archived SVG content is unavailable. Household context recovered, but Electricity/Renewable Energy exclusivity remains unresolved. Keep retired for provenance insufficiency; no source edit or serving approval.

### 14. Verification and remaining eight

- Canonical retirement-set contract re-executed: **11/11**, exact ordered ID equality across retirement table, embedded rollback UUIDs and pre-Round-O review table. No SQL statement executed.
- Selected identity checks: **3/3 PASS** for UUID, created_at, period vectors and original numeric matrices. The two Renewable→Renewable Energy abbreviations are explicitly accounted for; originals archived without modification. S3 prompt lengths agree with S1 strings, with the limitations stated above.
- Actual backend parser/preview adapter against the newly recovered original descriptions: **3/3 renderer_unsupported**, chart_data null, local serving false; original descriptions do not become supported just because context was found.
- Source integrity: **3/3 historical request/result pairs** matched by tool ID/project and six archived-line SHA-256 hashes verified. Git sanity: canonical first-observation commit resolved; equivalent retirement blobs matched; all-ref original-source searches recorded.
- `git diff --check`: PASS; prior document prefix preserved byte-for-byte; no application or migration diff. **No full pytest** for this investigation/docs-only change.

Do not begin content review of the remaining eight. Source recovery for these three demonstrated a useful search path, but it does not authorize bulk classification. Keep remaining review/serving states untouched. This round's outcome is **BLOCKED — original per-row SVG/unit evidence unavailable and C's category exclusivity unresolved**, with materially recovered prompts/base/source records committed. It is neither confirmed provenance loss nor a completed valid-for-serving review.

Production DB writes = 0; review PATCH = 0; migration apply = 0; deployment = 0; push = NO. Existing SQL-access, Render deployed-SHA and recoverability blockers are unchanged.
