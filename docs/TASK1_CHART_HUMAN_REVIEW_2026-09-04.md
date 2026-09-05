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
