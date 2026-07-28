# Incident State Impact Discovery

Companion to the **"Incident State Field Alignment — Current vs. OOTB vs.
Future"** one-pager. Both incident state fields (`state` and the legacy
`incident_state`) are being aligned back to the OOTB six-value standard —
**1 New · 2 In Progress · 3 On Hold · 6 Resolved · 7 Closed · 8 Canceled** —
with the custom "Awaiting" states preserved as **On Hold reasons**
(`hold_reason`).

Before developers update artifacts to the aligned values, they need the list
of **every object referencing those fields or the customized values/labels** —
without verifying each artifact manually.

Two discovery scripts produce that list — both **read-only ServiceNow
background scripts** (they never insert/update anything):

- **`incident-state-impact-discovery-strict.js` — the primary worklist.**
  Only reports objects with evidence they are about the incident table:
  table-scoped artifacts must sit on `incident` exactly; unscoped artifacts
  (script includes, widgets, UX scripts, jobs...) must also contain incident
  evidence in their code — `new GlideRecord('incident')` /
  `GlideRecordSecure` / `GlideAggregate` on incident, `table: 'incident'`, a
  quoted `'incident'` literal, or the incident-only fields `incident_state`
  / `hold_reason`. Results are limited to the **Global** application scope
  (`globalScopeOnly` / `applicationScopes` in CONFIG), task-level artifacts
  are excluded by default (`includeTaskLevel: false`), and the report shows
  how many state matches were excluded as noise.
- **`incident-state-impact-discovery.js` — the completeness cross-check.**
  Broad sweep: any state-field reference, `task`-level artifacts included,
  all application scopes. Expect noise (state fields on other tables match
  too); use it to catch what strict mode's filters can hide — e.g. a
  task-level business rule checking `current.state` (it does run for
  incidents), or a generic widget whose table name arrives via an instance
  option so `'incident'` never appears in its code.

Both share the same value map, HIGH/MEDIUM/REVIEW priorities, and CSV
format, so the two worklists are directly comparable.

### `incident-hold-reason-impact-discovery.js` — On Hold / hold reason only

A third, narrowly focused discovery script for the **On Hold** transition
alone. Under the alignment the custom "Awaiting" `incident_state` values
collapse into a single state (3 On Hold) and are distinguished by the
**On Hold Reason** (`hold_reason`) field. This script reports only objects
referencing those old waiting values or the `hold_reason` field — it ignores
plain New/In Progress/Resolved/Closed references — and prints each hit as a
direct **old → new** instruction.

Old → new map applied (from the handout's "What happens" column):

| Old reference | New |
|---|---|
| `incident_state=3` "Awaiting Problem" | On Hold (3) + reason **Awaiting Problem** |
| `incident_state=4` "Awaiting User Info" | On Hold (3) + reason **Awaiting Caller** |
| `incident_state=5` "Awaiting Evidence" | On Hold (3) + reason **Awaiting Evidence** *(custom, `keepAwaitingEvidence`)* |
| `incident_state=9` "Awaiting Release" | On Hold (3) + reason **Awaiting Change** |
| `state=11` "On Hold" (legacy) | On Hold (3) + reason *(manual — none derivable)* |

Report specifics:

- **HIGH** = the object references an old value that must be remapped;
  **REVIEW** = it touches `hold_reason`, or sets On Hold without setting a
  reason (those need a reason added post-alignment — counted separately).
- Each row shows `old → new` inline, and the CSV has a dedicated
  `old_to_new` column plus an `onhold_no_reason` flag.
- SLA definitions are covered first-class because On Hold reasons drive SLA
  **pause** behavior — the highest-value thing to verify after remapping.
- Same incident-evidence filter and Global-scope default as strict mode.
- Flip `keepAwaitingEvidence: false` in CONFIG to treat Awaiting Evidence as
  retired instead of a custom reason.

## What's pre-configured from the one-pager

The `CONFIG.customChoiceValues` block already encodes the alignment map:

- **Retired (HIGH priority — references break outright):** values
  `4, 5, 9, 10, 11, 12` and labels *Awaiting Problem, Awaiting User Info,
  Awaiting Evidence, Awaiting Release, Assigned, Active, Open,
  Work in Progress, Closed Complete, Pending Approval, Cancelled*.
- **Changed / restored (MEDIUM priority — logic may silently do the wrong
  thing):** values `1, 2, 3` (survive but change label — and value 3 means
  three different things today), plus `7, 8` (restored; absorb today's
  Closed Complete / Cancelled records) and the label *On Hold*.
- Everything else that touches `state` / `incident_state` / `hold_reason`
  without a recognized value nearby is reported as **REVIEW** (functional
  check needed).

`hold_reason` is scanned alongside the state fields because the Awaiting
states move there — anything already touching it is part of the same
migration.

## How to run

1. Open **System Definition > Scripts - Background** on a **sub-prod instance
   first** (elevate to `security_admin` if ACL scanning is restricted).
2. Review the `CONFIG` block (the value map is pre-filled; adjust only if the
   one-pager mapping changes). Set `scanFlowSnapshots: true` to also scan
   Flow Designer flows (heavier query — sub-prod recommended).
3. Run it. The output contains three sections:
   - **Current choices** — what `sys_choice` holds *now* for both state
     fields plus `hold_reason`, with value, label, inactive flag, and
     who/when last updated. Diff this against the one-pager's target column.
   - **Impact report** — impacted objects grouped by Server / Client /
     Portal / Workspace / Reporting, sorted HIGH → MEDIUM → REVIEW, each with
     the matched field, the retired/changed values found, and a record link.
   - **CSV export** — copy the block into a spreadsheet and use it as the
     developer worklist (priority column first).

## What it scans

| Category  | Artifacts |
|-----------|-----------|
| Server    | Business Rules, Script Includes, Fix Scripts, Scheduled Jobs, Script Actions, Scripted REST (inbound), REST Message Methods (outbound), Notifications, Email Templates, SLA Definitions, ACLs, Transform Maps, Workflow Activities, Flow Designer flows (optional), Metric Definitions, Data Policy Rules, Dictionary Overrides |
| Client    | Client Scripts (incl. catalog), UI Policies + UI Policy Actions, UI Actions |
| Portal    | Service Portal Widgets (server/client/HTML/link/demo data), Widget Instances (filters + options) |
| Workspace | UX Client Scripts, UX Transform Data Brokers, UI Builder Macroponents, Declarative Actions |
| Reporting | Reports, Saved Filters |

Scoped artifacts are limited to `incident` + `task` (task-level logic also
fires for incidents). Tables that don't exist on your ServiceNow version are
skipped and listed at the end of the report, not errored.

## How matching works

- An artifact is reported only when its script/condition genuinely references
  a field: quoted field name (`g_form.getValue('state')`), dot-walk
  (`current.state`, `current.hold_reason`), or condition/encoded-query syntax
  (`state=4`, `incident_stateIN4,5,9`). A stray word "state" in a comment
  does not match.
- Retired/changed values and labels are searched **within 80 characters of a
  state/hold_reason reference** (configurable via `proximityChars`), and the
  worst match wins the priority: any retired token → HIGH, else any changed
  token → MEDIUM, else REVIEW.

---

# Phase 2 — Executing the alignment

Two additional scripts carry out the migration itself. Both default to
`dryRun: true` (full simulation, zero writes) — flip to `false` only after
reviewing the dry-run output, and always rehearse on a clone first.

## Phase 2a — `incident-state-choice-setup.js` (choice alignment)

Run **inside an update set**, before any data moves. It:

- Inserts/updates the six target choices on **both** `state` and
  `incident_state` (1 New, 2 In Progress, 3 On Hold, 6 Resolved, 7 Closed,
  8 Canceled).
- Marks every other choice on those fields **inactive** — never deleted, so
  historical records still display until Phase 2b migrates them and the
  change stays reversible.
- Ensures the `hold_reason` choices exist: OOTB Awaiting Caller / Problem /
  Vendor / Change, plus the custom **Awaiting Evidence** (value `6` by
  default — change `awaitingEvidenceValue` if that collides on your
  instance).
- Idempotent: re-running it reports "nothing to do".

## Phase 2b — `incident-state-data-migration.js` (record migration)

Migrates historical records to the end state (both fields identical, six
values only) using the one-pager mapping, including Awaiting states → On
Hold + `hold_reason`, Closed Complete (3) → Closed (7), Cancelled (12) →
Canceled (8), Assigned (10) → In Progress (2).

Decisions to make **before** apply mode (both in `CONFIG`):

- `pendingApprovalTarget` — the one-pager only says state 10 "Pending
  Approval" is *"mapped during migration"*. Default is `'2'` (In Progress);
  confirm with the business.
- `defaultHoldReason` — used for records landing On Hold with no derivable
  reason (e.g. old state 11); default leaves it empty and counts them for
  manual review.
- `sourceOfTruth` — when the two fields disagree on a record's target,
  `incident_state` wins by default (it carries the richer waiting
  semantics); every conflict is sampled in the output for review.

Safety characteristics:

- `runEngines: false` → `setWorkflow(false)`: no business rules,
  notifications, or SLA engine fire during the mass update. Consequence:
  SLA stage/pause state does **not** recalculate — review active SLAs on
  migrated records as a follow-up.
- `preserveSysFields: true` → `autoSysFields(false)`: audit fields aren't
  flattened to the migration date.
- Records landing Closed/Canceled get `active=false`; records reaching
  Closed without a `closed_at` are counted for review (timestamps are never
  fabricated).
- Existing `hold_reason` values are never overwritten.
- Processes in `sys_id`-ordered batches — safe to stop and re-run; use
  `maxRecords` for a pilot batch.
- Dry-run output includes before-distribution, per-transition counts,
  conflict/unmappable samples, and a verification block; apply mode re-runs
  the verification (all counts should be 0).

## Recommended release sequence

1. **Discovery** (`incident-state-impact-discovery.js`) → CSV worklist.
2. **Phase 2a** choice alignment (update set, dry run → apply).
3. **Artifact remediation** from the worklist: HIGH first, then MEDIUM
   (especially value 3 semantics), then REVIEW — reports, SLAs,
   notifications, and integrations in the **same release**, per the
   one-pager.
4. **Phase 2b** data migration (clone rehearsal → dry run in prod → apply).
5. **Verify**: re-run discovery (HIGH should be zero) and Phase 2b's
   verification block (all counts zero); confirm the OOTB state sync
   between the two fields behaves natively again.
