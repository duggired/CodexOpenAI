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

`incident-state-impact-discovery.js` produces that list. It is a **read-only
ServiceNow background script** (it never inserts/updates anything).

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

## After discovery — executing the alignment

1. Open an **update set** so the alignment work is captured and movable
   across instances.
2. Work the CSV worklist top-down: **HIGH** rows reference retired values and
   break on day one; **MEDIUM** rows reference values whose meaning or label
   changes (especially value 3) and need review; **REVIEW** rows need a
   functional check.
3. Remember the one-pager's data-migration caveat: this script finds the
   *artifacts*; the historical **record migration** (value 3's three
   meanings, Closed Complete → Closed 7, Cancelled 12 → Canceled 8, Awaiting
   states → On Hold + `hold_reason`) is a separate, higher-risk workstream —
   reports, SLAs, notifications, and integrations must move in the same
   release.
4. Re-run the script afterwards — HIGH should reach zero, and the "current
   choices" section should match the six-value OOTB target in both fields.
   It doubles as verification evidence.
