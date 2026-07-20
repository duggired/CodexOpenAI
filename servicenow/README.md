# Incident State Impact Discovery

The choice list for the Incident **State** field (`incident.state` / legacy
`incident_state`) was reverted to out-of-the-box (OOTB), losing the customized
choices in use. Before a developer re-applies the customizations, they need to
know **every object that references the state field or its choice values** —
without verifying each artifact manually.

`incident-state-impact-discovery.js` produces that list. It is a **read-only
ServiceNow background script** (it never inserts/updates anything).

## How to run

1. Open **System Definition > Scripts - Background** on a **sub-prod instance
   first** (elevate to `security_admin` if ACL scanning is restricted).
2. Paste the script and edit the `CONFIG` block at the top:
   - `customChoiceValues`: the custom values **and** labels you had before the
     revert, e.g. `['10', '12', 'Awaiting Vendor']`. Leaving it empty still
     reports every state-field reference, just without the per-value match
     column.
   - `scanFlowSnapshots`: set `true` to also scan Flow Designer flows
     (heavier query — recommended on sub-prod only).
3. Run it. The output contains three sections:
   - **Current choices** — what `sys_choice` holds *now* (post-revert), with
     value, label, inactive flag, and who/when last updated. Diff this against
     your customized list to see exactly which choices must be re-added.
   - **Impact report** — impacted objects grouped by Server / Client / Portal /
     Workspace / Reporting, each with the matched field, any custom values
     found near the reference, and a direct link.
   - **CSV export** — copy the block into a spreadsheet and use it as the
     developer worklist (one row per impacted object).

## What it scans

| Category  | Artifacts |
|-----------|-----------|
| Server    | Business Rules, Script Includes, Fix Scripts, Scheduled Jobs, Script Actions, Scripted REST, Notifications, SLA Definitions, ACLs, Transform Maps, Workflow Activities, Flow Designer flows (optional), Metric Definitions, Dictionary Overrides |
| Client    | Client Scripts (incl. catalog), UI Policies + UI Policy Actions, UI Actions |
| Portal    | Service Portal Widgets (server/client/HTML/link/demo data), Widget Instances (filters + options) |
| Workspace | UX Client Scripts, UX Transform Data Brokers, UI Builder Macroponents, Declarative Actions |
| Reporting | Reports, Saved Filters |

Scoped artifacts are limited to `incident` + `task` (task-level logic also
fires for incidents). Tables that don't exist on your ServiceNow version are
skipped and listed at the end of the report, not errored.

## How matching works

- An artifact is reported only when its script/condition genuinely references
  the field: quoted field name (`g_form.getValue('state')`), dot-walk
  (`current.state`), or condition/encoded-query syntax (`state=6`,
  `stateIN1,2`). A stray word "state" in a comment does not match.
- If `customChoiceValues` is set, each value/label is searched **within 80
  characters of a state reference** (configurable via `proximityChars`), so
  the report flags which objects hard-code your custom values — those are the
  highest-priority fixes.

## After discovery — re-applying the customizations

1. Open an **update set** before touching anything so the re-customization is
   captured and movable across instances.
2. Re-add the missing choices on `incident.state` (right-click the State field
   > Configure Choices, or insert `sys_choice` records with the original
   value/label/sequence). Keep the original numeric values — the impact report
   shows which artifacts hard-code them.
3. Work through the CSV worklist top-down: rows with a non-empty
   `custom_values_found` column reference the reverted values directly and
   break first; field-reference-only rows need a functional check.
4. Re-run the script afterwards — the "current choices" section should now
   match your customized list, and it doubles as verification evidence.
