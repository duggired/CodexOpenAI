# Runbook — Incident State Alignment to OOTB + Major Incident Management Enablement

**Audience:** Production Support / Operations
**Change type:** Standard release — data model change + plugin enablement
**Risk:** High (historical data migration + two-field state change)
**Owner:** Platform / ServiceNow Development team
**Status:** Template — fill in bracketed `[…]` values at release planning.

---

## 1. What is changing and why

The Incident form carries two state fields (`state` and `incident_state`) that
ServiceNow keeps in sync out of the box. Years of divergent customization broke
that sync — the root cause of four active production defects — and blocked
adoption of Major Incident Management (MIM), which assumes the standard model.

This release:

1. **Realigns both state fields to the six OOTB values** — 1 New, 2 In Progress,
   3 On Hold, 6 Resolved, 7 Closed, 8 Canceled — identical in both fields.
2. **Converts the custom "Awaiting" states into On Hold reasons** (`hold_reason`)
   — the platform-standard pattern.
3. **Migrates historical incident records** to the aligned values.
4. **Enables Major Incident Management** on the clean foundation.

After this change, the native state sync works again and MIM operates as
designed with no custom retrofit.

---

## 2. Business impact — what users will notice

- **State labels change** on the Incident form and lists (e.g. "Open" → "New",
  "Active"/"Work in Progress" → "In Progress", custom "Awaiting …" states become
  **On Hold + a reason**). No functional loss — the waiting situations are now
  captured as On Hold reasons.
- **Historical incidents are re-stated** to the new values (migrated, not just
  relabeled). Old reports/filters referencing retired values are updated in this
  release.
- **New capability: Major Incident Management** — major incident promotion, the
  major incident workbench, communications, and post-incident review become
  available to authorized groups.
- **No outage expected** — this is a configuration + data change, not an
  infrastructure change.

---

## 3. Release window

| Item | Value |
|---|---|
| Planned window | `[date / time / TZ]` |
| Expected duration | `[e.g. 3–4 hrs incl. verification]` |
| Change record | `[CHG#######]` |
| Freeze | No new state-field or hold_reason customizations from `[T-minus date]` until close |
| Rollback deadline | `[go/no-go time]` |

---

## 4. Pre-checks (before the window opens)

- [ ] Latest **backup / clone** confirmed and restorable (`[timestamp]`).
- [ ] All release **update sets** completed on sub-prod and **rehearsed on a
      clone** with zero verification errors.
- [ ] Discovery re-run on clone: **HIGH count = 0** (no unremediated references
      to retired values).
- [ ] Data-migration script **dry-run on clone**: transition counts,
      conflicts, and unmappable records reviewed and signed off.
- [ ] Test evidence pack signed off (lifecycle, SLAs, notifications, reports,
      integrations, MIM end-to-end).
- [ ] Integrations owners notified `[list: e.g. monitoring, CMDB, ITSM peers]`.
- [ ] Two business decisions confirmed and recorded:
      **Pending Approval** target and **Awaiting Evidence** kept/retired.
- [ ] Go/no-go sign-off from `[change owner]`.

---

## 5. Deployment steps (high level)

> Detailed step-by-step and update-set names live in the deployment guide;
> this is the operational sequence.

1. **Open the change / enable a maintenance banner** if used.
2. **Commit choice-alignment update set** — restores the six OOTB state choices
   on both fields, sets up On Hold reasons, retires (does not delete) old
   choices.
3. **Commit remediation update set(s)** — the remediated business rules, client
   scripts, UI policies, SLA conditions, notifications, reports, and
   integrations.
4. **Run the data-migration script (apply mode)** — migrates historical records
   per the mapping; runs with engines off so no mass notifications/SLA churn.
5. **Enable the Major Incident Management plugin** and commit MIM configuration
   (promotion criteria, communication templates, workbench).
6. **Run the verification block** — all mismatch counts must return **zero**.
7. **Smoke test** (Section 7), then **close the window**.

---

## 6. Rollback plan

Trigger rollback if verification fails, a P1 regression appears, or the go/no-go
deadline passes without success.

- **Choices:** old choices were retired (inactive), not deleted — reactivate to
  restore prior labels.
- **Artifacts:** back out the remediation and choice update sets (update sets are
  reversible).
- **Data:** restore incident records from the pre-migration backup/clone
  `[method + owner]`. *Migration is the least reversible step — the backup is the
  primary rollback for data.*
- **MIM:** plugin enablement can be left inactive/hidden; it does not force schema
  loss if backed out per `[plan]`.
- Record outcome on the change and notify stakeholders.

**Point of no return:** once the data migration is applied in production and the
window is closed, forward-fix is preferred over full rollback — escalate to
`[dev owner]` before attempting a data restore.

---

## 7. Post-deployment smoke test (support-runnable)

- [ ] Open an incident → confirm state choices show **only** New / In Progress /
      On Hold / Resolved / Closed / Canceled, identical on both fields.
- [ ] Set an incident to **On Hold** → confirm the **On Hold Reason** field
      appears and its choices are correct.
- [ ] Move an incident through the lifecycle (New → In Progress → On Hold →
      Resolved → Closed) → no script errors, correct SLA pause on hold.
- [ ] Confirm a key **notification** fires with correct state wording.
- [ ] Open a saved **report/dashboard** on incident state → renders with new
      values, no blanks.
- [ ] **MIM:** promote a test incident to major → workbench opens, comms task
      created, resolve/close works.
- [ ] Confirm key **integrations** `[list]` still send/receive state correctly.

---

## 8. First 48 hours — what to watch

- **Error logs** for script errors referencing `state`, `incident_state`, or
  `hold_reason` (`System Log → Errors`).
- **SLA behavior** — On Hold pause/resume is the most sensitive area; watch for
  SLAs not pausing/resuming as expected.
- **Integration queues** — failed outbound/inbound messages carrying state.
- **Report/dashboard tickets** — users reporting blank or wrong state buckets.
- **Any incident stuck** in a state that won't advance.

---

## 9. Known-sensitive areas (triage hints)

| Symptom | Likely cause | First action |
|---|---|---|
| Incident won't leave On Hold / no reason shown | hold_reason not set or a missed remediation | Check `hold_reason`; consult the On Hold old→new map |
| SLA not pausing on hold | SLA pause condition references an old value | Verify SLA definition condition; escalate to dev |
| Report shows blank/other state bucket | Report filter references retired value | Update filter to new value |
| Integration rejects a record | Payload maps an old state value | Check integration field map; escalate to dev |
| Notification wording wrong | Template references old label | Update template text |

For anything touching **data correctness or scripts**, do **not** hand-edit in
production — log a ticket and **escalate to the Platform/Dev team**.

---

## 10. Contacts & escalation

| Role | Name | Contact | When |
|---|---|---|---|
| Change owner | `[name]` | `[contact]` | Go/no-go, rollback decision |
| ServiceNow dev lead | `[name]` | `[contact]` | Script/data/SLA issues |
| MIM process owner | `[name]` | `[contact]` | Major incident process questions |
| Integration owner(s) | `[name]` | `[contact]` | Integration failures |
| On-call / bridge | `[name/line]` | `[contact]` | P1 during/after window |

---

## 11. Reference

- Alignment one-pager and options comparison — `[link / repo path]`
- Discovery scripts (impact worklist), choice-setup and data-migration scripts,
  On Hold old→new map — `servicenow/` in `[repo]`
- Old → new On Hold reason map:
  Awaiting Problem → On Hold + Awaiting Problem · Awaiting User Info → On Hold +
  Awaiting Caller · Awaiting Evidence → On Hold + Awaiting Evidence (custom) ·
  Awaiting Release → On Hold + Awaiting Change · legacy On Hold (11) → On Hold (3)
- End state: **1 New · 2 In Progress · 3 On Hold · 6 Resolved · 7 Closed ·
  8 Canceled**, identical in both fields.
