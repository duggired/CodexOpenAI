/**
 * Phase 2b - Incident State Data Migration
 * ----------------------------------------
 * Run as: Fix Script or Background Script. Run Phase 2a (choice setup) FIRST
 * so target values 7/8 and the hold reasons are valid choices.
 *
 * Migrates historical incident records to the OOTB-aligned end state from the
 * one-pager: both state fields identical, six values only
 * (1 New, 2 In Progress, 3 On Hold, 6 Resolved, 7 Closed, 8 Canceled),
 * with the custom "Awaiting" states converted to On Hold + hold_reason.
 *
 * Mapping applied (from the one-pager):
 *   incident_state 3  "Awaiting Problem"    -> 3 On Hold + reason Awaiting Problem
 *   incident_state 4  "Awaiting User Info"  -> 3 On Hold + reason Awaiting Caller
 *   incident_state 5  "Awaiting Evidence"   -> 3 On Hold + reason Awaiting Evidence (custom)
 *   incident_state 9  "Awaiting Release"    -> 3 On Hold + reason Awaiting Change
 *   incident_state 10 "Assigned"            -> 2 In Progress
 *   state          3  "Closed Complete"     -> 7 Closed
 *   state          10 "Pending Approval"    -> CONFIG.pendingApprovalTarget (decide before apply!)
 *   state          11 "On Hold"             -> 3 On Hold
 *   state          12 "Cancelled"           -> 8 Canceled
 *   1 / 2 / 6 keep their value (label-only changes, no data movement).
 *
 * When the two fields disagree on the target for a record,
 * CONFIG.sourceOfTruth decides and the record is listed in the conflict
 * sample for review.
 *
 * Safety:
 *   - dryRun defaults to TRUE: full simulation - transition counts, conflict
 *     samples, verification - with zero writes. Set dryRun:false to apply.
 *   - runEngines:false (default) uses setWorkflow(false): no business rules,
 *     notifications, or SLA engine fire during the mass update.
 *   - preserveSysFields:true (default) uses autoSysFields(false): sys_updated_*
 *     stay intact so audit history is not flattened to the migration run.
 *   - Processes in sys_id-ordered batches; safe to stop and re-run (idempotent:
 *     already-aligned records fall out of the candidate query).
 *   - BACK UP FIRST: run on a clone, and export the candidate set (or take an
 *     instance backup) before apply mode in production.
 */

(function () {
    'use strict';

    var CONFIG = {
        dryRun: true,
        batchSize: 500,
        maxRecords: 0,               // 0 = no limit; set e.g. 1000 for a pilot batch
        // Which field wins when both map to different targets for one record.
        sourceOfTruth: 'incident_state',   // 'incident_state' or 'state'
        // state=10 "Pending Approval" - the one-pager only says "mapped during
        // migration". DECIDE with the business before apply mode:
        //   '2' = In Progress, or '3' = On Hold (+ defaultHoldReason).
        pendingApprovalTarget: '2',
        // hold_reason values (must exist - Phase 2a creates them):
        holdReasons: {
            awaitingCaller: '1',
            awaitingProblem: '3',
            awaitingChange: '5',
            awaitingEvidence: '6'    // custom value chosen in Phase 2a
        },
        // Used when target is On Hold but no reason is derivable (e.g. old
        // state=11 records). '' = leave empty and count for manual review.
        defaultHoldReason: '',
        runEngines: false,
        preserveSysFields: true,
        conflictSampleLimit: 50
    };

    var VALID_TARGETS = { '1': true, '2': true, '3': true, '6': true, '7': true, '8': true };

    // value -> { state: target, hold: hold_reason } per source field
    var INCIDENT_STATE_MAP = {
        '3':  { state: '3', hold: CONFIG.holdReasons.awaitingProblem },
        '4':  { state: '3', hold: CONFIG.holdReasons.awaitingCaller },
        '5':  { state: '3', hold: CONFIG.holdReasons.awaitingEvidence },
        '9':  { state: '3', hold: CONFIG.holdReasons.awaitingChange },
        '10': { state: '2' }
    };
    var STATE_MAP = {
        '3':  { state: '7' },
        '10': { state: CONFIG.pendingApprovalTarget },
        '11': { state: '3' },
        '12': { state: '8' }
    };

    function candidateFrom(map, value) {
        if (value === null || value === '')
            return null;
        if (map.hasOwnProperty(value))
            return map[value];
        if (VALID_TARGETS[value])
            return { state: value };
        return null; // unknown value - flagged as unmappable
    }

    // ------------------------------------------------------------------
    // Counters / samples
    // ------------------------------------------------------------------
    var stats = {
        scanned: 0, updated: 0, skippedAligned: 0, unmappable: 0,
        conflicts: 0, closedNoTimestamp: 0, onHoldNoReason: 0
    };
    var transitions = {};   // "is=4,st=2 -> state=3,hold=1" : count
    var conflictSample = [];
    var unmappableSample = [];

    function bump(mapObj, key) {
        mapObj[key] = (mapObj[key] || 0) + 1;
    }

    // ------------------------------------------------------------------
    // Per-record decision
    // ------------------------------------------------------------------
    function decide(gr) {
        var isVal = gr.getValue('incident_state');
        var stVal = gr.getValue('state');
        var fromIs = candidateFrom(INCIDENT_STATE_MAP, isVal);
        var fromSt = candidateFrom(STATE_MAP, stVal);

        if (!fromIs && !fromSt) {
            if ((isVal === null || isVal === '') && (stVal === null || stVal === ''))
                return null; // both empty - nothing to migrate
            stats.unmappable++;
            if (unmappableSample.length < CONFIG.conflictSampleLimit)
                unmappableSample.push(gr.getValue('number') + ' (incident_state=' + isVal + ', state=' + stVal + ')');
            return null;
        }

        var chosen;
        if (fromIs && fromSt && fromIs.state !== fromSt.state) {
            stats.conflicts++;
            chosen = (CONFIG.sourceOfTruth === 'state') ? fromSt : fromIs;
            if (conflictSample.length < CONFIG.conflictSampleLimit)
                conflictSample.push(gr.getValue('number') + ': incident_state=' + isVal + ' says ' + fromIs.state +
                                    ', state=' + stVal + ' says ' + fromSt.state + ' -> using ' + chosen.state);
        } else {
            chosen = fromIs || fromSt;
            // merge hold reason if the other candidate carries one
            if (!chosen.hold && fromIs && fromIs.hold)
                chosen = fromIs;
        }

        var target = { state: chosen.state, hold: chosen.hold || null };

        // hold_reason handling: only relevant when landing On Hold; never
        // clobber a reason already on the record.
        if (target.state === '3') {
            var currentHold = gr.getValue('hold_reason');
            if (currentHold) {
                target.hold = null; // keep existing
            } else if (!target.hold) {
                if (CONFIG.defaultHoldReason)
                    target.hold = CONFIG.defaultHoldReason;
                else
                    stats.onHoldNoReason++;
            }
        } else {
            target.hold = null;
        }

        var needsUpdate = (stVal !== target.state) || (isVal !== target.state) || !!target.hold;
        if (!needsUpdate) {
            stats.skippedAligned++;
            return null;
        }
        return target;
    }

    // ------------------------------------------------------------------
    // Migration loop - sys_id-windowed batches (safe against re-query loops)
    // ------------------------------------------------------------------
    var CANDIDATE_QUERY = 'incident_stateIN3,4,5,9,10' +
                          '^ORstateIN3,10,11,12' +
                          '^ORstateNSAMEASincident_state' +
                          '^ORincident_stateISEMPTY^stateISNOTEMPTY' +
                          '^ORstateISEMPTY^incident_stateISNOTEMPTY';

    function runMigration() {
        var lastSysId = '';
        var done = false;
        while (!done) {
            var gr = new GlideRecord('incident');
            gr.addEncodedQuery(CANDIDATE_QUERY);
            if (lastSysId)
                gr.addQuery('sys_id', '>', lastSysId);
            gr.orderBy('sys_id');
            gr.setLimit(CONFIG.batchSize);
            gr.query();
            var batchCount = 0;
            while (gr.next()) {
                batchCount++;
                lastSysId = gr.getUniqueValue();
                stats.scanned++;
                if (CONFIG.maxRecords && stats.scanned > CONFIG.maxRecords) {
                    done = true;
                    break;
                }
                var target = decide(gr);
                if (!target)
                    continue;

                var key = 'is=' + (gr.getValue('incident_state') || '(empty)') +
                          ',st=' + (gr.getValue('state') || '(empty)') +
                          ' -> state=' + target.state + (target.hold ? ',hold=' + target.hold : '');
                bump(transitions, key);

                var closing = (target.state === '7' || target.state === '8');
                if (target.state === '7' && !gr.getValue('closed_at'))
                    stats.closedNoTimestamp++;

                stats.updated++;
                if (!CONFIG.dryRun) {
                    gr.setValue('state', target.state);
                    gr.setValue('incident_state', target.state);
                    if (target.hold)
                        gr.setValue('hold_reason', target.hold);
                    if (closing)
                        gr.setValue('active', false);
                    if (!CONFIG.runEngines)
                        gr.setWorkflow(false);
                    if (CONFIG.preserveSysFields)
                        gr.autoSysFields(false);
                    gr.update();
                }
            }
            if (batchCount < CONFIG.batchSize)
                done = true;
        }
    }

    // ------------------------------------------------------------------
    // Verification - target: zero rows outside the six values, zero mismatches
    // ------------------------------------------------------------------
    function verifyCount(label, encodedQuery) {
        var ga = new GlideAggregate('incident');
        ga.addEncodedQuery(encodedQuery);
        ga.addAggregate('COUNT');
        ga.query();
        var n = ga.next() ? parseInt(ga.getAggregate('COUNT'), 10) : 0;
        gs.info('  ' + label + ': ' + n + (n === 0 ? '  [OK]' : '  [NEEDS ATTENTION]'));
        return n;
    }

    function verify(title) {
        gs.info('--- ' + title + ' ---');
        verifyCount('records where state != incident_state', 'stateNSAMEASincident_state');
        verifyCount('state outside 1,2,3,6,7,8', 'stateNOT IN1,2,3,6,7,8^stateISNOTEMPTY');
        verifyCount('incident_state outside 1,2,3,6,7,8', 'incident_stateNOT IN1,2,3,6,7,8^incident_stateISNOTEMPTY');
        verifyCount('On Hold without hold_reason', 'state=3^hold_reasonISEMPTY');
    }

    function distribution(title) {
        gs.info('--- ' + title + ' (state / incident_state / count) ---');
        var ga = new GlideAggregate('incident');
        ga.addAggregate('COUNT');
        ga.groupBy('state');
        ga.groupBy('incident_state');
        ga.query();
        while (ga.next())
            gs.info('  state=' + (ga.getValue('state') || '(empty)') +
                    ' | incident_state=' + (ga.getValue('incident_state') || '(empty)') +
                    ' | ' + ga.getAggregate('COUNT'));
    }

    // ------------------------------------------------------------------
    // Run
    // ------------------------------------------------------------------
    gs.info('=== Phase 2b: incident state data migration (' +
            (CONFIG.dryRun ? 'DRY RUN - no writes' : 'APPLY MODE - WRITING RECORDS') + ') ===');
    distribution('Distribution BEFORE');

    runMigration();

    gs.info('--- Result ---');
    gs.info('  scanned: ' + stats.scanned + ' | ' + (CONFIG.dryRun ? 'would update' : 'updated') + ': ' + stats.updated +
            ' | already aligned: ' + stats.skippedAligned);
    gs.info('  conflicts (fields disagreed, ' + CONFIG.sourceOfTruth + ' won): ' + stats.conflicts +
            ' | unmappable values: ' + stats.unmappable);
    gs.info('  landing On Hold without derivable reason: ' + stats.onHoldNoReason +
            ' | landing Closed without closed_at: ' + stats.closedNoTimestamp);

    gs.info('--- Transition counts ---');
    for (var t in transitions)
        gs.info('  ' + t + ' : ' + transitions[t]);

    if (conflictSample.length) {
        gs.info('--- Conflict sample (first ' + CONFIG.conflictSampleLimit + ') ---');
        for (var c = 0; c < conflictSample.length; c++)
            gs.info('  ' + conflictSample[c]);
    }
    if (unmappableSample.length) {
        gs.info('--- Unmappable sample (first ' + CONFIG.conflictSampleLimit + ') ---');
        for (var u = 0; u < unmappableSample.length; u++)
            gs.info('  ' + unmappableSample[u]);
    }

    if (CONFIG.dryRun) {
        verify('Verification (current state, pre-migration)');
        gs.info('Dry run complete. Review transition counts + conflicts, decide pendingApprovalTarget');
        gs.info('and defaultHoldReason, then set dryRun:false to apply (clone/backup first).');
    } else {
        distribution('Distribution AFTER');
        verify('Verification (post-migration; all counts should be 0)');
    }
})();
