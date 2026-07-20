/**
 * Phase 2a - Incident State Choice Alignment
 * ------------------------------------------
 * Run as: Fix Script or Background Script, inside an UPDATE SET.
 *
 * Aligns the choice lists of incident.state and incident.incident_state to
 * the OOTB six-value standard from the alignment one-pager:
 *   1 New - 2 In Progress - 3 On Hold - 6 Resolved - 7 Closed - 8 Canceled
 * and ensures the On Hold reason (hold_reason) choices exist, including the
 * custom "Awaiting Evidence" reason if still required.
 *
 * Behavior:
 *   - Idempotent: safe to re-run; it only inserts/updates what differs.
 *   - Old custom choices are marked INACTIVE, never deleted, so historical
 *     records still display until Phase 2b migrates them, and the change is
 *     reversible (reactivate the choice).
 *   - dryRun defaults to TRUE: it prints every action it WOULD take.
 *     Set dryRun:false to apply.
 *
 * Run order: this script must run BEFORE the data migration (Phase 2b) so
 * the target values 7/8 and the hold reasons are valid when records move.
 */

(function () {
    'use strict';

    var CONFIG = {
        dryRun: true,
        table: 'incident',
        elements: ['state', 'incident_state'],
        language: 'en',
        // Custom On Hold reason for the retired "Awaiting Evidence" state.
        // Set includeAwaitingEvidence:false if the business retires it fully.
        includeAwaitingEvidence: true,
        awaitingEvidenceValue: '6',
        awaitingEvidenceLabel: 'Awaiting Evidence'
    };

    // Target choice list - identical for both state fields (one-pager end state).
    var TARGET_STATE_CHOICES = [
        { value: '1', label: 'New',         sequence: 1 },
        { value: '2', label: 'In Progress', sequence: 2 },
        { value: '3', label: 'On Hold',     sequence: 3 },
        { value: '6', label: 'Resolved',    sequence: 6 },
        { value: '7', label: 'Closed',      sequence: 7 },
        { value: '8', label: 'Canceled',    sequence: 8 }
    ];

    // OOTB hold_reason choices the migration depends on (verified/created).
    var TARGET_HOLD_REASONS = [
        { value: '1', label: 'Awaiting Caller',  sequence: 1 },
        { value: '3', label: 'Awaiting Problem', sequence: 3 },
        { value: '4', label: 'Awaiting Vendor',  sequence: 4 },
        { value: '5', label: 'Awaiting Change',  sequence: 5 }
    ];

    var actions = [];

    function log(action) {
        actions.push(action);
        gs.info((CONFIG.dryRun ? '[DRY RUN] ' : '') + action);
    }

    function ensureChoice(element, value, label, sequence) {
        var c = new GlideRecord('sys_choice');
        c.addQuery('name', CONFIG.table);
        c.addQuery('element', element);
        c.addQuery('value', value);
        c.addQuery('language', CONFIG.language);
        c.query();
        if (c.next()) {
            var changes = [];
            if (c.getValue('label') !== label) changes.push('label: "' + c.getValue('label') + '" -> "' + label + '"');
            if (c.getValue('inactive') === 'true') changes.push('inactive: true -> false');
            if (c.getValue('sequence') !== String(sequence)) changes.push('sequence -> ' + sequence);
            if (!changes.length)
                return;
            log('UPDATE ' + CONFIG.table + '.' + element + ' value=' + value + ' (' + changes.join(', ') + ')');
            if (!CONFIG.dryRun) {
                c.setValue('label', label);
                c.setValue('inactive', false);
                c.setValue('sequence', sequence);
                c.update();
            }
        } else {
            log('INSERT ' + CONFIG.table + '.' + element + ' value=' + value + ' label="' + label + '"');
            if (!CONFIG.dryRun) {
                c.initialize();
                c.setValue('name', CONFIG.table);
                c.setValue('element', element);
                c.setValue('value', value);
                c.setValue('label', label);
                c.setValue('sequence', sequence);
                c.setValue('language', CONFIG.language);
                c.setValue('inactive', false);
                c.insert();
            }
        }
    }

    // Mark choices outside the target list inactive (never delete - records
    // still hold these values until Phase 2b runs, and it keeps rollback easy).
    function retireOthers(element, keepValues) {
        var c = new GlideRecord('sys_choice');
        c.addQuery('name', CONFIG.table);
        c.addQuery('element', element);
        c.addQuery('language', CONFIG.language);
        c.addQuery('value', 'NOT IN', keepValues.join(','));
        c.addQuery('inactive', false);
        c.query();
        while (c.next()) {
            log('RETIRE (inactive=true) ' + CONFIG.table + '.' + element +
                ' value=' + c.getValue('value') + ' label="' + c.getValue('label') + '"');
            if (!CONFIG.dryRun) {
                c.setValue('inactive', true);
                c.update();
            }
        }
    }

    // ------------------------------------------------------------------
    // Run
    // ------------------------------------------------------------------
    gs.info('=== Phase 2a: choice alignment for ' + CONFIG.table + ' (' +
            (CONFIG.dryRun ? 'DRY RUN - no changes applied' : 'APPLY MODE') + ') ===');

    var keep = [];
    for (var k = 0; k < TARGET_STATE_CHOICES.length; k++)
        keep.push(TARGET_STATE_CHOICES[k].value);

    for (var e = 0; e < CONFIG.elements.length; e++) {
        var el = CONFIG.elements[e];
        for (var i = 0; i < TARGET_STATE_CHOICES.length; i++)
            ensureChoice(el, TARGET_STATE_CHOICES[i].value, TARGET_STATE_CHOICES[i].label, TARGET_STATE_CHOICES[i].sequence);
        retireOthers(el, keep);
    }

    var reasons = TARGET_HOLD_REASONS.slice();
    if (CONFIG.includeAwaitingEvidence)
        reasons.push({ value: CONFIG.awaitingEvidenceValue, label: CONFIG.awaitingEvidenceLabel, sequence: 9 });
    var keepReasons = [];
    for (var r = 0; r < reasons.length; r++) {
        ensureChoice('hold_reason', reasons[r].value, reasons[r].label, reasons[r].sequence);
        keepReasons.push(reasons[r].value);
    }
    // Intentionally NOT retiring other hold_reason values - only ensure ours exist.

    gs.info('=== Done: ' + actions.length + ' action(s)' +
            (CONFIG.dryRun ? ' would be applied. Set dryRun:false inside an update set to apply.' : ' applied.') + ' ===');
    if (!CONFIG.dryRun && actions.length)
        gs.info('Reminder: verify these sys_choice changes were captured in your current update set.');
    if (!actions.length)
        gs.info('Choice lists already match the target - nothing to do.');
})();
