/**
 * Incident On Hold / Hold Reason Impact Discovery
 * -----------------------------------------------
 * Run as: Background Script. Read-only: it only queries, never updates.
 *
 * Narrow companion to the state discovery scripts, scoped ONLY to the
 * On Hold transition. Under the alignment (see the one-pager), the custom
 * "Awaiting" incident_state values collapse into a single state - 3 On Hold -
 * and are told apart by the On Hold Reason field (hold_reason):
 *
 *   OLD value (referenced by objects today)        NEW
 *   incident_state 3  "Awaiting Problem"     -> state 3 On Hold + reason Awaiting Problem
 *   incident_state 4  "Awaiting User Info"   -> state 3 On Hold + reason Awaiting Caller
 *   incident_state 5  "Awaiting Evidence"    -> state 3 On Hold + reason Awaiting Evidence (custom, if kept)
 *   incident_state 9  "Awaiting Release"     -> state 3 On Hold + reason Awaiting Change
 *   state          11 "On Hold" (legacy)     -> state 3 On Hold + (no reason derivable - manual)
 *
 * This script reports any object that references one of those OLD values /
 * labels, or the hold_reason field, so a developer gets a direct
 * "change this -> to that" worklist. It deliberately ignores plain
 * New / In Progress / Resolved / Closed references - those belong to the
 * general state discovery scripts.
 *
 * Business decisions baked in (change in CONFIG if they move):
 *   - "Awaiting User Info" -> hold reason "Awaiting Caller"
 *   - "Awaiting Release"   -> hold reason "Awaiting Change"
 *   - "Awaiting Evidence"  -> kept as custom hold reason (CONFIG.keepAwaitingEvidence)
 *
 * Report also flags objects that set state to On Hold (3 / "On Hold") but
 * never set a hold_reason - after alignment those need a reason added.
 */

(function () {
    'use strict';

    // ------------------------------------------------------------------
    // CONFIG
    // ------------------------------------------------------------------
    var CONFIG = {
        targetTable: 'incident',
        includeTaskLevel: false,
        globalScopeOnly: true,
        applicationScopes: ['global'],
        keepAwaitingEvidence: true,      // false = treat Awaiting Evidence as retired
        proximityChars: 80,
        scanFlowSnapshots: false,
        maxRowsPerTable: 2000
    };

    var tableScope = [CONFIG.targetTable];
    if (CONFIG.includeTaskLevel)
        tableScope.push('task');

    // ------------------------------------------------------------------
    // The old -> new On Hold map. Each entry is an OLD reference an object
    // might contain, and the NEW target it should be updated to.
    //   sourceField : which field the old value lives on
    //   oldValue    : the numeric value objects reference today
    //   oldLabels   : label strings objects might reference instead
    //   newHoldReason : target hold_reason label ('' = none derivable)
    // ------------------------------------------------------------------
    var HOLD_MAP = [
        { sourceField: 'incident_state', oldValue: '3', oldLabels: ['Awaiting Problem'],   newHoldReason: 'Awaiting Problem' },
        { sourceField: 'incident_state', oldValue: '4', oldLabels: ['Awaiting User Info'], newHoldReason: 'Awaiting Caller' },
        { sourceField: 'incident_state', oldValue: '5', oldLabels: ['Awaiting Evidence'],  newHoldReason: 'Awaiting Evidence', custom: true },
        { sourceField: 'incident_state', oldValue: '9', oldLabels: ['Awaiting Release'],   newHoldReason: 'Awaiting Change' },
        { sourceField: 'state',          oldValue: '11', oldLabels: ['On Hold'],           newHoldReason: '' }
    ];

    // Build the active map (drop Awaiting Evidence if retired) and a lookup
    // of every old token -> its target description.
    var activeMap = [];
    for (var h = 0; h < HOLD_MAP.length; h++) {
        if (HOLD_MAP[h].custom && !CONFIG.keepAwaitingEvidence)
            continue;
        activeMap.push(HOLD_MAP[h]);
    }

    function targetOf(entry) {
        return 'state 3 On Hold' + (entry.newHoldReason ? ' + reason "' + entry.newHoldReason + '"'
                                                        : ' + reason (manual - none derivable)');
    }

    // ------------------------------------------------------------------
    // Matching helpers
    // ------------------------------------------------------------------
    function escapeRegex(s) {
        return String(s).replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    }

    // A reference to the hold_reason field itself (any object already
    // touching On Hold Reason is in scope).
    var HOLD_REASON_FIELD_PATTERNS = [
        /['"]hold_reason['"]/,
        /\.hold_reason\b/,
        /\bhold_reason\s*(=|!=|>=|<=|>|<|IN\b|NOT ?IN\b|CHANGES)/i
    ];

    // A reference to a state/incident_state field near an old waiting value.
    // Anchors we look around for value proximity.
    var STATE_ANCHORS = ['state', 'hold_reason'];

    // Per old map entry, a value-proximity regex (numeric or label).
    var valueMatchers = [];
    for (var m = 0; m < activeMap.length; m++) {
        var e = activeMap[m];
        var tokens = [e.oldValue].concat(e.oldLabels);
        for (var tk = 0; tk < tokens.length; tk++) {
            var isNum = /^[0-9]+$/.test(tokens[tk]);
            valueMatchers.push({
                entry: e,
                token: tokens[tk],
                numeric: isNum,
                re: new RegExp('(^|[^\\w])' + escapeRegex(tokens[tk]) + '([^\\w]|$)', isNum ? '' : 'i')
            });
        }
    }

    function hasHoldReasonField(text) {
        for (var i = 0; i < HOLD_REASON_FIELD_PATTERNS.length; i++)
            if (HOLD_REASON_FIELD_PATTERNS[i].test(text))
                return true;
        return false;
    }

    // Only count a numeric old value (e.g. "4", "9") when it sits close to a
    // state/hold_reason anchor - avoids matching stray numbers. Labels are
    // distinctive enough to match anywhere in the text.
    function nearAnchor(text, tokenRe) {
        for (var a = 0; a < STATE_ANCHORS.length; a++) {
            var idx = text.indexOf(STATE_ANCHORS[a]);
            while (idx !== -1) {
                var win = text.substring(Math.max(0, idx - CONFIG.proximityChars), idx + CONFIG.proximityChars);
                if (tokenRe.test(win))
                    return true;
                idx = text.indexOf(STATE_ANCHORS[a], idx + STATE_ANCHORS[a].length);
            }
        }
        return false;
    }

    /**
     * Returns null, or:
     * { mappings: [{old, target}], holdReasonRef: bool, onHoldNoReason: bool }
     */
    function analyze(text) {
        if (!text)
            return null;
        var mappings = [];
        var seenOld = {};
        for (var i = 0; i < valueMatchers.length; i++) {
            var vm = valueMatchers[i];
            var hit = vm.numeric ? nearAnchor(text, vm.re) : vm.re.test(text);
            if (!hit)
                continue;
            var oldKey = vm.entry.sourceField + '=' + vm.entry.oldValue;
            if (seenOld[oldKey])
                continue;
            seenOld[oldKey] = true;
            mappings.push({
                old: oldKey + ' "' + vm.entry.oldLabels[0] + '"',
                target: targetOf(vm.entry)
            });
        }
        var holdRef = hasHoldReasonField(text);

        // Object sets On Hold but no hold_reason nearby -> needs a reason.
        var onHoldNoReason = false;
        if (!holdRef) {
            var setsOnHold = /\b(incident_state|state)\s*[=:]{1,3}\s*['"]?3['"]?\b/.test(text) ||
                             /setValue\s*\(\s*['"](state|incident_state)['"]\s*,\s*['"]?3['"]?\s*\)/.test(text) ||
                             /\b(state|incident_state)\s*(=|IN)\s*3\b/.test(text);
            if (setsOnHold)
                onHoldNoReason = true;
        }

        if (!mappings.length && !holdRef && !onHoldNoReason)
            return null;
        return { mappings: mappings, holdReasonRef: holdRef, onHoldNoReason: onHoldNoReason };
    }

    // ------------------------------------------------------------------
    // Artifact definitions (same coverage as strict discovery).
    // requireEvidence: unscoped artifacts must also prove incident context.
    // ------------------------------------------------------------------
    var INCIDENT_EVIDENCE = [
        /GlideRecord(Secure)?\s*\(\s*['"]incident['"]/i,
        /GlideAggregate\s*\(\s*['"]incident['"]/i,
        /\btable\s*(:|===?|=)\s*['"]incident['"]/i,
        /['"]incident['"]/,
        /['"]incident\.[a-z_]+['"]/i,
        /\bincident_state\b/,
        /\bhold_reason\b/
    ];
    function hasIncidentEvidence(text) {
        if (!text) return false;
        for (var i = 0; i < INCIDENT_EVIDENCE.length; i++)
            if (INCIDENT_EVIDENCE[i].test(text)) return true;
        return false;
    }

    var TEXT_ARTIFACTS = [
        { category: 'Server',    table: 'sys_script',              label: 'Business Rule',          fields: ['script', 'filter_condition', 'condition'], scopeField: 'collection' },
        { category: 'Server',    table: 'sys_script_include',      label: 'Script Include',         fields: ['script'], requireEvidence: true },
        { category: 'Server',    table: 'sys_script_fix',          label: 'Fix Script',             fields: ['script'], requireEvidence: true },
        { category: 'Server',    table: 'sysauto_script',          label: 'Scheduled Script Job',   fields: ['script', 'condition'], requireEvidence: true },
        { category: 'Server',    table: 'sysevent_script_action',  label: 'Script Action',          fields: ['script'], requireEvidence: true },
        { category: 'Server',    table: 'sys_ws_operation',        label: 'Scripted REST Operation', fields: ['operation_script'], requireEvidence: true },
        { category: 'Server',    table: 'sys_rest_message_fn',     label: 'REST Message Method (outbound)', fields: ['content'], requireEvidence: true },
        { category: 'Server',    table: 'sysevent_email_action',   label: 'Notification',           fields: ['condition', 'advanced_condition', 'subject', 'message_html'], scopeField: 'collection' },
        { category: 'Server',    table: 'sysevent_email_template', label: 'Email Template',         fields: ['subject', 'message_html'], scopeField: 'collection' },
        // SLA definitions are the most sensitive: On Hold reasons drive pause.
        { category: 'Server',    table: 'contract_sla',            label: 'SLA Definition',         fields: ['start_condition', 'stop_condition', 'pause_condition', 'reset_condition'], scopeField: 'collection' },
        { category: 'Server',    table: 'sys_security_acl',        label: 'ACL',                    fields: ['condition', 'script'], extraQuery: 'nameSTARTSWITHincident' },
        { category: 'Server',    table: 'sys_transform_script',    label: 'Transform Map Script',   fields: ['script'], extraQuery: 'map.target_table=incident' },

        { category: 'Client',    table: 'sys_script_client',       label: 'Client Script',          fields: ['script'], scopeField: 'table' },
        { category: 'Client',    table: 'sys_ui_policy',           label: 'UI Policy',              fields: ['conditions', 'script_true', 'script_false'], scopeField: 'table' },
        { category: 'Client',    table: 'sys_ui_action',           label: 'UI Action',              fields: ['script', 'condition'], scopeField: 'table' },

        { category: 'Portal',    table: 'sp_widget',               label: 'SP Widget',              fields: ['script', 'client_script', 'template', 'link', 'demo_data'], requireEvidence: true },
        { category: 'Portal',    table: 'sp_instance',             label: 'SP Widget Instance',     fields: ['filter', 'additional_options'], requireEvidence: true },

        { category: 'Workspace', table: 'sys_ux_client_script',    label: 'UX Client Script',       fields: ['script'], requireEvidence: true },
        { category: 'Workspace', table: 'sys_ux_data_broker_transform', label: 'UX Transform Data Broker', fields: ['script'], requireEvidence: true },
        { category: 'Workspace', table: 'sys_ux_macroponent',      label: 'UX Macroponent (UI Builder)', fields: ['composition'], requireEvidence: true },
        { category: 'Workspace', table: 'sys_declarative_action_assignment', label: 'Declarative Action', fields: ['condition'], scopeField: 'table' },

        { category: 'Reporting', table: 'sys_report',              label: 'Report',                 fields: ['filter'], scopeField: 'table' },
        { category: 'Reporting', table: 'sys_filter',              label: 'Saved Filter',           fields: ['filter'], scopeField: 'table' }
    ];

    // Direct-column reference: field is literally hold_reason on incident.
    var DIRECT_FIELD_ARTIFACTS = [
        { category: 'Client', table: 'sys_ui_policy_action', label: 'UI Policy Action',    fieldColumn: 'field',   query: 'ui_policy.table=' },
        { category: 'Server', table: 'metric_definition',    label: 'Metric Definition',   fieldColumn: 'field',   query: 'table=' },
        { category: 'Server', table: 'sys_dictionary_override', label: 'Dictionary Override', fieldColumn: 'element', query: 'name=' }
    ];

    // ------------------------------------------------------------------
    // Results
    // ------------------------------------------------------------------
    var results = [];
    var seen = {};
    var skippedTables = [];
    var truncatedTables = [];
    var excludedNoEvidence = 0;

    function addResult(category, artifactLabel, table, gr, matchedField, info) {
        var key = table + ':' + gr.getUniqueValue();
        var name = gr.isValidField('name') ? gr.getValue('name') : null;
        if (!name) name = gr.getDisplayValue();
        if (seen[key]) {
            var row = seen[key];
            if (row.matchedFields.indexOf(matchedField) === -1)
                row.matchedFields.push(matchedField);
            mergeInfo(row, info);
            return;
        }
        var entry = {
            category: category, artifact: artifactLabel, table: table,
            name: name || '(unnamed)', sysId: gr.getUniqueValue(),
            matchedFields: [matchedField],
            mappings: [], holdReasonRef: false, onHoldNoReason: false,
            scope: gr.isValidField('sys_scope') ? gr.getDisplayValue('sys_scope') : '',
            updatedBy: gr.getValue('sys_updated_by') || '',
            updatedOn: gr.getValue('sys_updated_on') || '',
            link: '/' + table + '.do?sys_id=' + gr.getUniqueValue()
        };
        mergeInfo(entry, info);
        seen[key] = entry;
        results.push(entry);
    }

    function mergeInfo(row, info) {
        if (!info) return;
        if (info.holdReasonRef) row.holdReasonRef = true;
        if (info.onHoldNoReason) row.onHoldNoReason = true;
        if (info.mappings) {
            for (var i = 0; i < info.mappings.length; i++) {
                var dup = false;
                for (var j = 0; j < row.mappings.length; j++)
                    if (row.mappings[j].old === info.mappings[i].old) { dup = true; break; }
                if (!dup) row.mappings.push(info.mappings[i]);
            }
        }
    }

    // Priority: HIGH = references an old value that must be remapped;
    // REVIEW = touches hold_reason or sets On Hold without a reason.
    function priorityOf(row) {
        return row.mappings.length ? 'HIGH' : 'REVIEW';
    }
    var PRIORITY_RANK = { HIGH: 0, REVIEW: 1 };

    function applyScopeFilter(gr, q) {
        if (CONFIG.globalScopeOnly && gr.isValidField('sys_scope'))
            q.push('sys_scopeIN' + CONFIG.applicationScopes.join(','));
    }

    // ------------------------------------------------------------------
    // Scanners
    // ------------------------------------------------------------------
    function scanTextArtifacts(def) {
        var gr = new GlideRecord(def.table);
        if (!gr.isValid()) { skippedTables.push(def.table); return; }
        var q = [];
        if (def.scopeField) q.push(def.scopeField + 'IN' + tableScope.join(','));
        if (def.extraQuery) q.push(def.extraQuery);
        applyScopeFilter(gr, q);

        var validFields = [];
        for (var i = 0; i < def.fields.length; i++)
            if (gr.isValidField(def.fields[i])) validFields.push(def.fields[i]);
        if (!validFields.length) { skippedTables.push(def.table + ' (no matching columns)'); return; }

        // Server-side narrowing: must mention hold_reason, a waiting label,
        // or "state" (for numeric-near-anchor and On-Hold-no-reason cases).
        var f, narrow = [];
        for (f = 0; f < validFields.length; f++) {
            narrow.push(validFields[f] + 'LIKEhold_reason');
            narrow.push(validFields[f] + 'LIKEAwaiting');
            narrow.push(validFields[f] + 'LIKEstate');
        }
        if (def.requireEvidence) {
            var evid = [];
            for (f = 0; f < validFields.length; f++) evid.push(validFields[f] + 'LIKEincident');
            q.push(evid.join('^OR'));
        }
        q.push(narrow.join('^OR'));
        gr.addEncodedQuery(q.join('^'));
        gr.setLimit(CONFIG.maxRowsPerTable);
        gr.query();
        var count = 0;
        while (gr.next()) {
            count++;
            if (def.requireEvidence) {
                var combined = '';
                for (f = 0; f < validFields.length; f++) combined += (gr.getValue(validFields[f]) || '') + '\n';
                if (!hasIncidentEvidence(combined)) { excludedNoEvidence++; continue; }
            }
            for (f = 0; f < validFields.length; f++) {
                var info = analyze(gr.getValue(validFields[f]));
                if (info) addResult(def.category, def.label, def.table, gr, validFields[f], info);
            }
        }
        if (count >= CONFIG.maxRowsPerTable) truncatedTables.push(def.table);
    }

    function scanDirectFieldArtifacts(def) {
        var gr = new GlideRecord(def.table);
        if (!gr.isValid()) { skippedTables.push(def.table); return; }
        var q = [def.fieldColumn + '=hold_reason', def.query + CONFIG.targetTable];
        applyScopeFilter(gr, q);
        gr.addEncodedQuery(q.join('^'));
        gr.setLimit(CONFIG.maxRowsPerTable);
        gr.query();
        while (gr.next())
            addResult(def.category, def.label, def.table, gr,
                      def.fieldColumn + '=hold_reason', { mappings: [], holdReasonRef: true, onHoldNoReason: false });
    }

    function scanWorkflows() {
        var vv = new GlideRecord('sys_variable_value');
        if (!vv.isValid()) { skippedTables.push('sys_variable_value'); return; }
        vv.addEncodedQuery('document=wf_activity^valueLIKEhold_reason^ORvalueLIKEAwaiting');
        vv.setLimit(CONFIG.maxRowsPerTable);
        vv.query();
        while (vv.next()) {
            var info = analyze(vv.getValue('value'));
            if (!info) continue;
            var act = new GlideRecord('wf_activity');
            if (!act.get(vv.getValue('document_key'))) continue;
            var wfTable = act.workflow_version.table + '';
            if (tableScope.indexOf(wfTable) === -1) continue;
            var key = 'wf_activity:' + act.getUniqueValue();
            if (seen[key]) { mergeInfo(seen[key], info); continue; }
            var entry = {
                category: 'Server', artifact: 'Workflow Activity', table: 'wf_activity',
                name: act.workflow_version.name + ' > ' + act.getValue('name'),
                sysId: act.getUniqueValue(), matchedFields: ['activity variable'],
                mappings: [], holdReasonRef: false, onHoldNoReason: false, scope: '',
                updatedBy: act.getValue('sys_updated_by') || '', updatedOn: act.getValue('sys_updated_on') || '',
                link: '/wf_activity.do?sys_id=' + act.getUniqueValue()
            };
            mergeInfo(entry, info);
            seen[key] = entry;
            results.push(entry);
        }
    }

    function scanFlowSnapshots() {
        var flow = new GlideRecord('sys_hub_flow');
        if (!flow.isValid()) { skippedTables.push('sys_hub_flow'); return; }
        flow.addActiveQuery();
        if (CONFIG.globalScopeOnly && flow.isValidField('sys_scope'))
            flow.addEncodedQuery('sys_scopeIN' + CONFIG.applicationScopes.join(','));
        flow.setLimit(CONFIG.maxRowsPerTable);
        flow.query();
        while (flow.next()) {
            var snap = new GlideRecord('sys_hub_flow_snapshot');
            if (!snap.get(flow.getValue('latest_snapshot'))) continue;
            var payload = snap.getValue('payload') || snap.getValue('snapshot') || '';
            if (!hasIncidentEvidence(payload)) { excludedNoEvidence++; continue; }
            var info = analyze(payload);
            if (info) addResult('Server', 'Flow (Flow Designer)', 'sys_hub_flow', flow, 'snapshot', info);
        }
    }

    // ------------------------------------------------------------------
    // Baseline: current hold_reason choices
    // ------------------------------------------------------------------
    function pad(s, n) {
        s = (s === null || s === undefined) ? '' : String(s);
        while (s.length < n) s += ' ';
        return s;
    }

    function dumpHoldReasonChoices() {
        gs.info('==================================================================');
        gs.info('CURRENT hold_reason (On Hold Reason) choices on ' + CONFIG.targetTable);
        gs.info('Target reasons: Awaiting Caller, Awaiting Problem, Awaiting Vendor, Awaiting Change' +
                (CONFIG.keepAwaitingEvidence ? ', Awaiting Evidence (custom)' : ''));
        gs.info('==================================================================');
        var c = new GlideRecord('sys_choice');
        c.addQuery('name', 'IN', tableScope.join(','));
        c.addQuery('element', 'hold_reason');
        c.addQuery('language', 'en');
        c.orderBy('sequence');
        c.query();
        var any = false;
        while (c.next()) {
            any = true;
            gs.info('  value=' + pad(c.getValue('value'), 6) + '| label=' + pad(c.getValue('label'), 26) +
                    '| inactive=' + (c.getValue('inactive') || 'false'));
        }
        if (!any) gs.info('  (no hold_reason choices found - Phase 2a choice setup creates them)');
    }

    function printMap() {
        gs.info('--- OLD -> NEW On Hold mapping applied ---');
        for (var i = 0; i < activeMap.length; i++) {
            var e = activeMap[i];
            gs.info('  ' + pad(e.sourceField + '=' + e.oldValue + ' "' + e.oldLabels[0] + '"', 40) +
                    ' -> ' + targetOf(e));
        }
        if (!CONFIG.keepAwaitingEvidence)
            gs.info('  (Awaiting Evidence treated as RETIRED per CONFIG.keepAwaitingEvidence=false)');
    }

    function csv(s) {
        s = String(s || '');
        if (s.indexOf(',') !== -1 || s.indexOf('"') !== -1) s = '"' + s.replace(/"/g, '""') + '"';
        return s;
    }

    function printReport() {
        var order = ['Server', 'Client', 'Portal', 'Workspace', 'Reporting'];
        var byCat = {};
        var counts = { HIGH: 0, REVIEW: 0, noReason: 0 };
        for (var i = 0; i < results.length; i++) {
            var r = results[i];
            r.priority = priorityOf(r);
            (byCat[r.category] = byCat[r.category] || []).push(r);
            counts[r.priority]++;
            if (r.onHoldNoReason) counts.noReason++;
        }

        gs.info('==================================================================');
        gs.info('INCIDENT ON HOLD / HOLD REASON IMPACT REPORT - ' + results.length + ' object(s)');
        gs.info('Filters: table=' + tableScope.join(',') +
                (CONFIG.globalScopeOnly ? ' | app scope=' + CONFIG.applicationScopes.join(',') : ' | all scopes'));
        gs.info('HIGH (references an old value to remap): ' + counts.HIGH +
                ' | REVIEW (hold_reason ref / On Hold set w/o reason): ' + counts.REVIEW);
        gs.info('Objects setting On Hold without a hold_reason: ' + counts.noReason +
                ' | excluded as noise (no incident evidence): ' + excludedNoEvidence);
        gs.info('==================================================================');

        for (var o = 0; o < order.length; o++) {
            var rows = byCat[order[o]] || [];
            rows.sort(function (a, b) { return PRIORITY_RANK[a.priority] - PRIORITY_RANK[b.priority]; });
            gs.info('');
            gs.info('--- ' + order[o].toUpperCase() + ' (' + rows.length + ') ---');
            for (var j = 0; j < rows.length; j++) {
                var row = rows[j];
                var change = row.mappings.length
                    ? row.mappings.map(function (mp) { return mp.old + ' -> ' + mp.target; }).join('  ||  ')
                    : (row.holdReasonRef ? 'references hold_reason (review value mapping)'
                                         : 'sets On Hold without hold_reason (add a reason)');
                gs.info(pad(row.priority, 7) + '| ' + pad(row.artifact, 24) + '| ' + pad(row.name, 42) +
                        '| ' + change);
                gs.info(pad('', 7) + '| ' + pad('', 24) + '| ' + pad('fields: ' + row.matchedFields.join(','), 42) +
                        '| ' + row.link);
            }
        }

        if (skippedTables.length) gs.info('\nSkipped (not present on this version): ' + skippedTables.join(', '));
        if (truncatedTables.length) gs.info('WARNING - hit row limit, incomplete for: ' + truncatedTables.join(', '));

        gs.info('');
        gs.info('=== CSV EXPORT (copy below this line) ===');
        gs.info('priority,category,artifact_type,table,name,sys_id,app_scope,old_to_new,onhold_no_reason,matched_fields,updated_by,updated_on,link');
        var sorted = results.slice().sort(function (a, b) { return PRIORITY_RANK[a.priority] - PRIORITY_RANK[b.priority]; });
        for (var k = 0; k < sorted.length; k++) {
            var e = sorted[k];
            var otn = e.mappings.map(function (mp) { return mp.old + ' => ' + mp.target; }).join(' ; ');
            gs.info([e.priority, e.category, e.artifact, e.table, csv(e.name), e.sysId, csv(e.scope),
                     csv(otn), (e.onHoldNoReason ? 'yes' : ''), csv(e.matchedFields.join(';')),
                     e.updatedBy, e.updatedOn, e.link].join(','));
        }
    }

    // ------------------------------------------------------------------
    // Run
    // ------------------------------------------------------------------
    dumpHoldReasonChoices();
    printMap();

    for (var t = 0; t < TEXT_ARTIFACTS.length; t++) scanTextArtifacts(TEXT_ARTIFACTS[t]);
    for (var d = 0; d < DIRECT_FIELD_ARTIFACTS.length; d++) scanDirectFieldArtifacts(DIRECT_FIELD_ARTIFACTS[d]);
    scanWorkflows();
    if (CONFIG.scanFlowSnapshots) scanFlowSnapshots();

    printReport();
})();
