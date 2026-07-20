/**
 * Incident State Impact Discovery - STRICT MODE
 * ---------------------------------------------
 * Run as: Background Script. Read-only: it only queries, never updates.
 *
 * Stricter companion to incident-state-impact-discovery.js. The broad script
 * reports anything touching a "state" field, which pulls in objects about
 * other tables (change, problem, sc_task...) and other scoped apps. This
 * version only reports an object as impacted when there is evidence it is
 * about the INCIDENT table specifically:
 *
 *   1. Table-scoped artifacts (business rules, client scripts, UI policies,
 *      UI actions, notifications, SLAs, reports...) must sit on table
 *      'incident' exactly (task-level inclusion is off by default).
 *   2. Unscoped artifacts (script includes, widgets, UX scripts, scheduled
 *      jobs, fix scripts...) must contain BOTH, anywhere across their
 *      scanned fields:
 *        - incident evidence: new GlideRecord('incident') /
 *          GlideRecordSecure / GlideAggregate on incident, table:'incident',
 *          a quoted 'incident' literal, or the incident-only fields
 *          incident_state / hold_reason, AND
 *        - a state field reference (state / incident_state / hold_reason).
 *   3. Only artifacts in the Global application (sys_scope=global) are
 *      reported (configurable).
 *
 * Trade-off: strictness can hide real impact - a task-level business rule
 * checking current.state DOES run for incidents, and a generic widget whose
 * table arrives via an instance option never says 'incident' in code. Use
 * this report as the primary worklist and the broad script as the
 * completeness cross-check. Loosen selectively via includeTaskLevel /
 * globalScopeOnly.
 *
 * Priority meanings (same as the broad script):
 *   HIGH   - references a RETIRED value/label (breaks after alignment).
 *   MEDIUM - references a value that survives but changes meaning/label.
 *   REVIEW - touches the field without a recognized value nearby.
 */

(function () {
    'use strict';

    // ------------------------------------------------------------------
    // CONFIG - adjust before running
    // ------------------------------------------------------------------
    var CONFIG = {
        targetTable: 'incident',
        // false = incident-only (strict). true = also task-level artifacts,
        // which do run for incidents but cover other task types too.
        includeTaskLevel: false,
        // Only report artifacts in these application scopes. 'global' is the
        // sys_id of the Global application. Set globalScopeOnly:false to
        // scan every scope.
        globalScopeOnly: true,
        applicationScopes: ['global'],

        stateFields: ['state', 'incident_state', 'hold_reason'],

        // From the alignment one-pager (same map as the broad script):
        customChoiceValues: {
            retired: [
                '4', '5', '9', '10', '11', '12',
                'Awaiting Problem', 'Awaiting User Info', 'Awaiting Evidence',
                'Awaiting Release', 'Assigned', 'Active',
                'Open', 'Work in Progress', 'Closed Complete',
                'Pending Approval', 'Cancelled'
            ],
            changed: ['1', '2', '3', '7', '8', 'On Hold']
        },

        proximityChars: 80,
        scanFlowSnapshots: false,
        maxRowsPerTable: 2000
    };

    var tableScope = [CONFIG.targetTable];
    if (CONFIG.includeTaskLevel)
        tableScope.push('task');

    // ------------------------------------------------------------------
    // Artifact tables. requireEvidence:true marks unscoped artifacts that
    // must additionally prove incident context in their own text.
    // ------------------------------------------------------------------
    var TEXT_ARTIFACTS = [
        // --- Server side ---
        { category: 'Server',    table: 'sys_script',              label: 'Business Rule',          fields: ['script', 'filter_condition', 'condition'], scopeField: 'collection' },
        { category: 'Server',    table: 'sys_script_include',      label: 'Script Include',         fields: ['script'], requireEvidence: true },
        { category: 'Server',    table: 'sys_script_fix',          label: 'Fix Script',             fields: ['script'], requireEvidence: true },
        { category: 'Server',    table: 'sysauto_script',          label: 'Scheduled Script Job',   fields: ['script', 'condition'], requireEvidence: true },
        { category: 'Server',    table: 'sysevent_script_action',  label: 'Script Action',          fields: ['script'], requireEvidence: true },
        { category: 'Server',    table: 'sys_ws_operation',        label: 'Scripted REST Operation', fields: ['operation_script'], requireEvidence: true },
        { category: 'Server',    table: 'sys_rest_message_fn',     label: 'REST Message Method (outbound)', fields: ['content'], requireEvidence: true },
        { category: 'Server',    table: 'sysevent_email_action',   label: 'Notification',           fields: ['condition', 'advanced_condition', 'subject', 'message_html'], scopeField: 'collection' },
        { category: 'Server',    table: 'sysevent_email_template', label: 'Email Template',         fields: ['subject', 'message_html'], scopeField: 'collection' },
        { category: 'Server',    table: 'contract_sla',            label: 'SLA Definition',         fields: ['start_condition', 'stop_condition', 'pause_condition', 'reset_condition'], scopeField: 'collection' },
        { category: 'Server',    table: 'sys_security_acl',        label: 'ACL',                    fields: ['condition', 'script'], extraQuery: 'nameSTARTSWITHincident' },
        { category: 'Server',    table: 'sys_transform_script',    label: 'Transform Map Script',   fields: ['script'], extraQuery: 'map.target_table=incident' },
        { category: 'Server',    table: 'sys_transform_entry',     label: 'Transform Field Map',    fields: ['source_script'], extraQuery: 'map.target_table=incident' },

        // --- Client side (classic UI) ---
        { category: 'Client',    table: 'sys_script_client',       label: 'Client Script',          fields: ['script'], scopeField: 'table' },
        { category: 'Client',    table: 'sys_ui_policy',           label: 'UI Policy',              fields: ['conditions', 'script_true', 'script_false'], scopeField: 'table' },
        { category: 'Client',    table: 'sys_ui_action',           label: 'UI Action',              fields: ['script', 'condition'], scopeField: 'table' },

        // --- Service Portal ---
        { category: 'Portal',    table: 'sp_widget',               label: 'SP Widget',              fields: ['script', 'client_script', 'template', 'link', 'demo_data'], requireEvidence: true },
        { category: 'Portal',    table: 'sp_instance',             label: 'SP Widget Instance',     fields: ['filter', 'additional_options'], requireEvidence: true },

        // --- Workspace / UI Builder ---
        { category: 'Workspace', table: 'sys_ux_client_script',    label: 'UX Client Script',       fields: ['script'], requireEvidence: true },
        { category: 'Workspace', table: 'sys_ux_data_broker_transform', label: 'UX Transform Data Broker', fields: ['script'], requireEvidence: true },
        { category: 'Workspace', table: 'sys_ux_macroponent',      label: 'UX Macroponent (UI Builder)', fields: ['composition'], requireEvidence: true },
        { category: 'Workspace', table: 'sys_declarative_action_assignment', label: 'Declarative Action', fields: ['condition'], scopeField: 'table' },

        // --- Reporting / conditions stored as encoded queries ---
        { category: 'Reporting', table: 'sys_report',              label: 'Report',                 fields: ['filter'], scopeField: 'table' },
        { category: 'Reporting', table: 'sys_filter',              label: 'Saved Filter',           fields: ['filter'], scopeField: 'table' }
    ];

    // Artifacts referencing the state field directly through a column -
    // scoped straight to incident, always impacted.
    var DIRECT_FIELD_ARTIFACTS = [
        { category: 'Client',    table: 'sys_ui_policy_action', label: 'UI Policy Action',      fieldColumn: 'field',   query: 'ui_policy.table=' },
        { category: 'Server',    table: 'sys_transform_entry',  label: 'Transform Field Map',   fieldColumn: 'target_field', query: 'map.target_table=' },
        { category: 'Server',    table: 'metric_definition',    label: 'Metric Definition',     fieldColumn: 'field',   query: 'table=' },
        { category: 'Server',    table: 'sys_data_policy_rule', label: 'Data Policy Rule',      fieldColumn: 'field',   query: 'sys_data_policy.model_table=' },
        { category: 'Server',    table: 'sys_dictionary_override', label: 'Dictionary Override', fieldColumn: 'element', query: 'name=' }
    ];

    // ------------------------------------------------------------------
    // Matching helpers
    // ------------------------------------------------------------------
    var FIELD_PATTERNS = [
        /['"](incident_state|state|hold_reason)['"]/,
        /\.(incident_state|state|hold_reason)\b/,
        /\b(incident_state|state|hold_reason)\s*(=|!=|>=|<=|>|<|IN\b|NOT ?IN\b|CHANGES)/i
    ];

    // Evidence that a script/text is actually about the incident table.
    var INCIDENT_EVIDENCE_PATTERNS = [
        /GlideRecord(Secure)?\s*\(\s*['"]incident['"]/i,     // new GlideRecord('incident')
        /GlideAggregate\s*\(\s*['"]incident['"]/i,
        /\btable\s*(:|===?|=)\s*['"]incident['"]/i,          // table: 'incident' / table == 'incident'
        /['"]incident['"]/,                                  // 'incident' passed to any API / option
        /['"]incident\.[a-z_]+['"]/i,                        // 'incident.state' style dot strings
        /\bincident_state\b/,                                // field exists only on incident
        /\bhold_reason\b/                                    // incident-specific field
    ];

    function hasIncidentEvidence(text) {
        if (!text)
            return false;
        for (var i = 0; i < INCIDENT_EVIDENCE_PATTERNS.length; i++)
            if (INCIDENT_EVIDENCE_PATTERNS[i].test(text))
                return true;
        return false;
    }

    var PRIORITY_RANK = { HIGH: 0, MEDIUM: 1, REVIEW: 2 };

    function escapeRegex(s) {
        return String(s).replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    }

    function buildValueRegexes(tokens, priority) {
        var out = [];
        for (var i = 0; i < tokens.length; i++) {
            out.push({
                raw: String(tokens[i]),
                priority: priority,
                re: new RegExp('(^|[^\\w])' + escapeRegex(tokens[i]) + '([^\\w]|$)', /^[0-9]+$/.test(tokens[i]) ? '' : 'i')
            });
        }
        return out;
    }

    var valueRegexes = buildValueRegexes(CONFIG.customChoiceValues.retired, 'HIGH')
        .concat(buildValueRegexes(CONFIG.customChoiceValues.changed, 'MEDIUM'));

    function analyze(text) {
        if (!text)
            return null;
        var fieldRef = false;
        for (var i = 0; i < FIELD_PATTERNS.length; i++) {
            if (FIELD_PATTERNS[i].test(text)) {
                fieldRef = true;
                break;
            }
        }
        if (!fieldRef)
            return null;

        var found = {};
        var priority = 'REVIEW';
        var anchors = ['state', 'hold_reason'];
        for (var a = 0; a < anchors.length; a++) {
            var idx = text.indexOf(anchors[a]);
            while (idx !== -1) {
                var win = text.substring(Math.max(0, idx - CONFIG.proximityChars), idx + CONFIG.proximityChars);
                for (var j = 0; j < valueRegexes.length; j++) {
                    if (valueRegexes[j].re.test(win)) {
                        found[valueRegexes[j].raw] = true;
                        if (PRIORITY_RANK[valueRegexes[j].priority] < PRIORITY_RANK[priority])
                            priority = valueRegexes[j].priority;
                    }
                }
                idx = text.indexOf(anchors[a], idx + anchors[a].length);
            }
        }
        var values = [];
        for (var k in found)
            values.push(k);
        return { values: values, priority: priority };
    }

    // ------------------------------------------------------------------
    // Result collection
    // ------------------------------------------------------------------
    var results = [];
    var seen = {};
    var skippedTables = [];
    var truncatedTables = [];
    var excludedNoEvidence = 0;   // matched a state ref but lacked incident evidence
    var excludedByScope = {};     // per-table count filtered out by app scope

    function addResult(category, artifactLabel, table, gr, matchedField, matchInfo) {
        var key = table + ':' + gr.getUniqueValue();
        var name = gr.isValidField('name') ? gr.getValue('name') : null;
        if (!name)
            name = gr.getDisplayValue();
        if (seen[key]) {
            var row = seen[key];
            if (row.matchedFields.indexOf(matchedField) === -1)
                row.matchedFields.push(matchedField);
            for (var i = 0; i < matchInfo.values.length; i++)
                if (row.values.indexOf(matchInfo.values[i]) === -1)
                    row.values.push(matchInfo.values[i]);
            if (PRIORITY_RANK[matchInfo.priority] < PRIORITY_RANK[row.priority])
                row.priority = matchInfo.priority;
            return;
        }
        var entry = {
            category: category,
            artifact: artifactLabel,
            table: table,
            name: name || '(unnamed)',
            sysId: gr.getUniqueValue(),
            matchedFields: [matchedField],
            values: matchInfo.values.slice(),
            priority: matchInfo.priority,
            scope: gr.isValidField('sys_scope') ? gr.getDisplayValue('sys_scope') : '',
            updatedBy: gr.getValue('sys_updated_by') || '',
            updatedOn: gr.getValue('sys_updated_on') || '',
            link: '/' + table + '.do?sys_id=' + gr.getUniqueValue()
        };
        seen[key] = entry;
        results.push(entry);
    }

    function applyScopeFilter(gr, q) {
        if (CONFIG.globalScopeOnly && gr.isValidField('sys_scope'))
            q.push('sys_scopeIN' + CONFIG.applicationScopes.join(','));
    }

    // ------------------------------------------------------------------
    // Scanners
    // ------------------------------------------------------------------
    function scanTextArtifacts(def) {
        var gr = new GlideRecord(def.table);
        if (!gr.isValid()) {
            skippedTables.push(def.table);
            return;
        }
        var q = [];
        if (def.scopeField)
            q.push(def.scopeField + 'IN' + tableScope.join(','));
        if (def.extraQuery)
            q.push(def.extraQuery);
        applyScopeFilter(gr, q);

        var validFields = [];
        for (var i = 0; i < def.fields.length; i++)
            if (gr.isValidField(def.fields[i]))
                validFields.push(def.fields[i]);
        if (!validFields.length) {
            skippedTables.push(def.table + ' (no matching columns)');
            return;
        }

        // Unscoped artifacts must contain 'incident' (covers incident_state
        // too) somewhere - enforced server-side to cut the candidate set,
        // then verified precisely with regexes below.
        var f;
        if (def.requireEvidence) {
            var incLikes = [];
            for (f = 0; f < validFields.length; f++)
                incLikes.push(validFields[f] + 'LIKEincident');
            q.push(incLikes.join('^OR'));
        }
        var stateLikes = [];
        for (f = 0; f < validFields.length; f++) {
            stateLikes.push(validFields[f] + 'LIKEstate');
            stateLikes.push(validFields[f] + 'LIKEhold_reason');
        }
        q.push(stateLikes.join('^OR'));

        gr.addEncodedQuery(q.join('^'));
        gr.setLimit(CONFIG.maxRowsPerTable);
        gr.query();
        var count = 0;
        while (gr.next()) {
            count++;
            // Evidence is judged across ALL scanned fields of the record, so
            // e.g. a widget with GlideRecord('incident') in its server script
            // and a state check in its client script still qualifies.
            if (def.requireEvidence) {
                var combined = '';
                for (f = 0; f < validFields.length; f++)
                    combined += (gr.getValue(validFields[f]) || '') + '\n';
                if (!hasIncidentEvidence(combined)) {
                    excludedNoEvidence++;
                    continue;
                }
            }
            for (f = 0; f < validFields.length; f++) {
                var m = analyze(gr.getValue(validFields[f]));
                if (m)
                    addResult(def.category, def.label, def.table, gr, validFields[f], m);
            }
        }
        if (count >= CONFIG.maxRowsPerTable)
            truncatedTables.push(def.table);
    }

    function scanDirectFieldArtifacts(def) {
        var gr = new GlideRecord(def.table);
        if (!gr.isValid()) {
            skippedTables.push(def.table);
            return;
        }
        var q = [def.fieldColumn + 'IN' + CONFIG.stateFields.join(','),
                 def.query + CONFIG.targetTable];
        applyScopeFilter(gr, q);
        gr.addEncodedQuery(q.join('^'));
        gr.setLimit(CONFIG.maxRowsPerTable);
        gr.query();
        while (gr.next())
            addResult(def.category, def.label, def.table, gr,
                      def.fieldColumn + '=' + gr.getValue(def.fieldColumn),
                      { values: [], priority: 'REVIEW' });
    }

    // Classic workflow activities - strictly workflows ON the incident table.
    function scanWorkflows() {
        var vv = new GlideRecord('sys_variable_value');
        if (!vv.isValid()) {
            skippedTables.push('sys_variable_value');
            return;
        }
        vv.addEncodedQuery('document=wf_activity^valueLIKEstate^ORvalueLIKEhold_reason');
        vv.setLimit(CONFIG.maxRowsPerTable);
        vv.query();
        while (vv.next()) {
            var m = analyze(vv.getValue('value'));
            if (!m)
                continue;
            var act = new GlideRecord('wf_activity');
            if (!act.get(vv.getValue('document_key')))
                continue;
            var wfTable = act.workflow_version.table + '';
            if (tableScope.indexOf(wfTable) === -1)
                continue;
            var key = 'wf_activity:' + act.getUniqueValue();
            if (seen[key]) {
                var row = seen[key];
                for (var i = 0; i < m.values.length; i++)
                    if (row.values.indexOf(m.values[i]) === -1)
                        row.values.push(m.values[i]);
                if (PRIORITY_RANK[m.priority] < PRIORITY_RANK[row.priority])
                    row.priority = m.priority;
                continue;
            }
            var entry = {
                category: 'Server',
                artifact: 'Workflow Activity',
                table: 'wf_activity',
                name: act.workflow_version.name + ' > ' + act.getValue('name'),
                sysId: act.getUniqueValue(),
                matchedFields: ['activity variable'],
                values: m.values.slice(),
                priority: m.priority,
                scope: '',
                updatedBy: act.getValue('sys_updated_by') || '',
                updatedOn: act.getValue('sys_updated_on') || '',
                link: '/wf_activity.do?sys_id=' + act.getUniqueValue()
            };
            seen[key] = entry;
            results.push(entry);
        }
    }

    // Flow Designer - only flows whose snapshot shows incident evidence.
    function scanFlowSnapshots() {
        var flow = new GlideRecord('sys_hub_flow');
        if (!flow.isValid()) {
            skippedTables.push('sys_hub_flow');
            return;
        }
        flow.addActiveQuery();
        if (CONFIG.globalScopeOnly && flow.isValidField('sys_scope'))
            flow.addEncodedQuery('sys_scopeIN' + CONFIG.applicationScopes.join(','));
        flow.setLimit(CONFIG.maxRowsPerTable);
        flow.query();
        while (flow.next()) {
            var snap = new GlideRecord('sys_hub_flow_snapshot');
            if (!snap.get(flow.getValue('latest_snapshot')))
                continue;
            var payload = snap.getValue('payload') || snap.getValue('snapshot') || '';
            if (!hasIncidentEvidence(payload)) {
                excludedNoEvidence++;
                continue;
            }
            var m = analyze(payload);
            if (m)
                addResult('Server', 'Flow (Flow Designer)', 'sys_hub_flow', flow, 'snapshot', m);
        }
    }

    // ------------------------------------------------------------------
    // Output
    // ------------------------------------------------------------------
    function pad(s, n) {
        s = (s === null || s === undefined) ? '' : String(s);
        while (s.length < n)
            s += ' ';
        return s;
    }

    function printReport() {
        var order = ['Server', 'Client', 'Portal', 'Workspace', 'Reporting'];
        var byCat = {};
        var byPriority = { HIGH: 0, MEDIUM: 0, REVIEW: 0 };
        for (var i = 0; i < results.length; i++) {
            var r = results[i];
            if (!byCat[r.category])
                byCat[r.category] = [];
            byCat[r.category].push(r);
            byPriority[r.priority]++;
        }

        gs.info('==================================================================');
        gs.info('INCIDENT STATE IMPACT REPORT (STRICT) - ' + results.length + ' impacted object(s)');
        gs.info('Filters: table=' + tableScope.join(',') +
                (CONFIG.globalScopeOnly ? ' | app scope=' + CONFIG.applicationScopes.join(',') : ' | all app scopes') +
                ' | incident evidence required on unscoped artifacts');
        gs.info('HIGH (retired value refs): ' + byPriority.HIGH +
                ' | MEDIUM (changed value refs): ' + byPriority.MEDIUM +
                ' | REVIEW (field ref only): ' + byPriority.REVIEW);
        gs.info('Excluded as noise (state ref but no incident evidence): ' + excludedNoEvidence);
        gs.info('==================================================================');

        for (var o = 0; o < order.length; o++) {
            var cat = order[o];
            var rows = byCat[cat] || [];
            rows.sort(function (a, b) {
                return PRIORITY_RANK[a.priority] - PRIORITY_RANK[b.priority];
            });
            gs.info('');
            gs.info('--- ' + cat.toUpperCase() + ' (' + rows.length + ') ---');
            for (var j = 0; j < rows.length; j++) {
                var row = rows[j];
                gs.info(pad(row.priority, 7) + '| ' + pad(row.artifact, 26) + '| ' + pad(row.name, 45) +
                        '| match: ' + pad(row.matchedFields.join(','), 30) +
                        '| values: ' + pad(row.values.length ? row.values.join(',') : '-', 12) +
                        '| ' + row.link);
            }
        }

        if (skippedTables.length)
            gs.info('\nSkipped (table not present on this instance/version): ' + skippedTables.join(', '));
        if (truncatedTables.length)
            gs.info('WARNING - hit maxRowsPerTable limit, results incomplete for: ' + truncatedTables.join(', '));

        gs.info('');
        gs.info('=== CSV EXPORT (copy below this line) ===');
        gs.info('priority,category,artifact_type,table,name,sys_id,app_scope,matched_fields,values_found,updated_by,updated_on,link');
        var sorted = results.slice().sort(function (a, b) {
            return PRIORITY_RANK[a.priority] - PRIORITY_RANK[b.priority];
        });
        for (var k = 0; k < sorted.length; k++) {
            var e = sorted[k];
            gs.info([e.priority, e.category, e.artifact, e.table, csv(e.name), e.sysId, csv(e.scope),
                     csv(e.matchedFields.join(';')), csv(e.values.join(';')),
                     e.updatedBy, e.updatedOn, e.link].join(','));
        }
    }

    function csv(s) {
        s = String(s || '');
        if (s.indexOf(',') !== -1 || s.indexOf('"') !== -1)
            s = '"' + s.replace(/"/g, '""') + '"';
        return s;
    }

    // ------------------------------------------------------------------
    // Run
    // ------------------------------------------------------------------
    for (var t = 0; t < TEXT_ARTIFACTS.length; t++)
        scanTextArtifacts(TEXT_ARTIFACTS[t]);
    for (var d = 0; d < DIRECT_FIELD_ARTIFACTS.length; d++)
        scanDirectFieldArtifacts(DIRECT_FIELD_ARTIFACTS[d]);
    scanWorkflows();
    if (CONFIG.scanFlowSnapshots)
        scanFlowSnapshots();

    printReport();
})();
