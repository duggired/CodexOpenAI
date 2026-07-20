/**
 * Incident State Impact Discovery
 * --------------------------------
 * Run as: Background Script (System Definition > Scripts - Background)
 *         or paste into a Fix Script. Read-only: it only queries, never updates.
 *
 * Purpose:
 *   The choice list for incident "state" / "incident_state" was reverted to
 *   out-of-the-box (OOTB). This script produces the list of server / client /
 *   UI (portal + workspace) objects that reference the state field or your
 *   customized choice values, so a developer can work from the report instead
 *   of verifying every artifact manually.
 *
 * How to use:
 *   1. Fill CONFIG.customChoiceValues with the custom values/labels you had
 *      before the revert (e.g. ['10', '12', 'Awaiting Vendor']). Leaving it
 *      empty still works: every artifact that references the state field is
 *      reported, just without the "custom value" match level.
 *   2. Run in a sub-prod instance first. Output goes to the script output /
 *      system log (gs.info).
 *   3. The report ends with a CSV block you can copy into a sheet and use as
 *      the developer worklist.
 */

(function () {
    'use strict';

    // ------------------------------------------------------------------
    // CONFIG - adjust before running
    // ------------------------------------------------------------------
    var CONFIG = {
        // Tables whose state choices were reverted.
        targetTables: ['incident'],
        // 'task' is included because task-level artifacts (BRs, UI policies,
        // notifications...) also fire for incident records.
        tableScope: ['incident', 'task'],
        // State columns to look for.
        stateFields: ['state', 'incident_state'],
        // Your customized choice values AND/OR labels lost in the revert,
        // as strings. Example: ['10', '12', 'awaiting_vendor', 'Awaiting Vendor']
        customChoiceValues: [],
        // How close (in characters) a custom value must be to the word
        // "state" inside a script/condition to count as a value match.
        proximityChars: 80,
        // Also scan Flow Designer flow snapshots (accurate but heavy on
        // large instances - enable on sub-prod).
        scanFlowSnapshots: false,
        // Safety limit per scanned artifact table.
        maxRowsPerTable: 2000
    };

    // ------------------------------------------------------------------
    // Artifact tables scanned by free-text match on their script/condition
    // fields. scopeField (when set) restricts to CONFIG.tableScope.
    // Tables that don't exist on your version are skipped and reported.
    // ------------------------------------------------------------------
    var TEXT_ARTIFACTS = [
        // --- Server side ---
        { category: 'Server',    table: 'sys_script',              label: 'Business Rule',          fields: ['script', 'filter_condition', 'condition'], scopeField: 'collection' },
        { category: 'Server',    table: 'sys_script_include',      label: 'Script Include',         fields: ['script'] },
        { category: 'Server',    table: 'sys_script_fix',          label: 'Fix Script',             fields: ['script'] },
        { category: 'Server',    table: 'sysauto_script',          label: 'Scheduled Script Job',   fields: ['script', 'condition'] },
        { category: 'Server',    table: 'sysevent_script_action',  label: 'Script Action',          fields: ['script'] },
        { category: 'Server',    table: 'sys_ws_operation',        label: 'Scripted REST Operation', fields: ['operation_script'] },
        { category: 'Server',    table: 'sysevent_email_action',   label: 'Notification',           fields: ['condition', 'advanced_condition', 'subject', 'message_html'], scopeField: 'collection' },
        { category: 'Server',    table: 'contract_sla',            label: 'SLA Definition',         fields: ['start_condition', 'stop_condition', 'pause_condition', 'reset_condition'], scopeField: 'collection' },
        { category: 'Server',    table: 'sys_security_acl',        label: 'ACL',                    fields: ['condition', 'script'], extraQuery: 'nameSTARTSWITHincident' },
        { category: 'Server',    table: 'sys_transform_script',    label: 'Transform Map Script',   fields: ['script'], extraQuery: 'map.target_tableINincident,task' },
        { category: 'Server',    table: 'sys_transform_entry',     label: 'Transform Field Map',    fields: ['source_script'], extraQuery: 'map.target_tableINincident,task' },

        // --- Client side (classic UI) ---
        { category: 'Client',    table: 'sys_script_client',       label: 'Client Script',          fields: ['script'], scopeField: 'table' },
        { category: 'Client',    table: 'sys_ui_policy',           label: 'UI Policy',              fields: ['conditions', 'script_true', 'script_false'], scopeField: 'table' },
        { category: 'Client',    table: 'sys_ui_action',           label: 'UI Action',              fields: ['script', 'condition'], scopeField: 'table' },

        // --- Service Portal ---
        { category: 'Portal',    table: 'sp_widget',               label: 'SP Widget',              fields: ['script', 'client_script', 'template', 'link', 'demo_data'] },
        { category: 'Portal',    table: 'sp_instance',             label: 'SP Widget Instance',     fields: ['filter', 'additional_options'] },

        // --- Workspace / UI Builder ---
        { category: 'Workspace', table: 'sys_ux_client_script',    label: 'UX Client Script',       fields: ['script'] },
        { category: 'Workspace', table: 'sys_ux_data_broker_transform', label: 'UX Transform Data Broker', fields: ['script'] },
        { category: 'Workspace', table: 'sys_ux_macroponent',      label: 'UX Macroponent (UI Builder)', fields: ['composition'], extraQuery: 'compositionLIKEincident' },
        { category: 'Workspace', table: 'sys_declarative_action_assignment', label: 'Declarative Action', fields: ['condition'], scopeField: 'table' },

        // --- Reporting / conditions stored as encoded queries ---
        { category: 'Reporting', table: 'sys_report',              label: 'Report',                 fields: ['filter'], scopeField: 'table' },
        { category: 'Reporting', table: 'sys_filter',              label: 'Saved Filter',           fields: ['filter'], scopeField: 'table' }
    ];

    // Artifacts that reference the state field directly through a column
    // (no text matching needed - these are always impacted).
    var DIRECT_FIELD_ARTIFACTS = [
        { category: 'Client',    table: 'sys_ui_policy_action', label: 'UI Policy Action',      fieldColumn: 'field',   query: 'ui_policy.tableIN' },
        { category: 'Server',    table: 'sys_transform_entry',  label: 'Transform Field Map',   fieldColumn: 'target_field', query: 'map.target_tableIN' },
        { category: 'Server',    table: 'metric_definition',    label: 'Metric Definition',     fieldColumn: 'field',   query: 'tableIN' },
        { category: 'Server',    table: 'sys_dictionary_override', label: 'Dictionary Override', fieldColumn: 'element', query: 'nameIN' }
    ];

    // ------------------------------------------------------------------
    // Matching helpers
    // ------------------------------------------------------------------
    var FIELD_PATTERNS = [
        /['"](incident_state|state)['"]/,                                        // 'state' as a quoted field name (g_form/GlideRecord APIs)
        /\.(incident_state|state)\b/,                                            // current.state, gr.incident_state, data.state ...
        /\b(incident_state|state)\s*(=|!=|>=|<=|>|<|IN\b|NOT ?IN\b|CHANGES)/i    // encoded queries and condition strings
    ];

    function escapeRegex(s) {
        return String(s).replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    }

    var valueRegexes = [];
    for (var v = 0; v < CONFIG.customChoiceValues.length; v++) {
        valueRegexes.push({
            raw: String(CONFIG.customChoiceValues[v]),
            re: new RegExp('(^|[^\\w])' + escapeRegex(CONFIG.customChoiceValues[v]) + '([^\\w]|$)')
        });
    }

    /**
     * Returns null if the text does not reference the state field, otherwise
     * { fieldRef: true, values: ['10','12'] } where values are the custom
     * choice values found within proximityChars of an occurrence of "state".
     */
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
        var idx = text.indexOf('state');
        while (idx !== -1) {
            var win = text.substring(Math.max(0, idx - CONFIG.proximityChars), idx + CONFIG.proximityChars);
            for (var j = 0; j < valueRegexes.length; j++) {
                if (valueRegexes[j].re.test(win))
                    found[valueRegexes[j].raw] = true;
            }
            idx = text.indexOf('state', idx + 5);
        }
        var values = [];
        for (var k in found)
            values.push(k);
        return { fieldRef: true, values: values };
    }

    // ------------------------------------------------------------------
    // Result collection
    // ------------------------------------------------------------------
    var results = [];
    var seen = {};
    var skippedTables = [];
    var truncatedTables = [];

    function addResult(category, artifactLabel, table, gr, matchedField, matchInfo) {
        var key = table + ':' + gr.getUniqueValue();
        var name = gr.isValidField('name') ? gr.getValue('name') : null;
        if (!name)
            name = gr.getDisplayValue();
        if (seen[key]) {
            // merge match info into the existing row
            var row = seen[key];
            if (row.matchedFields.indexOf(matchedField) === -1)
                row.matchedFields.push(matchedField);
            for (var i = 0; i < matchInfo.values.length; i++)
                if (row.values.indexOf(matchInfo.values[i]) === -1)
                    row.values.push(matchInfo.values[i]);
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
            updatedBy: gr.getValue('sys_updated_by') || '',
            updatedOn: gr.getValue('sys_updated_on') || '',
            link: '/' + table + '.do?sys_id=' + gr.getUniqueValue()
        };
        seen[key] = entry;
        results.push(entry);
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
            q.push(def.scopeField + 'IN' + CONFIG.tableScope.join(','));
        if (def.extraQuery)
            q.push(def.extraQuery);
        var likes = [];
        for (var i = 0; i < def.fields.length; i++) {
            if (gr.isValidField(def.fields[i]))
                likes.push(def.fields[i] + 'LIKEstate');
        }
        if (!likes.length) {
            skippedTables.push(def.table + ' (no matching columns)');
            return;
        }
        q.push(likes.join('^OR'));
        gr.addEncodedQuery(q.join('^'));
        gr.setLimit(CONFIG.maxRowsPerTable);
        gr.query();
        var count = 0;
        while (gr.next()) {
            count++;
            for (var f = 0; f < def.fields.length; f++) {
                if (!gr.isValidField(def.fields[f]))
                    continue;
                var m = analyze(gr.getValue(def.fields[f]));
                if (m)
                    addResult(def.category, def.label, def.table, gr, def.fields[f], m);
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
        var q = def.fieldColumn + 'IN' + CONFIG.stateFields.join(',') +
                '^' + def.query + CONFIG.tableScope.join(',');
        gr.addEncodedQuery(q);
        gr.setLimit(CONFIG.maxRowsPerTable);
        gr.query();
        while (gr.next())
            addResult(def.category, def.label, def.table, gr, def.fieldColumn + '=' + gr.getValue(def.fieldColumn), { values: [] });
    }

    // Classic workflow activities keep their scripts/conditions in
    // sys_variable_value rows pointing at wf_activity.
    function scanWorkflows() {
        var vv = new GlideRecord('sys_variable_value');
        if (!vv.isValid()) {
            skippedTables.push('sys_variable_value');
            return;
        }
        vv.addEncodedQuery('document=wf_activity^valueLIKEstate');
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
            if (wfTable && CONFIG.tableScope.indexOf(wfTable) === -1)
                continue;
            var key = 'wf_activity:' + act.getUniqueValue();
            if (seen[key])
                continue;
            var entry = {
                category: 'Server',
                artifact: 'Workflow Activity',
                table: 'wf_activity',
                name: act.workflow_version.name + ' > ' + act.getValue('name'),
                sysId: act.getUniqueValue(),
                matchedFields: ['activity variable'],
                values: m.values.slice(),
                updatedBy: act.getValue('sys_updated_by') || '',
                updatedOn: act.getValue('sys_updated_on') || '',
                link: '/wf_activity.do?sys_id=' + act.getUniqueValue()
            };
            seen[key] = entry;
            results.push(entry);
        }
    }

    // Flow Designer: match against the latest snapshot JSON of each flow.
    function scanFlowSnapshots() {
        var flow = new GlideRecord('sys_hub_flow');
        if (!flow.isValid()) {
            skippedTables.push('sys_hub_flow');
            return;
        }
        flow.addActiveQuery();
        flow.setLimit(CONFIG.maxRowsPerTable);
        flow.query();
        while (flow.next()) {
            var snap = new GlideRecord('sys_hub_flow_snapshot');
            if (!snap.get(flow.getValue('latest_snapshot')))
                continue;
            var payload = snap.getValue('payload') || snap.getValue('snapshot') || '';
            if (payload.indexOf('incident') === -1)
                continue;
            var m = analyze(payload);
            if (m)
                addResult('Server', 'Flow (Flow Designer)', 'sys_hub_flow', flow, 'snapshot', m);
        }
    }

    // ------------------------------------------------------------------
    // Baseline: what the choice lists look like right now (post-revert)
    // ------------------------------------------------------------------
    function dumpCurrentChoices() {
        gs.info('==================================================================');
        gs.info('CURRENT (post-revert) CHOICES for ' + CONFIG.targetTables.join(',') + '.' + CONFIG.stateFields.join('/'));
        gs.info('Compare this against your customized list; anything missing must be re-added.');
        gs.info('==================================================================');
        var c = new GlideRecord('sys_choice');
        c.addQuery('name', 'IN', CONFIG.targetTables.join(','));
        c.addQuery('element', 'IN', CONFIG.stateFields.join(','));
        c.addQuery('language', 'en');
        c.orderBy('element');
        c.orderBy('sequence');
        c.query();
        while (c.next()) {
            gs.info(pad(c.getValue('element'), 16) + '| value=' + pad(c.getValue('value'), 6) +
                    '| label=' + pad(c.getValue('label'), 28) +
                    '| inactive=' + pad(c.getValue('inactive') || 'false', 6) +
                    '| updated=' + c.getValue('sys_updated_on') + ' by ' + c.getValue('sys_updated_by'));
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
        for (var i = 0; i < results.length; i++) {
            var r = results[i];
            if (!byCat[r.category])
                byCat[r.category] = [];
            byCat[r.category].push(r);
        }

        gs.info('==================================================================');
        gs.info('INCIDENT STATE IMPACT REPORT - ' + results.length + ' impacted object(s)');
        gs.info('Custom values searched: ' + (CONFIG.customChoiceValues.length ? CONFIG.customChoiceValues.join(', ') : '(none configured - field-reference matches only)'));
        gs.info('==================================================================');

        for (var o = 0; o < order.length; o++) {
            var cat = order[o];
            var rows = byCat[cat] || [];
            gs.info('');
            gs.info('--- ' + cat.toUpperCase() + ' (' + rows.length + ') ---');
            for (var j = 0; j < rows.length; j++) {
                var row = rows[j];
                gs.info(pad(row.artifact, 26) + '| ' + pad(row.name, 45) +
                        '| match: ' + pad(row.matchedFields.join(','), 30) +
                        '| custom values: ' + pad(row.values.length ? row.values.join(',') : '-', 12) +
                        '| ' + row.link);
            }
        }

        if (skippedTables.length)
            gs.info('\nSkipped (table not present on this instance/version): ' + skippedTables.join(', '));
        if (truncatedTables.length)
            gs.info('WARNING - hit maxRowsPerTable limit, results incomplete for: ' + truncatedTables.join(', '));

        // CSV block - copy into a spreadsheet as the developer worklist
        gs.info('');
        gs.info('=== CSV EXPORT (copy below this line) ===');
        gs.info('category,artifact_type,table,name,sys_id,matched_fields,custom_values_found,updated_by,updated_on,link');
        for (var k = 0; k < results.length; k++) {
            var e = results[k];
            gs.info([e.category, e.artifact, e.table, csv(e.name), e.sysId,
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
    dumpCurrentChoices();

    for (var t = 0; t < TEXT_ARTIFACTS.length; t++)
        scanTextArtifacts(TEXT_ARTIFACTS[t]);
    for (var d = 0; d < DIRECT_FIELD_ARTIFACTS.length; d++)
        scanDirectFieldArtifacts(DIRECT_FIELD_ARTIFACTS[d]);
    scanWorkflows();
    if (CONFIG.scanFlowSnapshots)
        scanFlowSnapshots();

    printReport();
})();
