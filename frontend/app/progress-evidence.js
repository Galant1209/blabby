(function (root) {
    'use strict';

    const ALLOWED_STATUSES = new Set([
        'evidence_available',
        'insufficient_evidence',
        'still_working',
        'improvement_observed',
    ]);

    function clean(value, maxLength) {
        return String(value || '').replace(/\s+/g, ' ').trim().slice(0, maxLength);
    }

    function modelFrom(payload) {
        if (!payload || payload.has_evidence !== true) {
            return { hasEvidence: false, reason: 'insufficient_evidence' };
        }
        const tag = clean(payload.weakness?.tag, 64);
        const label = clean(payload.weakness?.label, 80);
        const before = clean(payload.before?.snippet, 181);
        const after = clean(payload.after?.snippet, 181);
        const beforeId = clean(payload.before?.record_id, 64);
        const afterId = clean(payload.after?.record_id, 64);
        const status = clean(payload.observation?.status, 40);
        const observation = clean(payload.observation?.label, 160);
        if (!tag || !label || !before || !after || !beforeId || !afterId || !ALLOWED_STATUSES.has(status)) {
            return { hasEvidence: false, reason: 'insufficient_evidence' };
        }
        return {
            hasEvidence: true,
            weakness: { tag, label },
            before: { id: beforeId, createdAt: payload.before?.created_at || '', snippet: before },
            after: { id: afterId, createdAt: payload.after?.created_at || '', snippet: after },
            observation: { status, label: observation },
        };
    }

    function eventProps(model, source) {
        return {
            weakness_category: model?.hasEvidence ? clean(model.weakness.tag, 64) : '',
            observation_status: model?.hasEvidence
                ? clean(model.observation.status, 40)
                : 'insufficient_evidence',
            source: clean(source || 'authenticated_home', 64),
            authenticated: true,
        };
    }

    function includesRecord(model, recordId) {
        return Boolean(model?.hasEvidence && recordId && model.after.id === String(recordId));
    }

    root.BlabbyProgressEvidence = { ALLOWED_STATUSES, modelFrom, eventProps, includesRecord };
})(typeof window !== 'undefined' ? window : globalThis);
