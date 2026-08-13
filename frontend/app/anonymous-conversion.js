(function (root) {
    'use strict';

    const DEFAULT_LIMIT = 10;

    function clampCount(value, limit) {
        const parsed = Number(value);
        if (!Number.isFinite(parsed)) return 0;
        return Math.max(0, Math.min(Math.floor(parsed), limit));
    }

    function stateFor(usedValue, limitValue) {
        const limit = Math.max(1, Number(limitValue) || DEFAULT_LIMIT);
        const used = clampCount(usedValue, limit);
        const remaining = Math.max(limit - used, 0);
        const base = { limit, used, remaining, lockRecorder: false, card: 'none' };

        if (used === 0) {
            return { ...base, status: `可免費體驗 ${limit} 次口說，不需註冊` };
        }
        if (used === 5) {
            return {
                ...base,
                card: 'milestone_5',
                status: `剩餘 ${remaining} 次免費練習`,
                title: '已完成 5 次練習',
                body: '建立帳號後可開始保存你的練習紀錄。',
            };
        }
        if (used >= 8 && remaining > 0) {
            return {
                ...base,
                card: 'milestone_8',
                status: `剩餘 ${remaining} 次免費練習`,
                title: `剩餘 ${remaining} 次免費練習`,
                body: '建立帳號後可保存歷史，並追蹤反覆出現的弱點。',
            };
        }
        if (remaining === 0) {
            return {
                ...base,
                card: 'complete',
                lockRecorder: true,
                status: '免費體驗完成',
                title: '免費體驗完成',
                body: '建立帳號後，後續練習可以保存歷史、追蹤反覆出現的弱點，並使用完整個人學習功能。',
            };
        }
        return { ...base, status: `剩餘 ${remaining} 次免費練習` };
    }

    function eventProps(state, source, authenticated) {
        return {
            anonymous_used_count: clampCount(state && state.used, DEFAULT_LIMIT),
            remaining_count: clampCount(state && state.remaining, DEFAULT_LIMIT),
            source: String(source || 'speaking_part1').slice(0, 64),
            authenticated: authenticated === true,
        };
    }

    root.BlabbyAnonymousConversion = { stateFor, eventProps };
})(typeof window !== 'undefined' ? window : globalThis);
