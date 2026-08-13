(function (root) {
    'use strict';

    const TAXONOMY = {
        weak_vocab: {
            label: '用字太安全',
            evidence: '最近的練習仍出現用字過度安全、缺少具體畫面的情況。',
            nextAction: '做 2 題口說，每題刻意替換一個過度重複的簡單詞。',
        },
        safe_answer: {
            label: '答得太保險',
            evidence: '最近的回答仍有立場不夠明確的情況。',
            nextAction: '做 2 題口說，每題先選一個明確立場，再補一個真實經驗。',
        },
        lack_detail: {
            label: '缺少具體細節',
            evidence: '最近的回答仍缺少原因、人物、地點或時間等具體細節。',
            nextAction: '做 2 題 Why 類口說，每題至少補一個具體原因或例子。',
        },
        grammar_minor: {
            label: '小文法',
            evidence: '最近的回答仍有影響句子穩定度的小文法問題。',
            nextAction: '重講 2 題，每題只修正一個最影響句意的小文法。',
        },
        off_topic: {
            label: '答非所問',
            evidence: '最近的回答仍有沒有直接接住題目核心的情況。',
            nextAction: '做 2 題口說，先用第一句直接回答問題，再補細節。',
        },
    };

    const FALLBACK = {
        label: '持續觀察中的口說問題',
        evidence: '最近的練習仍有一個問題需要繼續確認。',
        nextAction: '再做 2 題口說，我們會繼續確認這個問題是否仍出現。',
    };

    function focusFrom(record) {
        if (!record || record.resolved === true) return null;
        const tag = String(record.weakness_tag || '').trim();
        if (!tag || !record.id) return null;
        const copy = TAXONOMY[tag] || FALLBACK;
        return {
            id: String(record.id),
            tag,
            label: copy.label,
            status: '正在處理',
            evidence: copy.evidence,
            nextAction: {
                type: 'speaking',
                label: copy.nextAction,
                questionCount: 2,
            },
        };
    }

    function resumeModel(record) {
        const focus = focusFrom(record);
        if (focus) return { hasFocus: true, focus };
        return {
            hasFocus: false,
            title: '從下一輪開始建立續練線索',
            body: '開始一輪口說後，Blabby 會記住你反覆出現的問題，下一次直接從那裡繼續。',
            cta: '開始口說',
        };
    }

    function closureModel(feedback, answerCount) {
        const count = Math.max(0, Math.floor(Number(answerCount) || 0));
        const focus = feedback && feedback.persisted === true
            ? focusFrom({
                id: feedback.record_id,
                weakness_tag: feedback.weakness_tag,
                resolved: false,
            })
            : null;
        return {
            answerCount: count,
            today: `今天完成 ${count} 題 Speaking Part 1。`,
            focus,
            neutral: focus ? '' : '這一輪還沒有足夠的已保存資料可以設定下一個 focus。',
        };
    }

    function eventProps(tag, source, answerCount) {
        return {
            weakness_category: String(tag || '').slice(0, 64),
            source: String(source || 'authenticated_home').slice(0, 64),
            session_answer_count: Math.max(0, Math.floor(Number(answerCount) || 0)),
            authenticated: true,
        };
    }

    root.BlabbyRetention = { TAXONOMY, focusFrom, resumeModel, closureModel, eventProps };
})(typeof window !== 'undefined' ? window : globalThis);
