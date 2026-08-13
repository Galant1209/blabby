(function (root) {
    'use strict';

    const TAXONOMY = {
        weak_vocab: {
            label: '用字太安全',
            prescriptionTitle: '把更精確的詞說出口',
            evidence: '最近的練習仍出現用字過度安全、缺少具體畫面的情況。',
            nextAction: '做 2 題口說，每題刻意替換一個過度重複的簡單詞。',
        },
        safe_answer: {
            label: '答得太保險',
            prescriptionTitle: '把回答說得更具體',
            evidence: '最近的回答仍有立場不夠明確的情況。',
            nextAction: '做 2 題口說，每題先選一個明確立場，再補一個真實經驗。',
        },
        lack_detail: {
            label: '缺少具體細節',
            prescriptionTitle: '把理由說完整',
            evidence: '最近的回答仍缺少原因、人物、地點或時間等具體細節。',
            nextAction: '做 2 題口說，每題至少補一個具體原因或例子。',
        },
        grammar_minor: {
            label: '小文法',
            prescriptionTitle: '先把意思說完整',
            evidence: '最近的回答仍有影響句子穩定度的小文法問題。',
            nextAction: '做 2 題口說，先把意思說完整，再觀察同一個文法問題是否重複出現。',
        },
        off_topic: {
            label: '答非所問',
            prescriptionTitle: '先直接回答問題',
            evidence: '最近的回答仍有沒有直接接住題目核心的情況。',
            nextAction: '做 2 題口說，先用第一句直接回答問題，再補原因。',
        },
    };

    const FALLBACK = {
        label: '持續觀察中的口說問題',
        prescriptionTitle: '先把一個回答說完整',
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

    function prescriptionModel(record, dueCount) {
        const focus = focusFrom(record);
        if (focus) {
            const copy = TAXONOMY[focus.tag] || FALLBACK;
            return {
                type: 'speaking_focus',
                source: 'unresolved_weakness',
                title: copy.prescriptionTitle,
                description: copy.evidence,
                durationLabel: '約 4 分鐘',
                action: {
                    target: 'speaking',
                    targetLabel: 'Speaking',
                    recommendedQuestionCount: 2,
                    weaknessTag: focus.tag,
                    instruction: copy.nextAction,
                },
                cta: '繼續改善',
                focus,
                canResolve: true,
            };
        }

        const due = Math.max(0, Math.floor(Number(dueCount) || 0));
        if (due > 0) {
            return {
                type: 'vocabulary_review',
                source: 'vocabulary_due',
                title: '把查過的字重新拿回來用',
                description: due < 10
                    ? `有 ${due} 個你曾經存過的字現在適合複習。`
                    : '今天先複習幾個之前存過的字。',
                durationLabel: '約 2–4 分鐘',
                action: {
                    target: 'vocabulary',
                    targetLabel: 'Vocabulary',
                    recommendedQuestionCount: 0,
                    weaknessTag: '',
                    instruction: '從目前適合複習的單字開始。',
                },
                cta: '開始複習',
                focus: null,
                canResolve: false,
            };
        }

        return {
            type: 'speaking_baseline',
            source: 'insufficient_history',
            title: '先做一輪口說',
            description: '回答幾題後，Blabby 才能開始記住你反覆出現的問題。',
            durationLabel: '約 5 分鐘',
            action: {
                target: 'speaking',
                targetLabel: 'Speaking',
                recommendedQuestionCount: 3,
                weaknessTag: '',
                instruction: '先回答幾題，建立目前的口說觀察基準。',
            },
            cta: '開始口說',
            focus: null,
            canResolve: false,
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

    function prescriptionEventProps(model) {
        return {
            prescription_type: String(model?.type || '').slice(0, 64),
            source: String(model?.source || '').slice(0, 64),
            weakness_category: String(model?.action?.weaknessTag || '').slice(0, 64),
            target: String(model?.action?.target || '').slice(0, 32),
            recommended_question_count: Math.max(0, Math.floor(Number(model?.action?.recommendedQuestionCount) || 0)),
            authenticated: true,
        };
    }

    root.BlabbyRetention = {
        TAXONOMY,
        focusFrom,
        resumeModel,
        closureModel,
        prescriptionModel,
        eventProps,
        prescriptionEventProps,
    };
})(typeof window !== 'undefined' ? window : globalThis);
