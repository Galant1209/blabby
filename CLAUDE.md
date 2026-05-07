# Blabby — Claude Code 規範

## 產品定位

Blabby 是 AI IELTS 口說練習平台。已有早期真實使用者，第一輪回饋是「feels like one function, too simple」。

當前 P1：學生端 AI 診斷功能
當前 P2：弱點摘要視覺化

## 紅線（不要做的事）

這幾條是產品哲學，不是技術選擇。Claude 不要主動加入:

- 不做 streak（連續打卡）
- 不做 level / XP（等級系統）
- 不做 gamification（遊戲化、徽章、排行榜）
- 不做「鼓勵你回來」的推播文案

理由：Blabby 要像物理治療師，冷靜呈現事實，不是 Duolingo。

## 文案調性

- 像物理治療師，不像教練
- 描述事實，不下情緒判斷
- 「你的 fluency 在 6.0 區間，pronunciation 弱在 /θ/ /ð/」是對的
- 「你今天表現很棒！」是錯的

## Distribution > Engineering

- 5 分鐘的 distribution 動作勝過一整天的 engineering
- 寫新功能前先問：這個能讓我發一篇 Threads 嗎？不能就先停

## P1 功能規則（AI 診斷）

實作 AI 診斷時：

1. 先寫一個假輸入的測試，定義「成功的診斷輸出」長什麼樣
2. 再實作邏輯
3. 診斷輸出必須是具體可改進的點，不是整體評語
4. 輸出格式優先 structured JSON，方便 P2 視覺化串接

## 測試最低標

- 新增 API endpoint 必須有 happy path 測試
- 涉及 LLM 呼叫的要 mock，不要真的打 API

## Surgical Changes 補強

- 不要改 UI 文案，除非被明確要求
- 不要重構既有元件
- 不要加我沒提的優化
