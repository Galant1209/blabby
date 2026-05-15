# Blabby — 執行環境文件

## 產品定位
Blabby = AI IELTS 口說訓練平台。

目前階段：
- 已有早期真實使用者
- 主要回饋：
  「感覺像只有一個功能，太簡單了」

核心目標：
提升：
1. 使用者感知進步（Perceived Improvement）
2. 留存（Retention）
3. 轉換（Conversion）

不是增加功能數量。

---

# 優先序

## P1
學生端 AI 診斷功能

目標：
讓使用者立刻理解：
- 自己的口說弱點是什麼
- 為什麼會發生
- 要怎麼修正

必須輸出 structured JSON。

---

## P2
弱點摘要視覺化

依賴 P1 的 JSON 結構。

需要視覺化：
- 重複弱點 pattern
- 趨勢
- evidence

---

# 紅線

## 禁止新增
- streak
- XP
- level
- gamification
- 勵志型推播文案

---

## 文案調性
物理治療師。

描述：
- 可觀察的口說行為
- 可量化的弱點
- 修正路徑

不要：
- 情緒性鼓勵
- 泛用稱讚
- 像教練一樣說話

---

## 工程規則

### Surgical changes only
- 不做大型 refactor
- 不重寫架構
- 不做不必要 abstraction
- 沒有被要求時，不修改既有 copy

### Distribution > Engineering
新增任何功能前，先問：

「這能不能變成值得發 Threads 的畫面或 insight？」

如果不能：
降低優先級。

### 現有 UI 已足夠
優先：
- clarity
- feedback quality
- retention loops

不是：
- UI redesign
- 視覺重做

---

# 架構

## Frontend
blabby/frontend/app/index.html

規則：
- 所有 JS inline
- 所有 CSS inline
- 單檔架構
- 不要拆 module

---

## Backend
blabby/backend/main.py

Framework：
FastAPI

---

## 核心 Endpoint
POST /process

職責：
- Whisper transcription
- LLM response generation

同一流程完成。

---

## Storage
Supabase：
practice_records table

---

## LLM
run_claude()
main.py:938

---

# 已有功能

## Part 1
既有單題口說流程。

穩定。

除非明確要求，否則不要修改。

---

## Part 2
已實作：

- topic card
- 1 分鐘準備倒數
- 2 分鐘錄音
- Whisper transcription
- Claude evaluation
- Supabase persistence

### 題庫系統
backend/data/ielts_part2_topics.json

- 40 題
- 5 個 category
- 每類 8 題

Endpoint：
GET /part2/topics?category=

---

## 評分系統
POST /part2/evaluate

使用：
- 官方 IELTS band descriptors
- FC / LR / GRA / Pron

包含：
- anti-inflation scoring
- criterion-specific improvement actions

---

## 錯誤處理
已實作：
- 麥克風拒絕
- 錄音過短
- 網路失敗
- Claude JSON 異常

使用 modal-based UX。

---

## Fallback
如果評分失敗：
- 只顯示 transcript
- 不可 crash UI

---

# 測試最低標準

必要：
- 每個新 endpoint 都要有 happy-path test
- mock LLM calls
- 測試時不可真的打 API

---

# 決策規則

所有實作至少必須改善其中一項：

1. conversion
2. retention
3. perceived progress

如果三者都沒有：
不要做。

---

# Anti-Patterns

避免：
- over-engineering
- premature scaling
- generic dashboards
- decorative analytics
- enterprise abstractions
- unnecessary config systems
- React-style architecture patterns

這是一個 MVP revenue product。

速度與清晰度 > 純工程潔癖。

---

# Review Standard

所有重要實作都會被以下系統 review：
- ChatGPT
- Codex

程式品質、架構決策與 execution speed 會被交叉比較。

如果實作品質過低、過度 abstraction、執行過慢，或持續偏離產品方向，agent workflow 可能被替換。

因此：
- 優先 correctness
- 優先 execution speed
- 避免 over-engineering
- 避免 fake completeness
- 輸出 production-usable code
- Codex Will reivew your code
