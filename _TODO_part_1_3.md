# Part 1 / Part 3 實作規劃

**建立日期：** 2026-04-25
**狀態：** 擱置中（觸發條件未達）

---

## 為什麼今天不做

Part 1 / 2 / 3 是結構完全不同的三件事：

| | Part 1 | Part 2 | Part 3 |
|---|---|---|---|
| 題型 | 短問短答（4-5 句） | 長獨白（2 分鐘） | 深度討論（來回） |
| Recording flow | 不同 | 現有基準 | 不同 |
| 批改 prompt | 全部重寫 | 現有基準 | 全部重寫 |
| 預備時間 UI | 無 | 1 分鐘 prep | 無 |

一次做三個 = 重構整個 recording flow + 重寫所有批改 prompt。
違反「不動 recording flow」紅線。

---

## 觸發條件（任一達成才開始做）

1. 付費使用者明確 request Part 1 / 3 練習
2. Part 2 體驗已極致（批改品質穩定、題庫 ≥ 50 題、retention 數據健康）
3. 競品推出同類功能造成流失壓力

---

## 屆時的實作路徑

### Schema（已就緒）
`questions.part` 欄位已建，目前全部填 `2`。
新增 Part 1 / 3 題目只需 INSERT，不需 migration。

### Backend
- `GET /api/questions/next` 加 `?part=1` / `?part=3` query param
- 批改 endpoint 依 part 參數切換 prompt template
- Part 1 需要新的 prompt（短答評估邏輯與 Part 2 不同）

### Frontend
- Recording flow Part 1：移除 1 分鐘 prep timer
- Recording flow Part 3：改為對話模式（AI 問 → 用戶答 → AI 追問）
- Hub Daily Question：`Part 2 · {topic}` 標籤改為動態顯示 part

### 題庫
- Part 1：至少 30 題（覆蓋 work / study / hometown / daily life）
- Part 3：與 Part 2 題目配對（同 topic 延伸討論）

---

## 不要做的事

- 不要為了「完整性」提前做 Part 1 / 3
- 不要在 Part 2 體驗穩定前分散注意力
- Part 3 對話模式複雜度高，單獨評估，不要跟 Part 1 綁在一起做
