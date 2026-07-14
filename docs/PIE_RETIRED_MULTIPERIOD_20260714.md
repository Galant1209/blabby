# Pie 多期題退池記錄 — 2026-07-14

**可逆操作。** 這 11 個 pie 題是**合法的多期圓餅題**（跨年度比較），資料完好，
只是暫時退出 pool（`is_pregenerated = false`），因為 `80b46d3` 的 deterministic
pie renderer 只支援單一 pie（`PieChartData` 結構上只有一組 values），多期題會落到
純文字 fallback（69% 的 in-pool pie 不出圖）。

**沒有 DELETE。只改了 `is_pregenerated` 一個欄位。** 其餘欄位一字未動。

## 解封條件

**多期 pie deterministic renderer 完成後**，執行本文件末的 rollback SQL 讓這 11 題重新入池。

## 退池清單（11 個 UUID + chart_description）

| # | id | chart_description |
|---|---|---|
| 1 | `491a28c2-4bfa-47f3-8e3b-484695df73d2` | `Energy Source \| 2010 \| 2023` / Coal 28→12, Natural Gas 32→28, Renewable 15→42, Nuclear 20→14, Oil 5→4 |
| 2 | `3d8d7bc5-915f-4ca2-b343-7fab5461d4fc` | `Energy Source \| 2010 \| 2023` / Coal 35→22, Natural Gas 28→32, Renewable 18→36, Nuclear 12→7, Oil 7→3 |
| 3 | `ab70b2fd-b96b-435b-81ca-ee7864be2c10` | `Energy Source \| 2000 \| 2010 \| 2020` / Coal 32/24/12, Natural Gas 28/31/28, Renewable 8/12/35, Nuclear 18/18/15, Oil 14/15/10 |
| 4 | `6b3c1090-fca5-48c3-886f-92d25195fb69` | `Energy Source \| 2010 \| 2023` / Natural Gas 42→38, Electricity 35→44, Oil 15→12, Renewable 8→6 |
| 5 | `85f9e3b7-8695-467c-9cf5-402324f7e6da` | `Energy Source \| 2010 \| 2023` / Natural Gas 42→35, Electricity 28→32, Renewable 12→22, Oil 15→9, Other 3→2 |
| 6 | `642acea1-339d-4f75-a0f0-1eb25afdd32a` | `Energy Source \| 2010 \| 2023` / Coal 32→18, Natural Gas 28→26, Renewable 12→38, Nuclear 20→12, Oil 8→6 |
| 7 | `2e0a374b-68d2-458f-adca-d5a4a45da0c7` | `Sector \| 2010 \| 2023` / Industrial 32→28, Residential 26→31, Transport 25→22, Commercial 17→19 |
| 8 | `0727e5e7-2749-4a35-8922-92ad99181a2f` | `Energy Source \| 2010 \| 2023` / Coal 28→12, Natural Gas 32→26, Renewable 18→42, Nuclear 15→13, Oil 7→7 |
| 9 | `ed6a48ea-a177-4661-929c-df51b97e523e` | `Sector \| 2010 \| 2023` / Industrial 32→28, Residential 27→31, Transport 24→26, Commercial 17→15 |
| 10 | `ae202b45-7ad9-4622-8132-b0bd749ebdf6` | `Energy Source \| 2010 \| 2023` / Coal 28→12, Natural Gas 24→28, Nuclear 19→22, Renewable 18→32, Oil 11→6 |
| 11 | `a4665e22-f250-42c7-9f2f-4c6a77ea2d9c` | `Energy Source \| 2010 \| 2023` / Coal 28→15, Natural Gas 24→26, Renewable 12→31, Nuclear 19→18, Oil 17→10 |

判定方式：以 backend 實際的 `parse_legacy_chart_description()` 函式逐筆執行，回傳 `None`
（多期 / 非 2 欄）者退池。非啟發式猜測。

## Rollback SQL（解封時執行）

```sql
UPDATE writing_questions
SET is_pregenerated = true
WHERE id IN (
  '491a28c2-4bfa-47f3-8e3b-484695df73d2',
  '3d8d7bc5-915f-4ca2-b343-7fab5461d4fc',
  'ab70b2fd-b96b-435b-81ca-ee7864be2c10',
  '6b3c1090-fca5-48c3-886f-92d25195fb69',
  '85f9e3b7-8695-467c-9cf5-402324f7e6da',
  '642acea1-339d-4f75-a0f0-1eb25afdd32a',
  '2e0a374b-68d2-458f-adca-d5a4a45da0c7',
  '0727e5e7-2749-4a35-8922-92ad99181a2f',
  'ed6a48ea-a177-4661-929c-df51b97e523e',
  'ae202b45-7ad9-4622-8132-b0bd749ebdf6',
  'a4665e22-f250-42c7-9f2f-4c6a77ea2d9c'
);
```

## 退池後狀態（2026-07-14 執行當下）

- in-pool pie：16 → **5**（全部可反解成單一 pie，5/5 會出圖）
- 退池 11 題：資料完好（`retired_11_still_exist = 11`），僅 `is_pregenerated=false`
- writing_questions 總數：92（未刪任何列）
- Pool 補充：`_pregenerate_task1_subtype(target=8)` 會把 pie 從 5 補到 8（needed=3），
  由 backend 啟動 pregen + 6 小時 APScheduler 觸發；新 pregen 產出保證是單一 pie
  （`PieChartData` 結構上無法表達多期）。
