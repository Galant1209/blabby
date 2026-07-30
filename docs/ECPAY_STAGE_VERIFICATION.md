# 綠界 stage 端到端驗證手冊

分支 `feat/ecpay-backend`。這份文件的目的是讓**任何人**照著執行都能得到相同結果，
不需要讀過實作。

不動 main、不部署到 Render、不執行任何 production migration 以外的東西。
路徑是「本機 uvicorn + cloudflared tunnel」，綠界只需要一個公開可達的 HTTPS
網址，不要求是正式網域。

---

## 0. 這輪驗證涵蓋什麼、不涵蓋什麼

**涵蓋**：CheckMacValue 簽章、AioCheckOut 參數集、callback 的 `1|OK` 回應、
冪等鍵擋重送、`RtnCode != 1` 不發權、`/api/payment/return` 導向。

**不涵蓋 —— 3D Secure。**

正式環境已於 2026-07-30 啟用 3D 驗證。綠界測試收銀台的「測試付款請點此」
**會跳過 3D**，所以本輪走完全綠也**不代表** 3D 路徑正確。

3D 帶來的兩件事本輪都測不到：

1. 使用者在銀行 3D 頁面**放棄** → 綠界可能完全不發 callback
2. 3D 驗證**失敗** → `RtnCode != 1`，而且在 production 是常態而非邊緣

第 8 節有 production 首次真卡驗證的步驟，那一節不可省略。

---

## 1. 前置

必須已完成：

- migration `20260730_ecpay_backend.sql` 的 §1、§2 已對 production 執行
  （§1 不做的話，**每一筆真實 callback 都會撞 CHECK 違反並回 500**）
- 綠界測試特店的 MerchantID / HashKey / HashIV 在手
- 一個能登入 Blabby 的帳號，記下它的 `auth.users.id`（下面稱 `<GALANT_UUID>`）

安裝 tunnel：

```bash
brew install cloudflared
```

---

## 2. 啟動順序（先 tunnel，再 uvicorn）

順序不可對調。`PUBLIC_BACKEND_URL` 必須是 tunnel 的網址，而網址要等 tunnel
起來才知道。

**終端機 A —— 先開這個：**

```bash
cloudflared tunnel --url http://localhost:8000
```

輸出中會有一行類似 `https://xxxx-yyyy-zzzz.trycloudflare.com`。整段複製下來，
下面稱 `<TUNNEL_URL>`。這個網址在 cloudflared 關掉前有效，重開會換一個。

**終端機 B —— 填完變數再執行：**

```bash
cd /Users/yichengchiu/Desktop/Blabby/blabby/backend && APP_ENV=development SUPABASE_URL= SUPABASE_SERVICE_KEY= GROQ_API_KEY= OPENAI_API_KEY= ANTHROPIC_API_KEY= ECPAY_ENV=stage ECPAY_MERCHANT_ID= ECPAY_HASH_KEY= ECPAY_HASH_IV= PUBLIC_BACKEND_URL= PUBLIC_FRONTEND_URL= ./venv/bin/uvicorn main:app --port 8000
```

| 變數 | 填什麼 |
|---|---|
| `APP_ENV` | `development`（已填。這樣才不用 `EXPECTED_SUPABASE_PROJECT_REF`） |
| `SUPABASE_URL` | production Supabase URL |
| `SUPABASE_SERVICE_KEY` | production service role key |
| `GROQ_API_KEY` / `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | 任意非空字串即可，本輪用不到 |
| `ECPAY_ENV` | `stage`（已填。**不要**填 `prod`，只接受 `stage` / `production`） |
| `ECPAY_MERCHANT_ID` | 綠界測試特店 MerchantID |
| `ECPAY_HASH_KEY` | 綠界測試特店 HashKey |
| `ECPAY_HASH_IV` | 綠界測試特店 HashIV |
| `PUBLIC_BACKEND_URL` | `<TUNNEL_URL>`，不加結尾斜線 |
| `PUBLIC_FRONTEND_URL` | 你的前端網域，例如 `https://blabby.app` |

⚠️ 這一步會直接寫 **production Supabase**。這是刻意的：要驗的就是真實的
`payment_events` 冪等鍵。目前 `subscriptions` 與 `payment_events` 都是 0 列，
清理方式見第 7 節。

---

## 3. 起動自檢

```bash
curl -s https://<TUNNEL_URL>/health
```

預期：

```json
{"status":"ok","timestamp":"...","billing_config_ok":true}
```

`billing_config_ok` 若是 `false`，**停下來**。終端機 B 會有一行 CRITICAL：

```
[BILLING] ECPay is NOT configured — /api/payment/* will return 503: ...
```

訊息會說明是 `ECPAY_ENV is not set`、`ECPAY_ENV must be 'stage' or 'production'`
還是 `ECPAY_MERCHANT_ID is not set`。修好變數重啟即可。此時其他功能（Part 1 /
Part 2 / Reading / Writing）都仍正常，只有金流是死的。

---

## 4. 建立訂單

需要一個有效的 Supabase JWT。從瀏覽器登入 Blabby 後，在 DevTools Console：

```
JSON.parse(localStorage.getItem(Object.keys(localStorage).find(k => k.endsWith('-auth-token')))).access_token
```

```bash
curl -s -X POST https://<TUNNEL_URL>/api/payment/create-order -H "Authorization: Bearer <JWT>"
```

預期回傳：

```json
{
  "action_url": "https://payment-stage.ecpay.com.tw/Cashier/AioCheckOut/V5",
  "params": {
    "MerchantID": "...",
    "MerchantTradeNo": "20260730XXXXXXXXXXXX",
    "MerchantTradeDate": "2026/07/30 ...",
    "PaymentType": "aio",
    "TotalAmount": "199",
    "TradeDesc": "Subscription to the Blabby Pro Membership for a Term of Thirty Days",
    "ItemName": "Blabby Pro Membership - Thirty Days",
    "ReturnURL": "https://<TUNNEL_URL>/api/payment/callback",
    "ChoosePayment": "Credit",
    "ClientBackURL": "https://<前端>/success.html",
    "OrderResultURL": "https://<TUNNEL_URL>/api/payment/return",
    "EncryptType": "1",
    "CheckMacValue": "..."
  },
  "merchant_trade_no": "20260730XXXXXXXXXXXX",
  "amount": 199
}
```

檢查點：
- `MerchantTradeNo` 恰好 20 字元、純大寫英數、前 8 碼是今天日期
- `TotalAmount` 是 `"199"`（不是 299）
- `action_url` 含 `payment-stage`
- 三個 URL 都是絕對 HTTPS

此時 DB 應多一列：

```sql
select merchant_trade_no, status, amount, created_at
from subscriptions where user_id = '<GALANT_UUID>';
```
→ 1 列，`status='pending'`、`amount=199`、`created_at` 有值。

---

## 5. 付款

把上一步的 `params` 存成一個 HTML 檔（本機開啟即可，不用部署）：

```html
<form id="f" method="post" action="貼上 action_url">
  <input type="hidden" name="MerchantID" value="...">
  <!-- params 裡每一個 key 都要一個 input，CheckMacValue 也要 -->
</form>
<script>document.getElementById('f').submit()</script>
```

會跳到綠界測試收銀台。**畫面上有「測試付款請點此」按鈕，點它即可** ——
不需要真實卡號、不需要 3D 驗證。

### 預期觀察值

**終端機 B** 應出現：
```
[BILLING] activated mtn=20260730XXXXXXXXXXXX user_id=<GALANT_UUID> expires=2026-08-29T...
```

**綠界送來的 callback** 收到的是純文字 `1|OK`（4 bytes，`text/plain`）。
綠界後台該筆交易的通知狀態應為成功；若顯示失敗，代表回應 body 不對，停下來。

**payment_events**：
```sql
select source, merchant_trade_no, rtn_code, checkmac_valid, processed_at
from payment_events where merchant_trade_no = '20260730XXXXXXXXXXXX';
```
→ **恰好 1 列**，`source='ecpay_callback'`、`rtn_code='1'`、
`checkmac_valid=true`、`processed_at` 非 NULL。

**subscriptions**：
```sql
select status, expires_at, started_at from subscriptions
where merchant_trade_no = '20260730XXXXXXXXXXXX';
```
→ `status='active'`，`expires_at` = `started_at` + 30 天（誤差在秒級）。

**瀏覽器**：應被 303 導到 `https://<前端>/success.html?order=20260730XXXXXXXXXXXX`。
success.html 屬於 TASK 6B，此時大概率是 404 —— **這是預期的**，
只要網址列是對的就算通過。

**Pro 狀態**（只有在 `20260726 §2` 已執行後才會是 true）：
```sql
select is_user_pro('<GALANT_UUID>');
```

---

## 6. 重送驗證（最重要的一格）

綠界後台 → 該筆交易 → 「重新發送付款通知」。連按 3 次。

預期：
- 每一次都回 `1|OK`
- `payment_events` **仍然只有 1 列**（冪等鍵擋下）
- `subscriptions.updated_at` **不變**
- 終端機 B 出現 `[BILLING] duplicate callback mtn=...`

```sql
select count(*) from payment_events
where merchant_trade_no = '20260730XXXXXXXXXXXX';
```
→ 必須是 `1`。若大於 1，冪等鍵沒生效，停下來。

---

## 7. 清理

```sql
delete from subscriptions where user_id = '<GALANT_UUID>';
```

⚠️ **不要用 `where merchant_trade_no like 'BLB%'`。** MerchantTradeNo 的格式是
`yyyyMMdd` + 12 位隨機英數，沒有 `BLB` 前綴，那個條件會匹配 0 列而讓你以為
清乾淨了。

`payment_events` **不要刪，也刪不掉** —— `payment_events_immutable()` trigger
禁止 DELETE。那是稽核帳本，測試列留著是正確的。

---

## 8. production 首次真卡驗證（不可省略）

第 2–7 節在 stage 全綠**不代表 3D 路徑正確**。切到 production 後必須用真卡再跑一次。

切換方式：只改 4 個環境變數 —— `ECPAY_MERCHANT_ID` / `ECPAY_HASH_KEY` /
`ECPAY_HASH_IV` 換正式值，`ECPAY_ENV` 改 `production`。程式碼零改動。

### 8a. 成功路徑

真卡付款，完整通過 3D。預期與第 5 節相同。付完可在綠界後台辦理退刷。

### 8b. 在 3D 頁面**放棄**（刻意測這個）

進到銀行 3D 驗證頁後，**直接關掉分頁**，不要輸入驗證碼。

預期：
- 終端機/Render log **可能完全沒有任何 callback** —— 這是綠界的行為，不是 bug
- `payment_events` **0 列**（沒有 callback 就沒有事件）
- `subscriptions` 該列**永遠停在 `status='pending'`**
- 該帳號**不會**變成 Pro

這是一個**已知且被接受的累積**：Phase 1 不做 pending 清理 job。
pending 列帶 `created_at`（DB default），要盤點時：

```sql
select merchant_trade_no, user_id, created_at
from subscriptions
where status = 'pending' and created_at < now() - interval '1 day'
order by created_at;
```

以目前的量級（17 個使用者）這個累積無害。若日後轉換量上來，再獨立開一個
清理任務，不要塞進本輪。

### 8c. 3D 驗證**失敗**

輸入錯誤的 3D 驗證碼直到銀行拒絕。

預期：
- callback **有**送達，`RtnCode != 1`
- 回應仍是 `1|OK`（已記錄的失敗不需要重送）
- `payment_events` 1 列，`rtn_code` 是實際的失敗代碼、`rtn_msg` 有內容
- `subscriptions` 該列**仍是 `pending`**，`status` 沒有被寫過
- 該帳號**不是** Pro

把實際收到的 `rtn_code` 記下來 —— 那是量測 3D 流失率的基礎資料。

### 8d. 失敗後重試

8c 之後，重新 `create-order`（會產生**新的** MerchantTradeNo），這次正常付款。

預期：正常發權，`payment_events` 共 2 列（失敗一列、成功一列，不同單號），
`subscriptions` 有兩列（舊的 pending、新的 active）。
