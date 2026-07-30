# 綠界 stage 端到端驗證手冊

分支 `feat/ecpay-backend`。這份文件的目的是讓**任何人**照著執行都能得到相同結果，
不需要讀過實作。

不動 main、不部署到 Render、不執行任何 production migration 以外的東西。
路徑是「本機 uvicorn + cloudflared tunnel」，綠界只需要一個公開可達的 HTTPS
網址，不要求是正式網域。

---

## 0. 這輪驗證涵蓋什麼、不涵蓋什麼

**涵蓋**：CheckMacValue 簽章、AioCheckOut 參數集、callback 的 `1|OK` 回應、
冪等鍵擋重送、`RtnCode != 1` 不發權、`/api/payment/return` 導向、
SimulatePaid 在 production 不發權、CustomField1 交叉檢查。

**正式定價（Galant，2026-07-30）**：Blabby Pro 30 天費用為 **NT$199**。
`backend/ecpay.py` 的 `PRO_MONTHLY_TWD = 199` 是唯一 canonical price；
create-order 必須以此產生 `TradeAmt` 並寫入 subscription amount，client
不得傳入或覆寫金額。Callback 仍須比對該筆 subscription 已儲存的 amount。
除非另有產品決策，不加入折扣、早鳥價或其他價格分支。現存價格意向調查頁的
選項不是正式售價，也不是 ECPay checkout 的金額來源。

**3D Secure —— 修正一個先前的錯誤假設。**

先前判斷「stage 無法覆蓋 3D」。查證官方 /2856/ 後：**測試環境可以測 3D，
簡訊驗證碼固定為 `1234`，不需要收簡訊。**

所以 3D 路徑分成兩條，本手冊都涵蓋：

- **第 5 節**用「測試付款請點此」→ **跳過** 3D，最快驗證 callback 主線
- **第 6.5 節**用測試卡號走完整 3D → 可在 stage 就把成功、失敗、放棄三種
  情境全部演練過

仍然只有 production 能驗的是：真實發卡行的行為（逾時長度、拒絕理由、
放棄時是否真的完全不發 callback）。第 8 節不可省略，但它現在是「確認」
而不是「首次探索」。

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
    "CustomField1": "<你的 user_id>",
    "NeedExtraPaidInfo": "Y",
    "CheckMacValue": "..."
  },
  "merchant_trade_no": "20260730XXXXXXXXXXXX",
  "amount": 199
}
```

檢查點：
- `MerchantTradeNo` 恰好 20 字元、純大寫英數、前 8 碼是今天日期
- `TotalAmount` 是正式定價 `"199"`
- `action_url` 含 `payment-stage`
- 三個 URL 都是絕對 HTTPS
- `CustomField1` 等於你的 user_id
- **沒有**任何 `Period*` 參數（不做自動續訂）
- **沒有**任何 `Invoice*` / `InvoiceMark` 參數（Phase 1 不整合電子發票）

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
select status, expires_at, started_at, ecpay_trade_no from subscriptions
where merchant_trade_no = '20260730XXXXXXXXXXXX';
```
→ `status='active'`，`expires_at` = `started_at` + 30 天（誤差在秒級），
`ecpay_trade_no` 有值且與綠界後台顯示的交易編號相同。

**CustomField1 交叉檢查**：callback 會比對 `CustomField1`（下單時寫入的
user_id，受 CheckMacValue 保護）與 DB 中該筆訂單的 user_id。不一致會回 500
並印 CRITICAL，不發權。正常流程不會觸發。

### ⚠️ 關於「測試付款請點此」與 SimulatePaid

綠界的模擬付款會在 callback 帶 `SimulatePaid=1`，代表**零金流**。

- **stage（本節）**：允許發權，否則這份手冊無法驗證任何東西
- **production**：一律**不發權**，只記 `payment_events` 並印 CRITICAL

判斷依據是我們自己的 `ECPAY_ENV`，不是 callback 裡的任何欄位。所以正式上線
後在綠界後台按「模擬付款」**不會**讓帳號變成 Pro —— 那是刻意的，不是壞掉。
要驗證 production 請用第 8 節的真卡流程。

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

## 6.5 在 stage 演練 3D 三種結局

回到第 4 節重新 `create-order`（每次都要新的 MerchantTradeNo），這次在收銀台
**不要**按「測試付款請點此」，改用測試卡號手動輸入：

| 項目 | 值 |
|---|---|
| 卡號（國內） | `4311-9511-1111-1111` 或 `4311-9522-2222-2222` |
| 卡號（國外） | `4000-2011-1111-1111` |
| 到期日 | 任意未來日期 |
| CVV | 任意三碼 |
| **3D 簡訊驗證碼** | **`1234`**（測試環境固定值，不會真的發簡訊） |

來源：https://developers.ecpay.com.tw/?p=2856

依序演練三種結局，每種都用一筆新訂單：

**(a) 3D 通過** → 與第 5 節相同：`RtnCode='1'`、發權、`1|OK`。

**(b) 3D 驗證失敗**（輸入 `1234` 以外的碼直到被拒）
- callback **有**送達，`RtnCode != 1`（常見 `10100058`，與 3D 相關）
- 回應仍是 `1|OK` —— 已記錄的失敗不需要重送
- `payment_events` 1 列，`rtn_code` / `rtn_msg` 有實際內容
- `subscriptions` 該列**仍是 `pending`**，未被寫過
- 帳號**不是** Pro

**(c) 在 3D 頁面放棄**（直接關掉分頁）
- **可能完全沒有 callback** —— 這是綠界行為，不是 bug
- `payment_events` 0 列
- `subscriptions` 該列**永遠停在 `pending`**

(c) 是一個**已知且被接受的累積**。Phase 1 不做 pending 清理 job。
pending 列帶 `created_at`（DB default），要盤點時：

```sql
select merchant_trade_no, user_id, created_at
from subscriptions
where status = 'pending' and created_at < now() - interval '1 day'
order by created_at;
```

以目前量級（17 個使用者）這個累積無害。轉換量上來後再獨立開清理任務。

### 關於失敗代碼

不要試著列舉。綠界明說「錯誤代碼一直在新增」，完整清單只在廠商後台
（系統設定 → 交易狀態代碼查詢）。實作因此不列舉任何代碼 —— 只有 `RtnCode == "1"`
是成功，其餘一律原樣記進 `payment_events` 且不發權。新代碼不需要改程式。

### NeedExtraPaidInfo=Y 帶回什麼

下單時已設 `NeedExtraPaidInfo=Y`，成功的 callback 會多帶對帳欄位，全部進
`payment_events.raw_payload`（jsonb，零 schema 改動）：`eci`（3D 驗證結果
指標）、`auth_code`、`gwsr`、`process_date`、`card4no`、`card6no`、`amount`。

`eci` 是日後量測 3D 流失率的依據。這些欄位現在不讀，但沒收就只能靠再刷一筆
才拿得回來。

```sql
select raw_payload->>'eci', raw_payload->>'auth_code', raw_payload->>'card4no'
from payment_events where merchant_trade_no = '20260730XXXXXXXXXXXX';
```

⚠️ 官方明文：額外回傳的參數**全部都要納入檢查碼計算**。我們的驗簽是全欄位
納入、無白名單，所以多欄位不會破壞驗證 —— 但這也代表任何「只挑幾個欄位驗簽」
的最佳化都會直接掉單。

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

第 6.5 節已在 stage 演練過 3D 的三種結局，所以這一節是**確認真實發卡行的行為**，
不是首次探索。真銀行與測試環境會不同的地方：逾時長度、拒絕理由的措辭、
放棄時是否真的完全不發 callback。

### 切換方式

只改 4 個環境變數：`ECPAY_MERCHANT_ID` / `ECPAY_HASH_KEY` / `ECPAY_HASH_IV`
換正式值（廠商後台 → 系統設定 → 系統介接設定 → 介接資訊），`ECPAY_ENV` 改
`production`。**程式碼零改動、migration 零追加、前端零改動。**

附帶條件：綠界正式後台的 ReturnURL 白名單要指向正式 backend 網域。

### 8a. 成功路徑

真卡付款、完整通過 3D。預期與第 5 節相同。付完可在綠界後台辦理退刷。
額外確認 `raw_payload->>'eci'` 有值 —— 那證明走過 3D。

### 8b. 在 3D 頁面放棄

進到銀行 3D 驗證頁後**直接關掉分頁**。預期與第 6.5(c) 節相同：
可能零 callback、`subscriptions` 永遠 `pending`、帳號不是 Pro。

真實銀行與測試環境最可能不同的就是這一格 —— 記錄實際觀察到的行為。

### 8c. 3D 驗證失敗

輸入錯誤的驗證碼直到銀行拒絕。預期與第 6.5(b) 節相同。
把實際收到的 `rtn_code` 記下來，那是量測 3D 流失率的基礎資料。

### 8d. 失敗後重試

重新 `create-order`（新的 MerchantTradeNo），這次正常付款。
預期：正常發權，`payment_events` 共 2 列（不同單號），`subscriptions` 兩列
（舊的 pending、新的 active）。

### 8e. 確認 production 的模擬付款不發權

在**正式**後台按「模擬付款」。預期：
- callback 帶 `SimulatePaid=1`、`RtnCode=1`
- 回應 `1|OK`
- `payment_events` 有一列（看得見有人按過）
- `subscriptions` **未被啟用**，帳號**不是** Pro
- log 有一行 `CRITICAL [BILLING] SIMULATED payment refused outside stage`

這一格若通過了（帳號變 Pro），代表任何有後台權限的人都能免費開通 Pro，
**立刻停止上線**。

### 8f. 電子發票

Phase 1 不整合。首批付款後手動開立統一發票。
參數集裡不應出現任何 `Invoice*` 欄位 —— 若出現，代表有人「順手補齊」了，
必須重跑兩組 known-answer 向量驗證。
