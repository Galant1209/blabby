#!/usr/bin/env python3
"""ECPay production 商店存在性探針 —— 唯讀，不建立訂單。

問題：production 切過去之後，AioCheckOut 回 10200074（找不到商店）。從那個
錯誤本身分不出三件事：商店號在 production 根本不存在、金鑰是另一個環境的、
還是商店存在但 AIO 的其他條件沒滿足（例如信用卡收款未開通）。

做法：改用 QueryTradeInfo 去問。它是查詢 API，查一筆不存在的訂單不會有任何
副作用、不會產生扣款、不會寫入任何東西。但它會先驗商店與簽章 —— 所以它回
什麼錯誤，就把上面三件事分開了。

用法：
    export ECPAY_MERCHANT_ID=...
    export ECPAY_HASH_KEY=...
    export ECPAY_HASH_IV=...
    python3 backend/scripts/ecpay_probe.py

這支腳本不 import main.py，不碰資料庫，不呼叫 AioCheckOut。CheckMacValue
一律走 backend/ecpay.py 既有的實作 —— 探針同時也在驗那份實作。
"""

from __future__ import annotations

import os
import sys
import time

# ecpay.py 在上一層。production 跑的時候 backend/ 是 entry point 的所在目錄，
# 這裡手動鏡像同一件事。
_BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

import requests                                            # noqa: E402

import ecpay                                               # noqa: E402


QUERY_URLS = {
    "production": "https://payment.ecpay.com.tw/Cashier/QueryTradeInfo/V5",
    "stage":      "https://payment-stage.ecpay.com.tw/Cashier/QueryTradeInfo/V5",
}

TIMEOUT_SECONDS = 15
BAD_CHECK_MAC_VALUE = "0" * 64


def mask_merchant_id(value: str) -> str:
    return "****" + value[-3:] if len(value) >= 3 else "***"


def read_credentials() -> tuple[str, str, str]:
    """讀三個環境變數並 strip。缺任何一個就停，不猜。"""
    merchant_id = os.getenv("ECPAY_MERCHANT_ID") or ""
    hash_key = os.getenv("ECPAY_HASH_KEY") or ""
    hash_iv = os.getenv("ECPAY_HASH_IV") or ""

    missing = [
        name for name, value in (
            ("ECPAY_MERCHANT_ID", merchant_id),
            ("ECPAY_HASH_KEY", hash_key),
            ("ECPAY_HASH_IV", hash_iv),
        ) if not value.strip()
    ]
    if missing:
        print("缺少環境變數：" + ", ".join(missing))
        print("這支腳本不內建任何憑證，請先 export 再執行。")
        sys.exit(2)

    for name, value in (("ECPAY_MERCHANT_ID", merchant_id),
                        ("ECPAY_HASH_KEY", hash_key),
                        ("ECPAY_HASH_IV", hash_iv)):
        if value != value.strip():
            print(f"⚠  {name} 金鑰夾帶空白 —— 這是獨立問題，見待修清單")
            print(f"   （原始長度 {len(value)}，strip 後 {len(value.strip())}；"
                  f"探針以下一律使用 strip 後的值）")

    return merchant_id.strip(), hash_key.strip(), hash_iv.strip()


def probe(label: str, description: str, url: str, params: dict) -> str | None:
    """送一次 QueryTradeInfo，印出送出的參數與完整回應。回傳 body，失敗回 None。"""
    print()
    print("─" * 72)
    print(f"{label}  {description}")
    print(f"    endpoint         {url}")
    print(f"    MerchantID       {mask_merchant_id(params['MerchantID'])}")
    print(f"    MerchantTradeNo  {params['MerchantTradeNo']}")
    print(f"    TimeStamp        {params['TimeStamp']}")
    print(f"    CheckMacValue    {params['CheckMacValue'][:8]}…"
          f"（共 {len(params['CheckMacValue'])} 字元，只印前 8 碼）")

    try:
        response = requests.post(url, data=params, timeout=TIMEOUT_SECONDS)
    except requests.exceptions.Timeout:
        print(f"    ✗ 連線逾時（{TIMEOUT_SECONDS}s）—— 無法判定，非程式問題")
        return None
    except requests.exceptions.RequestException as exc:
        print(f"    ✗ 連線失敗：{type(exc).__name__}: {exc}")
        return None

    print(f"    HTTP status      {response.status_code}")
    print("    回應 body（完整）：")
    body = response.text
    for line in (body or "(空白)").splitlines() or ["(空白)"]:
        print(f"      {line}")
    return body


# ── 判讀 ─────────────────────────────────────────────────────────────────
# 綠界的錯誤字串沒有穩定的公開契約，所以比對是「盡力而為」：對不上就明說對不上，
# 讓人看原文，不編造結論。

def _matches(body: str | None, needles: tuple[str, ...]) -> bool:
    return bool(body) and any(needle in body for needle in needles)


NOT_FOUND_TRADE = ("查無此筆交易", "查無交易", "Invalid MerchantTradeNo",
                   "10200047", "Got Invalid")
BAD_MAC = ("CheckMacValue", "10200073", "10200052")
NO_MERCHANT = ("10200074", "MerchantID Error", "Invalid MerchantID",
               "查無此特店", "10100050")


def interpret(p1: str | None, p2: str | None, p3: str | None) -> None:
    print()
    print("═" * 72)
    print("判讀")
    print("═" * 72)

    p1_not_found = _matches(p1, NOT_FOUND_TRADE)
    p1_bad_mac = _matches(p1, BAD_MAC)
    p1_no_merchant = _matches(p1, NO_MERCHANT)
    p2_no_merchant = _matches(p2, NO_MERCHANT)
    p3_not_found = _matches(p3, NOT_FOUND_TRADE)

    if p1 is None:
        print("P1 沒有拿到回應（連線層失敗）。上面已印出原因，先修連線再重跑。")
        return

    if p1_no_merchant and p2_no_merchant:
        print("→ 商店在 production 不存在。")
        print("  P1 與 P2 都回「找不到商店」：無論簽章對錯都查不到這個商店號，")
        print("  代表問題在綠界後台，不在程式。停止 coding，去確認 production")
        print("  商店號是否真的開通、是否與後台顯示的一致。")
    elif p1_bad_mac and not p1_not_found:
        print("→ 商店存在，金鑰錯或夾帶空白。")
        print("  P1 過了商店這一關才卡在簽章。核對 Render 上的 HashKey / HashIV")
        print("  是否為 production 那一組，以及有沒有尾端換行。")
    elif p1_not_found:
        print("→ 商店存在 + 金鑰正確。")
        print("  P1 回的是「查無此筆交易」—— 商店與簽章都通過了，只是這筆刻意")
        print("  不存在的訂單號查不到，本來就該如此。")
        print("  所以 10200074 出在 AIO 的其他條件，多半是信用卡收款未開通。")
        print("  下一步去綠界後台確認該商店的信用卡收款狀態，不是改程式。")
    elif p3_not_found:
        print("→ 手上這組是 stage 憑證。")
        print("  P3 打 stage 成功而 P1 打 production 失敗，代表憑證屬於測試環境。")
    else:
        print("→ 無法自動判定。")
        print("  綠界回的字串不在已知的比對集合裡。請直接看上面三段的完整 body，")
        print("  照下表對照：")

    print()
    print("  對照表")
    print("  ┌────────────────────────────┬──────────────────────────────────┐")
    print("  │ P1 回「查無此筆交易」類訊息 │ 商店存在 + 金鑰正確，10200074 出 │")
    print("  │                            │ 在 AIO 其他條件（多半信用卡未開）│")
    print("  │ P1 回 CheckMacValue 錯誤    │ 商店存在，金鑰錯或夾帶空白       │")
    print("  │ P1 與 P2 都回「找不到商店」 │ 商店在 production 不存在，程式無 │")
    print("  │                            │ 關，停止 coding                  │")
    print("  │ P3 成功而 P1 失敗           │ 手上這組是 stage 憑證            │")
    print("  └────────────────────────────┴──────────────────────────────────┘")


def main() -> None:
    merchant_id, hash_key, hash_iv = read_credentials()

    # "PROBE" + epoch 毫秒 = 18 字元，在 MerchantTradeNo 的 20 字元上限內，
    # 且必然對不到任何真實訂單。
    trade_no = f"PROBE{int(time.time() * 1000)}"

    base = {
        "MerchantID":      merchant_id,
        "MerchantTradeNo": trade_no,
        "TimeStamp":       str(int(time.time())),
    }
    check_mac = ecpay.build_check_mac_value(base, hash_key, hash_iv)

    print("ECPay 商店存在性探針 —— QueryTradeInfo，唯讀，不建立訂單")
    print(f"MerchantID {mask_merchant_id(merchant_id)}"
          f"（長度 {len(merchant_id)}）")

    p1 = probe(
        "P1", "正確簽章 → production：商店與金鑰是否成立",
        QUERY_URLS["production"], dict(base, CheckMacValue=check_mac),
    )
    p2 = probe(
        "P2", "故意錯誤的簽章 → production：分辨「商店不存在」與「簽章錯」",
        QUERY_URLS["production"], dict(base, CheckMacValue=BAD_CHECK_MAC_VALUE),
    )
    p3 = probe(
        "P3", "同一組憑證 → stage：這組憑證是不是其實屬於測試環境",
        QUERY_URLS["stage"], dict(base, CheckMacValue=check_mac),
    )

    interpret(p1, p2, p3)


if __name__ == "__main__":
    main()
