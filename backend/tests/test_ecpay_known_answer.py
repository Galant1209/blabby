"""
綠界 CheckMacValue known-answer 向量。

來源：綠界 API 測試工具 /Cashier/AioCheckOut/V5，測試特店 3002607
取得日期：2026-07-30
HashKey / HashIV 為綠界公開之測試特店金鑰，非正式憑證。

這是第二個獨立向量。tests/test_ecpay.py 的向量取自官方文件說明頁
(developers.ecpay.com.tw/2902)；這一個取自實際 API 測試工具的輸出。兩者
來源不同、參數不同、雜湊不同，同時通過才足以排除「文件頁範例本身抄錯」
這種單點失效。

此向量涵蓋四個高風險路徑：
  1. ItemName 含中文 → UTF-8 percent-encode 與轉小寫的先後順序
  2. MerchantTradeDate 含 / : 空白 → 三種不同編碼規則
  3. ReturnURL 含 :// → URL 不被特殊處理
  4. 分隔符 & = 本身也被 encode（%26 / %3d）

未涵蓋：~ 的 .NET 對等性。CPython 的 quote_plus 把 ~ 留成字面值，
HttpUtility.UrlEncode 會轉成 %7E —— 綠界官方 Python SDK 有同樣的差異
（safe='-_.!*()' 是加到 _ALWAYS_SAFE 之上，不是取代），所以我們與綠界
自己的實作一致。TradeDesc / ItemName 為模組常數且限定 ASCII 字母數字
空白連字號，~ 不可能出現在我們送出的參數裡。
"""

from ecpay import build_check_mac_value, _ecpay_urlencode

TEST_HASH_KEY = "pwFHCqoQZGmho4w6"
TEST_HASH_IV = "EkRm7iFT261dpevs"

VECTOR_PARAMS = {
    "ChoosePayment": "ALL",
    "EncryptType": "1",
    "ItemName": "ABCD商品",
    "MerchantID": "3002607",
    "MerchantTradeDate": "2026/07/30 11:11:47",
    "MerchantTradeNo": "20260730NK7K11IZGCA1",
    "PaymentType": "aio",
    "ReturnURL": "https://www.ecpay.com.tw",
    "TotalAmount": "100",
    "TradeDesc": "Test",
}
VECTOR_EXPECTED = (
    "AAFD7EC1722966B2941F15BECA32F2348C7665C123DD1DE6A54DCA919C449DEF"
)


def test_known_answer():
    assert build_check_mac_value(
        VECTOR_PARAMS, TEST_HASH_KEY, TEST_HASH_IV
    ) == VECTOR_EXPECTED


def test_check_mac_value_field_is_excluded():
    polluted = dict(VECTOR_PARAMS, CheckMacValue="GARBAGE")
    assert build_check_mac_value(
        polluted, TEST_HASH_KEY, TEST_HASH_IV
    ) == VECTOR_EXPECTED


def test_separators_are_encoded():
    """分隔符必須被 encode —— 這是最容易寫錯的一格。"""
    assert _ecpay_urlencode("a=1&b=2") == "a%3d1%26b%3d2"


def test_dotnet_replacements():
    assert _ecpay_urlencode("!") == "!"
    assert _ecpay_urlencode("*") == "*"
    assert _ecpay_urlencode("(") == "("
    assert _ecpay_urlencode(")") == ")"
    assert _ecpay_urlencode("-_.") == "-_."


def test_space_becomes_plus_and_cjk_is_utf8():
    assert _ecpay_urlencode("a b") == "a+b"
    assert _ecpay_urlencode("商品") == "%e5%95%86%e5%93%81"


def test_non_string_values_are_coerced_not_crashed():
    """TotalAmount 常被當 int 傳進來；簽的必須是顯而易見的那個值。"""
    assert _ecpay_urlencode(100) == "100"


def test_existing_main_implementation_matches_vector():
    """
    8e11b93 的 _ecpay_check_mac_value 必須產出同一個雜湊。
    若此測試紅燈，既有實作是錯的，而在此之前沒有任何東西會發現。
    """
    from main import _ecpay_check_mac_value  # noqa: PLC0415
    assert _ecpay_check_mac_value(
        VECTOR_PARAMS, TEST_HASH_KEY, TEST_HASH_IV
    ) == VECTOR_EXPECTED
