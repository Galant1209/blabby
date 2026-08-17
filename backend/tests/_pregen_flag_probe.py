"""子行程 probe：在乾淨的行程裡跑完整 startup，回報 scheduler 上真正有哪些 job。

PREGEN_ENABLED 在 import 時求值一次，而 job 註冊發生在 FastAPI 的 startup
事件裡 —— 兩件事都無法在一個已經 import 過 main 的行程裡重新觀察。所以每個
案例都跑在自己的子行程。

兩個 pregeneration 函式被 patch 成 no-op：進 lifespan 會啟動 scheduler，而
startup prime 是 30 秒後觸發的 date job。測試不會等那麼久（TestClient 離開
時就 shutdown），但把真的會呼叫 Anthropic 的函式留在排程上，只為了測「它有
沒有被排程」，是不值得冒的險。
"""

import json
import os
import sys
from unittest.mock import patch

_BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from fastapi.testclient import TestClient                  # noqa: E402

import main                                                # noqa: E402

with patch.object(main, "pregenerate_writing_questions", lambda *a, **k: None), \
     patch.object(main, "pregenerate_reading_passages", lambda *a, **k: None):
    with TestClient(main.app):
        job_ids = sorted(job.id for job in main._scheduler.get_jobs())

print(json.dumps({
    "pregen_enabled": main.PREGEN_ENABLED,
    "job_ids":        job_ids,
}))
