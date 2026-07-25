"""
Pytest setup for the Blabby backend tests.

The backend imports `from reading_prompts import ...` etc. as siblings; in
production main.py is the entry point and `backend/` is on the path. For
pytest we mirror that by prepending the parent directory to sys.path so
`reading_validator` and `reading_prompts` resolve.
"""

import os
import sys

_BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

# Unit tests patch ``main.supabase_admin`` at the call site and must never
# construct a real Supabase client during module import.  Explicit empty
# values also prevent python-dotenv from loading stale/invalid local secrets;
# load_dotenv() deliberately does not override variables already in os.environ.
os.environ["SUPABASE_URL"] = ""
os.environ["SUPABASE_SERVICE_KEY"] = ""

# main.py constructs all three provider clients at *import* time, not lazily, so
# a missing key breaks pytest during collection rather than inside a test — the
# whole module fails to import and its tests never run at all:
#
#   main.py:128  Groq(api_key=os.getenv("GROQ_API_KEY"))            -> GroqError
#   main.py:220  OpenAI(api_key=os.getenv("OPENAI_API_KEY"))        -> OpenAIError
#   main.py:130  anthropic.Anthropic(api_key=os.environ.get(..., ""))  safe today
#
# Anthropic is safe only because of that `, ""` default; it is pinned here anyway
# so the invariant lives in one place and a future `os.environ[...]` cannot
# silently reintroduce the same collection failure.
#
# setdefault, not assignment: a real key already in the environment still wins,
# which is what the credential-gated e2e tests need.
for _var in ("GROQ_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
    os.environ.setdefault(_var, "test-key")
