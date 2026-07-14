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
