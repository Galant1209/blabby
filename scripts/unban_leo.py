import os
import sys
from supabase import create_client

SUPABASE_URL = os.environ.get('SUPABASE_URL')
SUPABASE_SERVICE_KEY = os.environ.get('SUPABASE_SERVICE_KEY')
LEO_ID = '5edc300e-1cc6-4b0f-96d0-a4f4b92641d7'

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    print("ERROR: Missing SUPABASE_URL or SUPABASE_SERVICE_KEY env vars")
    sys.exit(1)

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

print(f"Updating Leo's profile (id={LEO_ID})...")
result = supabase.table('profiles').update({
    'is_suspended': False,
    'auto_ban_strikes': 0,
    'suspended_at': None,
    'suspended_reason': None,
}).eq('id', LEO_ID).execute()
print(f"Update returned: {result.data}")

print("\nVerifying current state...")
verify = supabase.table('profiles').select(
    'id, is_suspended, auto_ban_strikes, suspended_at, suspended_reason'
).eq('id', LEO_ID).execute()
print(f"Current state: {verify.data}")

if verify.data and verify.data[0].get('is_suspended') is False:
    print("\n✅ Leo unbanned successfully")
else:
    print("\n❌ Still suspended — service_role key 也被擋了，需要進一步調查")
    sys.exit(2)
