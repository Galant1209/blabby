-- Roll back only the anonymous trial counter. No member or billing data touched.

BEGIN;

DROP FUNCTION IF EXISTS public.consume_anonymous_process_quota(text, text, integer);
DROP FUNCTION IF EXISTS public.get_anonymous_process_quota(text, text, integer);
DROP TABLE IF EXISTS public.anonymous_process_usage;

COMMIT;
