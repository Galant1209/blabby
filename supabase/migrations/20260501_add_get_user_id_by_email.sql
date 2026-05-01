-- Helper for the LemonSqueezy webhook: backend receives an email from
-- LS but profiles has no email column, so we need to look up auth.users.
-- security definer + EXECUTE revoke = service role can call it, anon/authed cannot.

create or replace function public.get_user_id_by_email(email_input text)
returns uuid
language sql
security definer
set search_path = public
as $$
  select id from auth.users where email = email_input limit 1;
$$;

revoke execute on function public.get_user_id_by_email(text) from public, anon, authenticated;
grant  execute on function public.get_user_id_by_email(text) to service_role;
