-- Aggregated user activity for the admin dashboard.
-- security definer + service_role-only execute = backend can call it,
-- anon/authenticated users cannot enumerate other users.

create or replace function public.get_admin_user_activity()
returns table (
  user_id uuid,
  is_pro boolean,
  practice_count bigint,
  drill_count bigint,
  last_practice timestamptz
)
language sql
security definer
set search_path = public
as $$
  select
    p.id as user_id,
    p.is_pro,
    count(distinct pr.id) as practice_count,
    count(distinct du.id) as drill_count,
    max(pr.created_at) as last_practice
  from profiles p
  left join practice_records pr on pr.user_id = p.id
  left join drill_usage du on du.user_id = p.id
  group by p.id, p.is_pro
  order by last_practice desc nulls last;
$$;

revoke execute on function public.get_admin_user_activity() from public, anon, authenticated;
grant  execute on function public.get_admin_user_activity() to service_role;
