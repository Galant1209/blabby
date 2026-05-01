-- Full admin users view: stats, conversion score, churn risk, waitlist flag.
-- Replaces ad-hoc python aggregation in /admin/users.
-- security definer + service_role-only execute = backend can call it,
-- anon/authenticated users cannot enumerate other users.

create or replace function public.get_admin_users_full()
returns table (
  user_id uuid,
  email text,
  is_pro boolean,
  created_at timestamptz,
  practice_count bigint,
  practice_count_7d bigint,
  drill_count bigint,
  last_practice timestamptz,
  main_weakness_tag text,
  is_waitlisted boolean,
  conversion_score int,
  churn_risk text
)
language sql
security definer
set search_path = public
as $$
  with base as (
    select
      p.id as user_id,
      u.email,
      p.is_pro,
      p.created_at,
      count(distinct pr.id) as practice_count,
      count(distinct pr.id) filter (where pr.created_at >= now() - interval '7 days') as practice_count_7d,
      count(distinct du.id) as drill_count,
      max(pr.created_at) as last_practice,
      mode() within group (order by pr.weakness_tag) filter (where pr.weakness_tag is not null) as main_weakness_tag
    from profiles p
    join auth.users u on u.id = p.id
    left join practice_records pr on pr.user_id = p.id
    left join drill_usage du on du.user_id = p.id
    group by p.id, u.email, p.is_pro, p.created_at
  ),
  waitlist as (
    select distinct lower(email) as email from upgrade_intent
  )
  select
    b.user_id,
    b.email,
    b.is_pro,
    b.created_at,
    b.practice_count,
    b.practice_count_7d,
    b.drill_count,
    b.last_practice,
    b.main_weakness_tag,
    (lower(b.email) in (select email from waitlist)) as is_waitlisted,
    (
      case when b.practice_count_7d >= 3 then 20 else 0 end +
      case when b.practice_count >= 10 then 15 else 0 end +
      case when b.drill_count > 0 then 20 else 0 end +
      case when b.main_weakness_tag is not null then 15 else 0 end +
      case when lower(b.email) in (select email from waitlist) then 30 else 0 end
    )::int as conversion_score,
    case
      when b.last_practice >= now() - interval '1 day' then 'low'
      when b.last_practice >= now() - interval '3 days' then 'medium'
      when b.practice_count >= 3 then 'high'
      when b.practice_count = 0 then 'new'
      else 'medium'
    end as churn_risk
  from base b
  order by conversion_score desc, last_practice desc nulls last;
$$;

revoke execute on function public.get_admin_users_full() from public, anon, authenticated;
grant  execute on function public.get_admin_users_full() to service_role;
