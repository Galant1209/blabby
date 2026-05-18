-- Sprint Reading-1 polish: add reading_band_updated_at to profiles.
--
-- Mirrors profiles.band_updated_at which the Speaking module already
-- maintains. _update_user_band_reading() in main.py is updated in the same
-- patch to write this column whenever it touches user_band_reading.
--
-- Note: prompt 5 references this file's path as
-- blabby/backend/supabase/migrations/, but the actual project convention
-- (and every prior migration) lives at blabby/supabase/migrations/. Filed
-- there for consistency.

alter table public.profiles
    add column if not exists reading_band_updated_at timestamptz;
