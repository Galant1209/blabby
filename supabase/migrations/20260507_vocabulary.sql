-- Vocabulary system — three tables + RLS.
--   vocabulary_items        : shared catalog (publicly readable)
--   user_vocabulary         : per-user collection with SRS state
--   vocabulary_review_logs  : audit log of every flashcard review
--
-- RLS: catalog is public-read; user tables gated by auth.uid().

create table if not exists vocabulary_items (
  id uuid primary key default gen_random_uuid(),
  word text not null,
  part_of_speech text,
  zh_meaning text not null,
  difficulty_level text,
  ielts_band_level text,
  topic text,
  tags text[],
  simple_definition_en text,
  common_chunk text,
  speaking_sentence text,
  common_mistake text,
  better_than text[],
  usage_note_zh text,
  created_at timestamptz default now()
);

create table if not exists user_vocabulary (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  vocabulary_item_id uuid not null references vocabulary_items(id) on delete cascade,
  status text default 'new',
  srs_level int default 0,
  review_count int default 0,
  correct_count int default 0,
  wrong_count int default 0,
  last_reviewed_at timestamptz,
  next_review_at timestamptz default now(),
  source text,
  source_practice_record_id uuid,
  created_at timestamptz default now(),
  unique(user_id, vocabulary_item_id)
);

create table if not exists vocabulary_review_logs (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  user_vocabulary_id uuid not null references user_vocabulary(id) on delete cascade,
  vocabulary_item_id uuid not null references vocabulary_items(id) on delete cascade,
  review_type text not null,
  result text not null,
  previous_level int,
  new_level int,
  created_at timestamptz default now()
);

alter table vocabulary_items enable row level security;
alter table user_vocabulary enable row level security;
alter table vocabulary_review_logs enable row level security;

create policy "Anyone can read vocabulary_items"
  on vocabulary_items for select using (true);

create policy "Users manage own vocabulary"
  on user_vocabulary for all
  using (auth.uid() = user_id)
  with check (auth.uid() = user_id);

create policy "Users manage own review logs"
  on vocabulary_review_logs for all
  using (auth.uid() = user_id)
  with check (auth.uid() = user_id);
