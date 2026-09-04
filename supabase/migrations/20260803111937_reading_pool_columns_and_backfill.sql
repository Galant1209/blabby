-- reading pool: 池化欄位 + 分桶索引
ALTER TABLE reading_passages
  ADD COLUMN is_pregenerated boolean NOT NULL DEFAULT false,
  ADD COLUMN questions_ready boolean NOT NULL DEFAULT false,
  ADD COLUMN used_count integer NOT NULL DEFAULT 0;

CREATE INDEX idx_reading_passages_pool
  ON reading_passages (topic, is_pregenerated, questions_ready);

-- 回填：有完整題目的 passage 翻 questions_ready + 入池
-- （5 篇孤兒維持 false/false，自動排除）
UPDATE reading_passages p
SET questions_ready = true,
    is_pregenerated = true
WHERE EXISTS (
  SELECT 1 FROM reading_questions q WHERE q.passage_id = p.id
);
