-- Vocabulary seed — 30 items across 10 topics (3 per topic).
-- Columns inserted: word, part_of_speech, zh_meaning, difficulty_level,
-- ielts_band_level, topic, common_chunk, speaking_sentence, usage_note_zh.
-- Other columns (tags, better_than, common_mistake, etc.) left NULL —
-- safe to backfill later via UPDATE without re-seeding.

insert into vocabulary_items
  (word, part_of_speech, zh_meaning, difficulty_level, ielts_band_level, topic, common_chunk, speaking_sentence, usage_note_zh)
values
-- ── people (3) ───────────────────────────────────────────────
('considerate','adjective','體貼的','B1','6.0','people','considerate of others','My grandmother is incredibly considerate of everyone around her.','形容對他人感受敏感、設想周到，比 kind 更具體'),
('charismatic','adjective','有魅力的','B2','6.5','people','a charismatic personality','Our team leader is so charismatic that people naturally follow her ideas.','形容有領導魅力或吸引人的氣場，正式場合適用'),
('approachable','adjective','平易近人的','B2','6.5','people','an approachable manager','I like working with managers who are approachable and easy to talk to.','常用於工作或師長語境，比 friendly 更含「容易接近」的意思'),
-- ── place (3) ────────────────────────────────────────────────
('bustling','adjective','熙攘的、繁華的','B2','6.5','place','a bustling city','Taipei is a bustling city full of street food and night markets.','形容人多熱鬧的地方，比 busy 更有畫面感'),
('picturesque','adjective','風景如畫的','B2','6.5','place','picturesque scenery','The village we visited had picturesque scenery that looked like a postcard.','形容美得像畫的地方，旅遊話題常用'),
('vibrant','adjective','充滿活力的','B2','6.5','place','a vibrant atmosphere','The neighborhood has a vibrant atmosphere especially on weekends.','形容地方或氣氛有活力色彩，比 lively 更正式'),
-- ── experience (3, 1 given) ──────────────────────────────────
('memorable','adjective','難忘的','B1','6.0','experience','a memorable experience','It was one of the most memorable experiences I had because it was completely different from my routine.','適合描述旅行、活動、重要時刻'),
('unforgettable','adjective','難以忘懷的','B1','6.0','experience','an unforgettable trip','Visiting Kyoto in spring was an unforgettable trip for the whole family.','和 memorable 近義，但語氣更強，常用於正面回憶'),
('eye-opening','adjective','大開眼界的','B2','6.5','experience','an eye-opening experience','Working overseas was an eye-opening experience that changed how I see things.','形容讓人改變想法的經歷，比 interesting 更有重量'),
-- ── shopping (3) ─────────────────────────────────────────────
('bargain','noun','划算品、便宜貨','B1','6.0','shopping','a real bargain','I found a real bargain at the night market last weekend.','可指便宜好物，也可作動詞 bargain（討價還價）'),
('splurge','verb','揮霍、放縱花一筆','B2','6.5','shopping','splurge on something','Once in a while I splurge on a nice dinner with my friends.','形容偶爾為某物花較多錢，帶點享受意味'),
('durable','adjective','耐用的','B1','6.0','shopping','durable material','I prefer to buy durable shoes even if they cost a little more.','評論商品品質常用，比 strong 更精準'),
-- ── travel (3) ───────────────────────────────────────────────
('itinerary','noun','行程表','B2','6.5','travel','plan an itinerary','I always plan an itinerary before any long trip so we do not waste time.','旅遊話題高頻，比 plan 更專業'),
('off-the-beaten-track','adjective','偏離常規路線的','C1','7.0','travel','off-the-beaten-track destination','We chose an off-the-beaten-track destination to avoid the tourist crowds.','形容非熱門但有特色的地點，正式寫作和口說都常見'),
('venture','verb','冒險嘗試','B2','6.5','travel','venture into','I would love to venture into the mountains and try some real hiking.','形容嘗試陌生或具挑戰的環境，比 go 更有探索感'),
-- ── study (3, 1 given) ───────────────────────────────────────
('broaden','verb','拓展','B1','6.0','study','broaden my horizons','Studying abroad really helped me broaden my horizons.','常與 horizons, knowledge, perspective 搭配'),
('grasp','verb','理解、掌握','B1','6.0','study','grasp the concept','It took me a few classes to fully grasp the concept of statistics.','形容對抽象概念的理解，比 understand 更具體'),
('well-rounded','adjective','全面均衡的','B2','6.5','study','a well-rounded education','I think a well-rounded education matters more than just exam scores.','形容學習或人格各方面均衡發展'),
-- ── work (3, 2 given) ────────────────────────────────────────
('meticulous','adjective','一絲不苟的','C1','7.0','work','meticulous attention to detail','She is meticulous about her work which makes her stand out.','比 careful 更強調細節'),
('commute','verb/noun','通勤','B1','5.5','work','daily commute','My daily commute takes about an hour each way.','可作動詞也可作名詞'),
('versatile','adjective','多才多藝的、用途廣的','B2','6.5','work','a versatile employee','Companies value versatile employees who can handle different roles.','可形容人或工具，比 flexible 更有「能勝任多項」的意思'),
-- ── technology (3) ───────────────────────────────────────────
('cutting-edge','adjective','尖端的','B2','6.5','technology','cutting-edge technology','My company invests heavily in cutting-edge technology to stay ahead.','形容最先進的技術，比 advanced 更有畫面'),
('intuitive','adjective','直覺易上手的','B2','6.5','technology','an intuitive interface','The app has an intuitive interface that even my parents can use.','常用於評論 UX、產品設計'),
('tech-savvy','adjective','精通科技的','B2','6.5','technology','tech-savvy user','My younger brother is way more tech-savvy than I am.','形容某人擅長使用科技產品，輕鬆口語'),
-- ── emotion (3) ──────────────────────────────────────────────
('overwhelmed','adjective','不知所措、被壓垮的','B2','6.5','emotion','feel overwhelmed','I felt overwhelmed when I first started university because everything was new.','負面情緒，描述被事情壓得喘不過氣'),
('nostalgic','adjective','懷舊的','B2','6.5','emotion','feel nostalgic about','I always feel nostalgic about the summers I spent at my grandparents house.','形容對過去的溫暖回憶，正向懷念感'),
('frustrated','adjective','沮喪、受挫的','B1','6.0','emotion','deeply frustrated','I get deeply frustrated when I cannot find the right word in English.','負面情緒，常與工作、學習挫折搭配'),
-- ── hobby (3, 1 given) ───────────────────────────────────────
('spontaneous','adjective','自發的、隨興的','B2','6.5','hobby','a spontaneous decision','I am quite spontaneous when it comes to weekend plans.','形容沒有計畫的行動，比 unplanned 更正式'),
('fulfilling','adjective','充實有成就感的','B2','6.5','hobby','a fulfilling hobby','Painting is a fulfilling hobby that lets me switch off from work.','形容讓人滿足、有意義的活動'),
('dabble','verb','涉略、玩玩','B2','6.5','hobby','dabble in painting','I dabble in painting on weekends but I am still very much a beginner.','形容業餘嘗試某項興趣，不深入');
