"""
Regression tests for Bug A (2026-07-30): _extract_vocab_targets_haiku silently
returned [] whenever Haiku 4.5 wrapped its JSON array in ```json fences,
because — unlike every other Haiku call site in main.py — it never stripped
them before json.loads(). Confirmed via live API repro: two identical calls,
one fenced, one bare.
"""
import json
import logging
from types import SimpleNamespace
from unittest.mock import patch

import main


def _mock_response(text):
    return SimpleNamespace(content=[SimpleNamespace(text=text)])


def test_fenced_json_array_is_parsed_not_silently_dropped():
    """The exact failure mode: Haiku wraps the array in ```json fences."""
    fenced = '```json\n["migration", "breeding", "precision", "landmark", "geolocator", "meandering"]\n```'
    with patch.object(main.anthropic_client.messages, "create", return_value=_mock_response(fenced)):
        result = main._extract_vocab_targets_haiku("some passage text " * 20, 6.5)
    assert result == [
        "migration", "breeding", "precision", "landmark", "geolocator", "meandering",
    ]


def test_bare_json_array_still_parses():
    """Baseline: the un-fenced case must keep working after the fix."""
    bare = json.dumps(["urbanization", "ecosystems", "displacing", "drainage", "alter", "coastal"])
    with patch.object(main.anthropic_client.messages, "create", return_value=_mock_response(bare)):
        result = main._extract_vocab_targets_haiku("some passage text " * 20, 6.5)
    assert result == ["urbanization", "ecosystems", "displacing", "drainage", "alter", "coastal"]


def test_fenced_with_trailing_whitespace_and_no_json_hint_also_parses():
    """Some fences omit the 'json' language hint — must still strip cleanly."""
    fenced = '```\n["a1","b2","c3","d4","e5","f6"]\n```  '
    with patch.object(main.anthropic_client.messages, "create", return_value=_mock_response(fenced)):
        result = main._extract_vocab_targets_haiku("passage " * 20, 6.5)
    assert result == ["a1", "b2", "c3", "d4", "e5", "f6"]


def test_empty_response_returns_empty_list_and_logs_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="main"):
        with patch.object(main.anthropic_client.messages, "create", return_value=_mock_response("")):
            result = main._extract_vocab_targets_haiku("passage text", 6.5)
    assert result == []
    messages = [r.getMessage() for r in caplog.records]
    assert any("empty response" in m for m in messages)
    assert any("claude-haiku-4-5-20251001" in m for m in messages)


def test_unparseable_garbage_returns_empty_list_and_logs_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="main"):
        with patch.object(main.anthropic_client.messages, "create", return_value=_mock_response("not json at all")):
            result = main._extract_vocab_targets_haiku("passage text", 6.5)
    assert result == []
    messages = [r.getMessage() for r in caplog.records]
    assert any("parse failure" in m for m in messages)


def test_fewer_than_six_items_still_returned_but_logs_warning(caplog):
    fenced = '```json\n["a", "b", "c"]\n```'
    with caplog.at_level(logging.WARNING, logger="main"):
        with patch.object(main.anthropic_client.messages, "create", return_value=_mock_response(fenced)):
            result = main._extract_vocab_targets_haiku("passage text", 6.5)
    assert result == ["a", "b", "c"]
    messages = [r.getMessage() for r in caplog.records]
    assert any("only 3 items" in m for m in messages)
