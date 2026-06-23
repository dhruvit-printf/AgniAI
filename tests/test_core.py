from unittest.mock import patch

import pytest

from config import (
    _fuzzy_normalize_query,
    classify_intent,
    detect_answer_style,
    trim_to_complete_sentence,
)
from ingest import chunk_text, clean_text
from rag import _meaningful_tokens, _normalise_text, make_response_cache_key


@pytest.fixture
def no_index_io():
    with patch("rag.load_docstore", return_value=[]), patch("rag.load_index"):
        yield


def test_classify_intent_greeting():
    assert classify_intent("hello") == "chat"


def test_classify_intent_rag_salary():
    assert classify_intent("what is the salary?") == "rag"


def test_classify_intent_reject():
    assert classify_intent("recommend me a pizza recipe") == "reject"


def test_fuzzy_normalize():
    assert "agniveer" in _fuzzy_normalize_query("agniver exam")


def test_detect_style_short():
    assert detect_answer_style("briefly explain")[0] == "short"


def test_detect_style_detail():
    assert detect_answer_style("explain in detail")[0] == "detail"


def test_chunk_text_basic():
    assert len(chunk_text("word " * 500)) > 1


def test_chunk_text_empty():
    assert chunk_text("") == []


def test_chunk_text_no_negative_start():
    chunks = chunk_text("word " * 1000)
    assert chunks
    assert all(isinstance(chunk, str) and chunk.strip() for chunk in chunks)


def test_clean_text():
    assert clean_text("  hello\x00\n\n\nworld  ") == "hello\n\nworld"


def test_normalise_text():
    assert _normalise_text("Hello  World") == "hello world"


def test_meaningful_tokens_removes_stopwords():
    assert "the" not in _meaningful_tokens("the quick brown fox")


def test_make_response_cache_key_deterministic():
    key1 = make_response_cache_key(
        "salary",
        style="short",
        model="mistral",
        context="Year 1 salary",
        session_id="abc",
    )
    key2 = make_response_cache_key(
        "salary",
        style="short",
        model="mistral",
        context="Year 1 salary",
        session_id="abc",
    )
    assert key1 == key2


def test_trim_to_complete_sentence():
    assert trim_to_complete_sentence("This is complete.") == "This is complete."
    # Added to verify the no-match branch behavior in trim_to_complete_sentence for both short and long texts without sentence boundary punctuation.
    assert trim_to_complete_sentence("short phrase") == "short phrase"
    assert trim_to_complete_sentence("this is a very long phrase with more than ten words in it and no punctuation") == "this is a very long phrase with more than ten words in it and no punctuation"
