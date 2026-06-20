"""Retriever behavior tests around scope gating and exact-token fallback."""
from __future__ import annotations

from app.config import Settings
from app.retriever import Retriever


def test_in_scope_bread_question_survives_scope_gate():
    result = Retriever(Settings()).retrieve("What did the Victorians add to bread?")
    assert result.in_scope is True
    assert 2 <= len(result.hits) <= 3


def test_out_of_scope_question_still_abstains():
    result = Retriever(Settings()).retrieve("What is the capital of Australia?")
    assert result.in_scope is False
    assert result.hits == []
