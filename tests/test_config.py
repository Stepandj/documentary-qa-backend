"""Configuration validation tests."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config import Settings


def test_default_timeout_stays_within_30_second_budget():
    settings = Settings()
    assert settings.request_timeout < 30


def test_final_k_is_limited_to_spec_range():
    with pytest.raises(ValidationError):
        Settings(final_k=1)
    with pytest.raises(ValidationError):
        Settings(final_k=4)


def test_top_k_must_cover_final_k():
    with pytest.raises(ValidationError, match="TOP_K must be >= FINAL_K"):
        Settings(top_k=2, final_k=3)


def test_chunk_stride_must_not_exceed_window():
    with pytest.raises(ValidationError, match="CHUNK_STRIDE must be <= CHUNK_SEGMENTS"):
        Settings(chunk_segments=2, chunk_stride=3)
