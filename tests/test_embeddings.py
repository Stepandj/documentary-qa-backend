"""Embedding tests for the offline fallback path."""
from __future__ import annotations

import numpy as np

from app.config import Settings
from app.embeddings import EmbeddingClient


def test_local_fallback_embeddings_are_finite_and_normalized():
    client = EmbeddingClient(Settings())
    client._st_model = False  # force the zero-download fallback path

    vectors = client.embed(
        [
            "What did the Victorians add to bread?",
            "Annie Gray prepared three loaves to show the adulterants.",
        ]
    )

    assert vectors.shape == (2, 1536)
    assert np.isfinite(vectors).all()
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
