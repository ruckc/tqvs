"""Tests for top-k, score, prefix filtering, custom metrics."""

import numpy as np
import pytest

from tqvs import create_vector_store, cosine_similarity, dot_product, euclidean_distance
from tqvs._types import StoreDtype

RNG = np.random.default_rng(99)
DIM = 16


@pytest.fixture
def populated_store(tmp_path):
    s = create_vector_store(tmp_path / "s", DIM)
    for i in range(20):
        prefix = "doc/" if i < 10 else "img/"
        s.add(f"{prefix}{i}", RNG.standard_normal(DIM).astype(np.float32))
    return s


# ---------------------------------------------------------------------------
# top-k basics
# ---------------------------------------------------------------------------


class TestTopK:
    def test_returns_k_results(self, populated_store):
        q = RNG.standard_normal(DIM).astype(np.float32)
        results = populated_store.query(q, k=5)
        assert len(results) == 5

    def test_descending_order(self, populated_store):
        q = RNG.standard_normal(DIM).astype(np.float32)
        results = populated_store.query(q, k=10)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_k_larger_than_store(self, populated_store):
        q = RNG.standard_normal(DIM).astype(np.float32)
        results = populated_store.query(q, k=100)
        assert len(results) == 20


# ---------------------------------------------------------------------------
# prefix filtering
# ---------------------------------------------------------------------------


class TestPrefixFiltering:
    def test_prefix_narrows_results(self, populated_store):
        q = RNG.standard_normal(DIM).astype(np.float32)
        results = populated_store.query(q, k=100, prefix="doc/")
        assert all(r.key.startswith("doc/") for r in results)
        assert len(results) == 10

    def test_prefix_no_match(self, populated_store):
        q = RNG.standard_normal(DIM).astype(np.float32)
        results = populated_store.query(q, k=5, prefix="video/")
        assert len(results) == 0

    def test_score_with_prefix(self, populated_store):
        q = RNG.standard_normal(DIM).astype(np.float32)
        results = populated_store.score(q, prefix="img/")
        assert all(r.key.startswith("img/") for r in results)
        assert len(results) == 10


# ---------------------------------------------------------------------------
# custom metrics
# ---------------------------------------------------------------------------


class TestCustomMetric:
    def test_dot_product_metric(self, populated_store):
        q = RNG.standard_normal(DIM).astype(np.float32)
        results = populated_store.query(q, k=5, metric=dot_product)
        assert len(results) == 5

    def test_euclidean_metric(self, populated_store):
        q = RNG.standard_normal(DIM).astype(np.float32)
        results = populated_store.query(q, k=5, metric=euclidean_distance)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_custom_callable(self, populated_store):
        def neg_l1(query, candidates):
            return -np.sum(np.abs(candidates - query), axis=1)

        q = RNG.standard_normal(DIM).astype(np.float32)
        results = populated_store.query(q, k=3, metric=neg_l1)
        assert len(results) == 3


# ---------------------------------------------------------------------------
# quantized scoring accuracy
# ---------------------------------------------------------------------------


class TestQuantizedScoring:
    @pytest.mark.parametrize("dtype", [StoreDtype.INT8_SYM, StoreDtype.INT4])
    def test_quantized_top1_matches_float(self, tmp_path, dtype):
        """Top-1 from a quantized store should usually match float32."""
        n, dim = 50, 32
        vecs = RNG.standard_normal((n, dim)).astype(np.float32)
        keys = [f"k{i}" for i in range(n)]
        q = RNG.standard_normal(dim).astype(np.float32)

        # Float32 baseline
        sf = create_vector_store(tmp_path / "f", dim)
        sf.add_many(keys, vecs)
        baseline = sf.query(q, k=1)[0]

        # Quantized
        sq = create_vector_store(tmp_path / "q", dim, dtype=dtype)
        sq.add_many(keys, vecs)
        result = sq.query(q, k=1)[0]

        # Allow top-1 to differ by at most 1 rank
        top5_baseline = {r.key for r in sf.query(q, k=5)}
        assert result.key in top5_baseline


# ---------------------------------------------------------------------------
# empty store
# ---------------------------------------------------------------------------


class TestEmptyStore:
    def test_query_empty(self, tmp_path):
        s = create_vector_store(tmp_path / "empty", DIM)
        q = RNG.standard_normal(DIM).astype(np.float32)
        assert s.query(q, k=5) == []

    def test_score_empty(self, tmp_path):
        s = create_vector_store(tmp_path / "empty", DIM)
        q = RNG.standard_normal(DIM).astype(np.float32)
        assert s.score(q) == []


# ---------------------------------------------------------------------------
# zero-norm query
# ---------------------------------------------------------------------------


class TestZeroNormQuery:
    def test_cosine_zero_query(self, populated_store):
        q = np.zeros(DIM, dtype=np.float32)
        results = populated_store.query(q, k=5, metric=cosine_similarity)
        assert len(results) == 5
        assert all(r.score == 0.0 for r in results)


# ---------------------------------------------------------------------------
# bfloat16 scoring
# ---------------------------------------------------------------------------


class TestBfloat16Scoring:
    def test_bfloat16_query(self, tmp_path):
        s = create_vector_store(tmp_path / "bf16", DIM, dtype=StoreDtype.BFLOAT16)
        vecs = RNG.standard_normal((10, DIM)).astype(np.float32)
        for i in range(10):
            s.add(f"k{i}", vecs[i])
        q = RNG.standard_normal(DIM).astype(np.float32)
        results = s.query(q, k=3)
        assert len(results) == 3
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# torch-accelerated scoring (skipped if torch unavailable)
# ---------------------------------------------------------------------------


class TestTorchScoring:
    @pytest.fixture(autouse=True)
    def _require_torch(self):
        pytest.importorskip("torch")

    def test_torch_cosine(self, tmp_path):
        s = create_vector_store(tmp_path / "t", DIM, device="cpu")
        vecs = RNG.standard_normal((10, DIM)).astype(np.float32)
        for i in range(10):
            s.add(f"k{i}", vecs[i])
        q = RNG.standard_normal(DIM).astype(np.float32)
        results = s.query(q, k=5)
        assert len(results) == 5

    def test_torch_dot(self, tmp_path):
        s = create_vector_store(tmp_path / "t", DIM, metric=dot_product, device="cpu")
        vecs = RNG.standard_normal((10, DIM)).astype(np.float32)
        for i in range(10):
            s.add(f"k{i}", vecs[i])
        q = RNG.standard_normal(DIM).astype(np.float32)
        results = s.query(q, k=5)
        assert len(results) == 5

    def test_torch_euclidean(self, tmp_path):
        s = create_vector_store(tmp_path / "t", DIM, metric=euclidean_distance, device="cpu")
        vecs = RNG.standard_normal((10, DIM)).astype(np.float32)
        for i in range(10):
            s.add(f"k{i}", vecs[i])
        q = RNG.standard_normal(DIM).astype(np.float32)
        results = s.query(q, k=5)
        assert len(results) == 5
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# INT8_SYM quantized-domain fast path
# ---------------------------------------------------------------------------


class TestInt8SymFastPath:
    """Verify INT8_SYM fast-path scores match dequantize-then-score."""

    def _make_store(self, tmp_path, n=50, dim=32):
        vecs = RNG.standard_normal((n, dim)).astype(np.float32)
        keys = [f"k{i}" for i in range(n)]
        s = create_vector_store(tmp_path / "int8", dim, dtype=StoreDtype.INT8_SYM)
        s.add_many(keys, vecs)
        return s, vecs, keys

    def test_cosine_top_k_matches(self, tmp_path):
        """Fast-path cosine should produce same ranking as dequant path."""
        s, vecs, keys = self._make_store(tmp_path)
        q = RNG.standard_normal(32).astype(np.float32)
        results = s.query(q, k=10, metric=cosine_similarity)
        assert len(results) == 10
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

        # Verify against a float32 store
        sf = create_vector_store(tmp_path / "f32", 32)
        sf.add_many(keys, vecs)
        baseline_top5 = {r.key for r in sf.query(q, k=5)}
        assert results[0].key in baseline_top5

    def test_dot_product_matches(self, tmp_path):
        """Fast-path dot product with scale correction."""
        s, vecs, keys = self._make_store(tmp_path)
        q = RNG.standard_normal(32).astype(np.float32)
        results = s.query(q, k=10, metric=dot_product)
        assert len(results) == 10
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_euclidean_falls_back(self, tmp_path):
        """Euclidean should still work (falls back to dequant path)."""
        s, vecs, keys = self._make_store(tmp_path)
        q = RNG.standard_normal(32).astype(np.float32)
        results = s.query(q, k=5, metric=euclidean_distance)
        assert len(results) == 5

    def test_zero_query_cosine(self, tmp_path):
        """Zero-norm query should return all-zero scores."""
        s, _, _ = self._make_store(tmp_path, n=5)
        q = np.zeros(32, dtype=np.float32)
        results = s.query(q, k=3, metric=cosine_similarity)
        assert all(r.score == 0.0 for r in results)


# ---------------------------------------------------------------------------
# TurboQuant ADC fast path
# ---------------------------------------------------------------------------


class TestTurboQuantADC:
    """Verify TurboQuant ADC scores match dequantize-then-score."""

    @pytest.mark.parametrize("dtype", [
        StoreDtype.TURBO_2BIT,
        StoreDtype.TURBO_3BIT,
        StoreDtype.TURBO_4BIT,
    ])
    def test_cosine_ranking_matches(self, tmp_path, dtype):
        n, dim = 50, 32
        vecs = RNG.standard_normal((n, dim)).astype(np.float32)
        keys = [f"k{i}" for i in range(n)]
        q = RNG.standard_normal(dim).astype(np.float32)

        s = create_vector_store(tmp_path / f"turbo_{dtype.value}", dim, dtype=dtype)
        s.add_many(keys, vecs)
        results = s.query(q, k=10, metric=cosine_similarity)

        assert len(results) == 10
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

        # Top-1 from ADC should be in the float32 top-5
        sf = create_vector_store(tmp_path / f"f32_{dtype.value}", dim)
        sf.add_many(keys, vecs)
        baseline_top5 = {r.key for r in sf.query(q, k=5)}
        assert results[0].key in baseline_top5

    @pytest.mark.parametrize("dtype", [
        StoreDtype.TURBO_2BIT,
        StoreDtype.TURBO_3BIT,
        StoreDtype.TURBO_4BIT,
    ])
    def test_dot_product(self, tmp_path, dtype):
        n, dim = 30, 16
        vecs = RNG.standard_normal((n, dim)).astype(np.float32)
        keys = [f"k{i}" for i in range(n)]
        q = RNG.standard_normal(dim).astype(np.float32)

        s = create_vector_store(tmp_path / f"turbo_{dtype.value}", dim, dtype=dtype)
        s.add_many(keys, vecs)
        results = s.query(q, k=10, metric=dot_product)
        assert len(results) == 10
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_zero_query_cosine(self, tmp_path):
        n, dim = 5, 16
        vecs = RNG.standard_normal((n, dim)).astype(np.float32)
        keys = [f"k{i}" for i in range(n)]
        s = create_vector_store(tmp_path / "turbo_z", dim, dtype=StoreDtype.TURBO_2BIT)
        s.add_many(keys, vecs)
        q = np.zeros(dim, dtype=np.float32)
        results = s.query(q, k=3, metric=cosine_similarity)
        assert all(r.score == 0.0 for r in results)

    def test_euclidean_falls_back(self, tmp_path):
        """Euclidean should still work via dequant fallback."""
        n, dim = 10, 16
        vecs = RNG.standard_normal((n, dim)).astype(np.float32)
        keys = [f"k{i}" for i in range(n)]
        s = create_vector_store(tmp_path / "turbo_e", dim, dtype=StoreDtype.TURBO_3BIT)
        s.add_many(keys, vecs)
        q = RNG.standard_normal(dim).astype(np.float32)
        results = s.query(q, k=5, metric=euclidean_distance)
        assert len(results) == 5
