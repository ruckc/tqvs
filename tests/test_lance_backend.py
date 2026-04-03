"""Tests for LanceBackend — save / load round-trips across all LoadModes."""

lance = __import__("pytest").importorskip("lance")

import numpy as np
import pytest

from tqvs._types import LoadMode, StoreDtype
from tqvs.backends.lance import LanceBackend


@pytest.fixture
def backend() -> LanceBackend:
    return LanceBackend()


@pytest.fixture
def store_path(tmp_path):
    return tmp_path / "test_store"


def _make_data(n: int = 10, dim: int = 32):
    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((n, dim)).astype(np.float32)
    keys = [f"key_{i}" for i in range(n)]
    manifest = {"dim": dim, "dtype": StoreDtype.FLOAT32.value, "keys": keys}
    metadata = {f"key_{i}": {"i": i} for i in range(n)}
    return vectors, None, manifest, metadata


# ---------------------------------------------------------------------------
# Basic round-trip
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def test_exists_false_initially(self, backend, store_path):
        assert not backend.exists(store_path)

    def test_save_then_exists(self, backend, store_path):
        vecs, qp, manifest, meta = _make_data()
        backend.save(store_path, vecs, qp, manifest, meta)
        assert backend.exists(store_path)

    @pytest.mark.parametrize("mode", list(LoadMode))
    def test_round_trip(self, backend, store_path, mode):
        vecs, qp, manifest, meta = _make_data()
        backend.save(store_path, vecs, qp, manifest, meta)

        loaded_vecs, loaded_qp, loaded_manifest = backend.load(store_path, mode)
        loaded_meta = backend.load_metadata(store_path)

        if mode is LoadMode.LAZY:
            assert loaded_vecs is None
        else:
            np.testing.assert_array_almost_equal(loaded_vecs, vecs)
        assert loaded_qp is None
        assert loaded_manifest["dim"] == 32
        assert loaded_manifest["keys"] == manifest["keys"]
        assert loaded_meta == meta

    def test_mmap_is_read_only(self, backend, store_path):
        vecs, qp, manifest, meta = _make_data()
        backend.save(store_path, vecs, qp, manifest, meta)
        loaded_vecs, _, _ = backend.load(store_path, LoadMode.MMAP)
        assert not loaded_vecs.flags.writeable


# ---------------------------------------------------------------------------
# Quant params round-trip
# ---------------------------------------------------------------------------


class TestQuantParams:
    def test_quant_params_saved_and_loaded(self, backend, store_path):
        rng = np.random.default_rng(1)
        vecs = rng.integers(-127, 128, (10, 32), dtype=np.int8)
        qp = rng.random((10, 1)).astype(np.float32)
        manifest = {
            "dim": 32,
            "dtype": StoreDtype.INT8_SYM.value,
            "keys": [f"k{i}" for i in range(10)],
        }
        backend.save(store_path, vecs, qp, manifest, {})

        loaded_vecs, loaded_qp, _ = backend.load(store_path, LoadMode.EAGER)
        np.testing.assert_array_equal(loaded_vecs, vecs)
        np.testing.assert_array_almost_equal(loaded_qp, qp)

    def test_quant_params_absent_when_none(self, backend, store_path):
        rng = np.random.default_rng(2)
        # First save with quant_params
        vecs = rng.integers(-127, 128, (5, 16), dtype=np.int8)
        qp = rng.random((5, 1)).astype(np.float32)
        manifest = {
            "dim": 16,
            "dtype": StoreDtype.INT8_SYM.value,
            "keys": [f"k{i}" for i in range(5)],
        }
        backend.save(store_path, vecs, qp, manifest, {})

        # Now overwrite without quant_params
        vecs_f = rng.standard_normal((5, 16)).astype(np.float32)
        manifest2 = {
            "dim": 16,
            "dtype": StoreDtype.FLOAT32.value,
            "keys": [f"k{i}" for i in range(5)],
        }
        backend.save(store_path, vecs_f, None, manifest2, {})

        loaded_vecs, loaded_qp, _ = backend.load(store_path, LoadMode.EAGER)
        assert loaded_qp is None


# ---------------------------------------------------------------------------
# Coverage: missing metadata
# ---------------------------------------------------------------------------


class TestLoadMetadataNoFile:
    def test_returns_empty_when_no_store(self, backend, store_path):
        """load_metadata should return {} when the store doesn't exist."""
        store_path.mkdir(parents=True, exist_ok=True)
        result = backend.load_metadata(store_path)
        assert result == {}
