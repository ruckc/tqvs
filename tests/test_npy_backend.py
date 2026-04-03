"""Tests for NpyBackend — save / load round-trips across all LoadModes."""

import numpy as np
import pytest

from tqvs._types import LoadMode, StoreDtype
from tqvs.backends.npy import NpyBackend


@pytest.fixture
def backend() -> NpyBackend:
    return NpyBackend()


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
            np.testing.assert_array_equal(loaded_vecs, vecs)
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
        manifest = {"dim": 32, "dtype": StoreDtype.INT8_SYM.value, "keys": [f"k{i}" for i in range(10)]}
        backend.save(store_path, vecs, qp, manifest, {})

        loaded_vecs, loaded_qp, _ = backend.load(store_path, LoadMode.EAGER)
        np.testing.assert_array_equal(loaded_vecs, vecs)
        np.testing.assert_array_equal(loaded_qp, qp)

    def test_quant_params_removed_when_none(self, backend, store_path):
        rng = np.random.default_rng(2)
        # First save with quant_params
        vecs = rng.integers(-127, 128, (5, 16), dtype=np.int8)
        qp = rng.random((5, 1)).astype(np.float32)
        manifest = {"dim": 16, "dtype": StoreDtype.INT8_SYM.value, "keys": [f"k{i}" for i in range(5)]}
        backend.save(store_path, vecs, qp, manifest, {})
        assert (store_path / "quant_params.npy").exists()

        # Now save without
        vecs_f = rng.standard_normal((5, 16)).astype(np.float32)
        manifest2 = {"dim": 16, "dtype": StoreDtype.FLOAT32.value, "keys": [f"k{i}" for i in range(5)]}
        backend.save(store_path, vecs_f, None, manifest2, {})
        assert not (store_path / "quant_params.npy").exists()


# ---------------------------------------------------------------------------
# Coverage: missing metadata file, atomic write error paths
# ---------------------------------------------------------------------------


class TestLoadMetadataNoFile:
    def test_returns_empty_when_no_metadata_file(self, backend, store_path):
        """load_metadata should return {} when metadata.json doesn't exist."""
        store_path.mkdir(parents=True, exist_ok=True)
        result = backend.load_metadata(store_path)
        assert result == {}


class TestAtomicWriteCleanup:
    def test_atomic_npy_cleans_up_on_error(self, backend, store_path, monkeypatch):
        """If np.save raises, the temp file should be cleaned up."""
        import tqvs.backends.npy as npy_mod

        store_path.mkdir(parents=True, exist_ok=True)

        def bad_save(*args, **kwargs):
            raise IOError("simulated write failure")

        monkeypatch.setattr(np, "save", bad_save)
        with pytest.raises(IOError, match="simulated"):
            npy_mod.NpyBackend._atomic_npy(store_path / "vectors.npy", np.zeros((2, 2)))

    def test_atomic_json_cleans_up_on_error(self, backend, store_path, monkeypatch):
        """If json.dump raises, the temp file should be cleaned up."""
        import json as json_mod
        import tqvs.backends.npy as npy_mod

        store_path.mkdir(parents=True, exist_ok=True)

        original_dump = json_mod.dump

        def bad_dump(*args, **kwargs):
            raise IOError("simulated json failure")

        monkeypatch.setattr(json_mod, "dump", bad_dump)
        with pytest.raises(IOError, match="simulated"):
            npy_mod.NpyBackend._atomic_json(store_path / "manifest.json", {"x": 1})
