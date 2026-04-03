"""Tests for VectorStore — CRUD, dimension validation, dtype round-trips."""

import numpy as np
import pytest

from tqvs import VectorStore, NpyBackend, create_vector_store
from tqvs._types import LoadMode, StoreDtype

RNG = np.random.default_rng(42)
DIM = 32


@pytest.fixture
def store(tmp_path) -> VectorStore:
    return create_vector_store(tmp_path / "store", DIM)


@pytest.fixture
def store_path(tmp_path):
    return tmp_path / "store"


# ---------------------------------------------------------------------------
# add / get
# ---------------------------------------------------------------------------


class TestAdd:
    def test_add_and_get(self, store):
        vec = RNG.standard_normal(DIM).astype(np.float32)
        store.add("a", vec, metadata={"x": 1})
        got, md = store.get("a")
        np.testing.assert_allclose(got, vec, atol=1e-6)
        assert md == {"x": 1}

    def test_add_duplicate_raises(self, store):
        store.add("a", RNG.standard_normal(DIM).astype(np.float32))
        with pytest.raises(KeyError, match="Duplicate"):
            store.add("a", RNG.standard_normal(DIM).astype(np.float32))

    def test_add_wrong_dim_raises(self, store):
        with pytest.raises(ValueError, match="Dimension mismatch"):
            store.add("a", RNG.standard_normal(DIM + 1).astype(np.float32))

    def test_add_many(self, store):
        vecs = RNG.standard_normal((5, DIM)).astype(np.float32)
        keys = [f"k{i}" for i in range(5)]
        store.add_many(keys, vecs)
        assert len(store) == 5
        for i, k in enumerate(keys):
            got, _ = store.get(k)
            np.testing.assert_allclose(got, vecs[i], atol=1e-6)


# ---------------------------------------------------------------------------
# get raw
# ---------------------------------------------------------------------------


class TestGetRaw:
    def test_raw_float(self, store):
        vec = RNG.standard_normal(DIM).astype(np.float32)
        store.add("a", vec)
        data, qp, md = store.get("a", raw=True)
        assert qp is None
        np.testing.assert_allclose(data, vec, atol=1e-6)

    def test_raw_quantized(self, store_path):
        s = create_vector_store(store_path, DIM, dtype=StoreDtype.INT8_SYM)
        vec = RNG.standard_normal(DIM).astype(np.float32)
        s.add("a", vec)
        data, qp, md = s.get("a", raw=True)
        assert data.dtype == np.int8
        assert qp is not None


# ---------------------------------------------------------------------------
# update
# ---------------------------------------------------------------------------


class TestUpdate:
    def test_update_vector(self, store):
        v1 = RNG.standard_normal(DIM).astype(np.float32)
        v2 = RNG.standard_normal(DIM).astype(np.float32)
        store.add("a", v1)
        store.update("a", vector=v2)
        got, _ = store.get("a")
        np.testing.assert_allclose(got, v2, atol=1e-6)

    def test_update_metadata(self, store):
        store.add("a", RNG.standard_normal(DIM).astype(np.float32), metadata={"x": 1})
        store.update("a", metadata={"x": 2})
        _, md = store.get("a")
        assert md == {"x": 2}

    def test_update_missing_raises(self, store):
        with pytest.raises(KeyError):
            store.update("missing", vector=RNG.standard_normal(DIM).astype(np.float32))


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


class TestDelete:
    def test_delete(self, store):
        store.add("a", RNG.standard_normal(DIM).astype(np.float32))
        store.add("b", RNG.standard_normal(DIM).astype(np.float32))
        store.delete("a")
        assert "a" not in store
        assert "b" in store
        assert len(store) == 1

    def test_delete_missing_raises(self, store):
        with pytest.raises(KeyError):
            store.delete("nope")

    def test_delete_last(self, store):
        store.add("only", RNG.standard_normal(DIM).astype(np.float32))
        store.delete("only")
        assert len(store) == 0


# ---------------------------------------------------------------------------
# keys / contains / len
# ---------------------------------------------------------------------------


class TestEnumeration:
    def test_contains(self, store):
        store.add("a", RNG.standard_normal(DIM).astype(np.float32))
        assert "a" in store
        assert "b" not in store

    def test_len(self, store):
        assert len(store) == 0
        store.add("a", RNG.standard_normal(DIM).astype(np.float32))
        assert len(store) == 1

    def test_keys_prefix(self, store):
        for k in ["doc/a", "doc/b", "img/c"]:
            store.add(k, RNG.standard_normal(DIM).astype(np.float32))
        assert sorted(store.keys("doc/")) == ["doc/a", "doc/b"]
        assert sorted(store.keys()) == ["doc/a", "doc/b", "img/c"]


# ---------------------------------------------------------------------------
# save / reload
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_save_reload(self, store_path):
        s = create_vector_store(store_path, DIM)
        vec = RNG.standard_normal(DIM).astype(np.float32)
        s.add("x", vec, metadata={"v": 1})
        s.save()

        s2 = create_vector_store(store_path, DIM)
        got, md = s2.get("x")
        np.testing.assert_allclose(got, vec, atol=1e-6)
        assert md == {"v": 1}

    def test_save_reload_quantized(self, store_path):
        s = create_vector_store(store_path, DIM, dtype=StoreDtype.INT8_SYM)
        vec = RNG.standard_normal(DIM).astype(np.float32)
        s.add("x", vec)
        s.save()

        s2 = create_vector_store(store_path, DIM, dtype=StoreDtype.INT8_SYM)
        got, _ = s2.get("x")
        # precision loss expected
        np.testing.assert_allclose(got, vec, atol=0.05)

    def test_lazy_load(self, store_path):
        s = create_vector_store(store_path, DIM)
        s.add("x", RNG.standard_normal(DIM).astype(np.float32))
        s.save()

        s2 = create_vector_store(store_path, DIM, load_mode=LoadMode.LAZY)
        # Vectors not loaded yet, but keys are available via manifest
        assert "x" in s2
        # Accessing triggers eager load
        got, _ = s2.get("x")
        assert got.shape == (DIM,)

    def test_mmap_load(self, store_path):
        s = create_vector_store(store_path, DIM)
        vec = RNG.standard_normal(DIM).astype(np.float32)
        s.add("x", vec)
        s.save()

        s2 = create_vector_store(store_path, DIM, load_mode=LoadMode.MMAP)
        got, _ = s2.get("x")
        np.testing.assert_allclose(got, vec, atol=1e-6)


# ---------------------------------------------------------------------------
# dtype round-trips through store
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype,atol",
    [
        (StoreDtype.FLOAT64, 1e-6),
        (StoreDtype.FLOAT32, 1e-6),
        (StoreDtype.FLOAT16, 1e-2),
        (StoreDtype.BFLOAT16, 2e-2),
        (StoreDtype.INT8_SYM, 0.05),
        (StoreDtype.INT8_ASYM, 0.05),
        (StoreDtype.INT4, 0.5),
        (StoreDtype.INT3, 1.5),
    ],
)
def test_dtype_add_get_round_trip(tmp_path, dtype, atol):
    s = create_vector_store(tmp_path / "s", DIM, dtype=dtype)
    vec = RNG.standard_normal(DIM).astype(np.float32)
    s.add("a", vec)
    got, _ = s.get("a")
    np.testing.assert_allclose(got, vec, atol=atol)


# ---------------------------------------------------------------------------
# Coverage: edge cases & missing paths
# ---------------------------------------------------------------------------


class TestPathProperty:
    def test_path(self, store):
        assert store.path.name == "store"


class TestAddManyEdgeCases:
    def test_add_many_1d_raises_or_works(self, store):
        """add_many with a single 1-D vector should auto-reshape."""
        vec = RNG.standard_normal(DIM).astype(np.float32)
        store.add_many(["single"], vec)
        got, _ = store.get("single")
        np.testing.assert_allclose(got, vec, atol=1e-6)

    def test_add_many_length_mismatch_raises(self, store):
        vecs = RNG.standard_normal((3, DIM)).astype(np.float32)
        with pytest.raises(ValueError, match="keys length"):
            store.add_many(["a", "b"], vecs)


class TestGetMissingKey:
    def test_get_missing_raises(self, store):
        with pytest.raises(KeyError):
            store.get("nonexistent")


class TestQuantizedUpdateDelete:
    def test_update_vector_quantized(self, store_path):
        s = create_vector_store(store_path, DIM, dtype=StoreDtype.INT8_SYM)
        v1 = RNG.standard_normal(DIM).astype(np.float32)
        v2 = RNG.standard_normal(DIM).astype(np.float32)
        s.add("a", v1)
        s.update("a", vector=v2)
        got, _ = s.get("a")
        np.testing.assert_allclose(got, v2, atol=0.05)

    def test_delete_with_swap_quantized(self, store_path):
        """Delete non-last entry in a quantized store (swap-with-last path)."""
        s = create_vector_store(store_path, DIM, dtype=StoreDtype.INT8_SYM)
        v_a = RNG.standard_normal(DIM).astype(np.float32)
        v_b = RNG.standard_normal(DIM).astype(np.float32)
        s.add("a", v_a)
        s.add("b", v_b)
        s.delete("a")
        assert "a" not in s
        assert "b" in s
        got, _ = s.get("b")
        np.testing.assert_allclose(got, v_b, atol=0.05)


class TestSaveEmpty:
    def test_save_empty_store(self, store_path):
        s = create_vector_store(store_path, DIM)
        s.save()  # should not crash


class TestReload:
    def test_reload_discards_changes(self, store_path):
        s = create_vector_store(store_path, DIM)
        vec = RNG.standard_normal(DIM).astype(np.float32)
        s.add("x", vec)
        s.save()

        s.add("y", RNG.standard_normal(DIM).astype(np.float32))
        assert len(s) == 2
        s.reload()
        assert len(s) == 1
        assert "x" in s
        assert "y" not in s


class TestValidateVector:
    def test_2d_vector_input(self, store):
        vec = RNG.standard_normal((1, DIM)).astype(np.float32)
        store.add("a", vec)
        got, _ = store.get("a")
        assert got.shape == (DIM,)

    def test_3d_vector_raises(self, store):
        vec = RNG.standard_normal((1, 1, DIM)).astype(np.float32)
        with pytest.raises(ValueError, match="1-D or 2-D"):
            store.add("a", vec)


class TestMmapMutation:
    def test_mmap_detach_on_update(self, store_path):
        """Updating an mmap'd store should detach vectors to RAM."""
        s = create_vector_store(store_path, DIM)
        s.add("a", RNG.standard_normal(DIM).astype(np.float32))
        s.save()

        s2 = create_vector_store(store_path, DIM, load_mode=LoadMode.MMAP)
        s2.update("a", vector=RNG.standard_normal(DIM).astype(np.float32))
        assert s2._vectors is not None
        assert s2._vectors.flags.writeable

    def test_mmap_detach_on_delete(self, store_path):
        s = create_vector_store(store_path, DIM)
        s.add("a", RNG.standard_normal(DIM).astype(np.float32))
        s.add("b", RNG.standard_normal(DIM).astype(np.float32))
        s.save()

        s2 = create_vector_store(store_path, DIM, load_mode=LoadMode.MMAP)
        s2.delete("a")
        assert len(s2) == 1

    def test_mmap_detach_on_add(self, store_path):
        s = create_vector_store(store_path, DIM)
        s.add("a", RNG.standard_normal(DIM).astype(np.float32))
        s.save()

        s2 = create_vector_store(store_path, DIM, load_mode=LoadMode.MMAP)
        s2.add("b", RNG.standard_normal(DIM).astype(np.float32))
        assert len(s2) == 2

    def test_mmap_detach_quantized(self, store_path):
        """Mmap detach works for quantized stores with quant_params."""
        s = create_vector_store(store_path, DIM, dtype=StoreDtype.INT8_SYM)
        s.add("a", RNG.standard_normal(DIM).astype(np.float32))
        s.save()

        s2 = create_vector_store(store_path, DIM, dtype=StoreDtype.INT8_SYM, load_mode=LoadMode.MMAP)
        s2.update("a", vector=RNG.standard_normal(DIM).astype(np.float32))
        assert s2._vectors is not None
        assert s2._vectors.flags.writeable
        assert s2._quant_params is not None
        assert s2._quant_params.flags.writeable
