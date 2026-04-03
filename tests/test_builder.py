"""Tests for VectorStoreBuilder and create_vector_store factory."""

import numpy as np
import pytest

from tqvs import (
    VectorStoreBuilder,
    create_vector_store,
    dot_product,
    NpyBackend,
)
from tqvs._types import LoadMode, StoreDtype

DIM = 16
RNG = np.random.default_rng(7)


class TestBuilder:
    def test_basic_build(self, tmp_path):
        store = (
            VectorStoreBuilder()
            .at(tmp_path / "s")
            .with_dim(DIM)
            .build()
        )
        assert store.dim == DIM
        assert store.dtype == StoreDtype.FLOAT32

    def test_all_options(self, tmp_path):
        store = (
            VectorStoreBuilder()
            .at(tmp_path / "s")
            .with_dim(DIM)
            .with_backend(NpyBackend())
            .with_load_mode(LoadMode.EAGER)
            .with_dtype(StoreDtype.INT4)
            .with_metric(dot_product)
            .build()
        )
        assert store.dtype == StoreDtype.INT4

    def test_missing_path_raises(self):
        with pytest.raises(ValueError, match="path"):
            VectorStoreBuilder().with_dim(DIM).build()

    def test_missing_dim_raises(self, tmp_path):
        with pytest.raises(ValueError, match="dimension"):
            VectorStoreBuilder().at(tmp_path / "s").build()

    def test_zero_dim_raises(self, tmp_path):
        with pytest.raises(ValueError, match="positive"):
            VectorStoreBuilder().at(tmp_path / "s").with_dim(0).build()


class TestFactory:
    def test_factory_default(self, tmp_path):
        s = create_vector_store(tmp_path / "s", DIM)
        assert s.dim == DIM
        assert s.dtype == StoreDtype.FLOAT32

    def test_factory_with_dtype(self, tmp_path):
        s = create_vector_store(tmp_path / "s", DIM, dtype=StoreDtype.INT8_SYM)
        assert s.dtype == StoreDtype.INT8_SYM

    def test_factory_equals_builder(self, tmp_path):
        """Factory and builder produce stores with same behaviour."""
        vec = RNG.standard_normal(DIM).astype(np.float32)

        s1 = create_vector_store(tmp_path / "f", DIM, dtype=StoreDtype.INT8_SYM)
        s1.add("a", vec)

        s2 = (
            VectorStoreBuilder()
            .at(tmp_path / "b")
            .with_dim(DIM)
            .with_dtype(StoreDtype.INT8_SYM)
            .build()
        )
        s2.add("a", vec)

        g1, _ = s1.get("a")
        g2, _ = s2.get("a")
        np.testing.assert_array_equal(g1, g2)


class TestBuilderDevice:
    def test_with_device(self, tmp_path):
        store = (
            VectorStoreBuilder()
            .at(tmp_path / "s")
            .with_dim(DIM)
            .with_device("cpu")
            .build()
        )
        assert store._device == "cpu"


class TestStoreDtypeNumpyDtype:
    def test_numpy_dtype_for_all(self):
        for dt in StoreDtype:
            assert dt.numpy_dtype is not None
