"""Performance benchmarks — backends × dtypes × vector counts.

Run with:
    pytest tests/test_perf.py -m perf -v -s

Quick smoke-test:
    pytest tests/test_perf.py -m perf -k "npy and float32 and 100" -v -s

The report table is printed at session end (requires ``-s``).
"""

from __future__ import annotations

import importlib.util
import time

import numpy as np
import pytest

from tqvs import VectorStore, create_vector_store
from tqvs._types import LoadMode, StoreDtype
from tqvs.backends.npy import NpyBackend

from tests.conftest import record_perf

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DIM = 128
RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# Optional backend availability
# ---------------------------------------------------------------------------

_HAS_H5PY = importlib.util.find_spec("h5py") is not None
_HAS_LMDB = importlib.util.find_spec("lmdb") is not None
_HAS_LANCE = importlib.util.find_spec("lance") is not None
_HAS_PYARROW = importlib.util.find_spec("pyarrow") is not None

# ---------------------------------------------------------------------------
# Backend param list (conditionally skip unavailable ones)
# ---------------------------------------------------------------------------


def _make_backend_params():
    """Build the list of pytest params for backends."""
    params = [pytest.param(NpyBackend, id="npy")]

    if _HAS_H5PY:
        from tqvs.backends.hdf5 import Hdf5Backend

        params.append(pytest.param(Hdf5Backend, id="hdf5"))
    else:
        params.append(
            pytest.param(
                None,
                id="hdf5",
                marks=pytest.mark.skip(reason="h5py not installed"),
            )
        )

    if _HAS_LMDB:
        from tqvs.backends.lmdb import LmdbBackend

        params.append(pytest.param(LmdbBackend, id="lmdb"))
    else:
        params.append(
            pytest.param(
                None,
                id="lmdb",
                marks=pytest.mark.skip(reason="lmdb not installed"),
            )
        )

    if _HAS_LANCE:
        from tqvs.backends.lance import LanceBackend

        params.append(pytest.param(LanceBackend, id="lance"))
    else:
        params.append(
            pytest.param(
                None,
                id="lance",
                marks=pytest.mark.skip(reason="pylance not installed"),
            )
        )

    if _HAS_PYARROW:
        from tqvs.backends.parquet import ParquetBackend

        params.append(pytest.param(ParquetBackend, id="parquet"))
    else:
        params.append(
            pytest.param(
                None,
                id="parquet",
                marks=pytest.mark.skip(reason="pyarrow not installed"),
            )
        )

    return params


BACKEND_PARAMS = _make_backend_params()

# ---------------------------------------------------------------------------
# Dtype params
# ---------------------------------------------------------------------------

DTYPE_PARAMS = [pytest.param(dt, id=dt.value) for dt in StoreDtype]

# ---------------------------------------------------------------------------
# Vector-count params (powers of 10 up to 1 M)
# ---------------------------------------------------------------------------

SIZE_PARAMS = [
    pytest.param(n, id=f"n{n}")
    for n in [10, 100, 1_000, 10_000, 100_000, 1_000_000]
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Cache pre-generated vectors per size to avoid re-generating for every combo
_VEC_CACHE: dict[int, np.ndarray] = {}


def _get_vectors(n: int) -> np.ndarray:
    """Return a cached (n, DIM) float32 array."""
    if n not in _VEC_CACHE:
        _VEC_CACHE[n] = RNG.standard_normal((n, DIM)).astype(np.float32)
    return _VEC_CACHE[n]


def _make_keys(n: int) -> list[str]:
    """First half prefixed 'doc/', second half 'img/'."""
    half = n // 2
    return [f"doc/{i}" for i in range(half)] + [f"img/{i}" for i in range(half, n)]


def _backend_name(backend_cls: type) -> str:
    return backend_cls.__name__.replace("Backend", "").lower()


def _needs_rotation(dtype: StoreDtype) -> dict:
    """Return extra kwargs for create_vector_store when TurboQuant is used."""
    if dtype.is_turboquant:
        return {"rotation_seed": 42}
    return {}


def _build_store(
    tmp_path,
    backend_cls: type,
    dtype: StoreDtype,
    n: int,
) -> VectorStore:
    """Create a store and bulk-insert *n* vectors (not timed)."""
    store = create_vector_store(
        tmp_path / "store",
        DIM,
        backend=backend_cls(),
        dtype=dtype,
        **_needs_rotation(dtype),
    )
    vecs = _get_vectors(n)
    keys = _make_keys(n)
    store.add_many(keys, vecs)
    return store


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(params=BACKEND_PARAMS)
def backend_cls(request):
    return request.param


@pytest.fixture(params=DTYPE_PARAMS)
def dtype(request):
    return request.param


@pytest.fixture(params=SIZE_PARAMS)
def n_vectors(request):
    return request.param


@pytest.fixture()
def populated_store(backend_cls, dtype, n_vectors, tmp_path) -> VectorStore:
    """Build and return a pre-populated store (setup time is NOT recorded)."""
    return _build_store(tmp_path, backend_cls, dtype, n_vectors)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.perf


class TestAddManyPerf:
    """Time bulk insertion via ``add_many``."""

    def test_add_many(self, backend_cls, dtype, n_vectors, tmp_path):
        vecs = _get_vectors(n_vectors)
        keys = _make_keys(n_vectors)

        store = create_vector_store(
            tmp_path / "store",
            DIM,
            backend=backend_cls(),
            dtype=dtype,
            **_needs_rotation(dtype),
        )

        t0 = time.perf_counter()
        store.add_many(keys, vecs)
        elapsed = time.perf_counter() - t0

        record_perf(
            _backend_name(backend_cls),
            dtype.value,
            n_vectors,
            "add_many",
            elapsed,
        )


class TestPrefixQueryPerf:
    """Time ``query(prefix=...)`` on a pre-populated store."""

    def test_prefix_query(self, populated_store, backend_cls, dtype, n_vectors):
        query_vec = RNG.standard_normal(DIM).astype(np.float32)

        t0 = time.perf_counter()
        populated_store.query(query_vec, k=10, prefix="doc/")
        elapsed = time.perf_counter() - t0

        record_perf(
            _backend_name(backend_cls),
            dtype.value,
            n_vectors,
            "query_prefix",
            elapsed,
        )


class TestPrefixKeysPerf:
    """Time ``keys(prefix=...)`` iteration on a pre-populated store."""

    def test_prefix_keys(self, populated_store, backend_cls, dtype, n_vectors):
        t0 = time.perf_counter()
        _ = list(populated_store.keys(prefix="doc/"))
        elapsed = time.perf_counter() - t0

        record_perf(
            _backend_name(backend_cls),
            dtype.value,
            n_vectors,
            "keys_prefix",
            elapsed,
        )


class TestSavePerf:
    """Time ``save()`` — persisting an in-memory store to disk."""

    def test_save(self, populated_store, backend_cls, dtype, n_vectors):
        t0 = time.perf_counter()
        populated_store.save()
        elapsed = time.perf_counter() - t0

        record_perf(
            _backend_name(backend_cls),
            dtype.value,
            n_vectors,
            "save",
            elapsed,
        )


class TestReloadPerf:
    """Time opening an existing store from disk (eager load)."""

    def test_reload(self, backend_cls, dtype, n_vectors, tmp_path):
        # Setup: build, save, then close (not timed)
        store = _build_store(tmp_path, backend_cls, dtype, n_vectors)
        store.save()
        del store

        # Time: open a fresh store that loads from disk
        t0 = time.perf_counter()
        create_vector_store(
            tmp_path / "store",
            DIM,
            backend=backend_cls(),
            dtype=dtype,
            load_mode=LoadMode.EAGER,
            **_needs_rotation(dtype),
        )
        elapsed = time.perf_counter() - t0

        record_perf(
            _backend_name(backend_cls),
            dtype.value,
            n_vectors,
            "reload",
            elapsed,
        )
