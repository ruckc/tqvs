"""tqvs — composable vector store with pluggable persistence and dtype."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from tqvs._types import (
    LoadMode,
    Metadata,
    MetricFn,
    QueryResult,
    StoreDtype,
)
from tqvs.backends.base import Backend
from tqvs.backends.npy import NpyBackend
from tqvs.builder import VectorStoreBuilder
from tqvs.metrics import cosine_similarity, dot_product, euclidean_distance
from tqvs.store import VectorStore

__all__ = [
    # Core
    "VectorStore",
    "VectorStoreBuilder",
    "create_vector_store",
    # Types
    "LoadMode",
    "Metadata",
    "MetricFn",
    "QueryResult",
    "StoreDtype",
    # Backends
    "Backend",
    "NpyBackend",
    # Metrics
    "cosine_similarity",
    "dot_product",
    "euclidean_distance",
]

try:
    from tqvs.backends.lmdb import LmdbBackend

    __all__ += ["LmdbBackend"]
except ImportError:
    pass

try:
    from tqvs.backends.hdf5 import Hdf5Backend

    __all__ += ["Hdf5Backend"]
except ImportError:
    pass

try:
    from tqvs.backends.lance import LanceBackend

    __all__ += ["LanceBackend"]
except ImportError:
    pass

try:
    from tqvs.backends.parquet import ParquetBackend

    __all__ += ["ParquetBackend"]
except ImportError:
    pass


def create_vector_store(
    path: str | Path,
    dim: int,
    *,
    backend: Backend | None = None,
    load_mode: LoadMode = LoadMode.EAGER,
    dtype: StoreDtype = StoreDtype.FLOAT32,
    metric: Callable = cosine_similarity,
    device: str | None = None,
    rotation_seed: int | None = None,
) -> VectorStore:
    """Factory function to create (or open) a :class:`VectorStore`.

    Parameters
    ----------
    path : str | Path
        Directory where the store data lives.
    dim : int
        Fixed vector dimensionality.
    backend : Backend | None
        Persistence backend.  Defaults to :class:`NpyBackend`.
    load_mode : LoadMode
        ``EAGER`` (default), ``LAZY``, or ``MMAP``.
    dtype : StoreDtype
        Storage dtype (default ``FLOAT32``).
    metric : callable
        Default similarity metric (default :func:`cosine_similarity`).
    device : str | None
        Torch device for accelerated scoring (e.g. ``"cuda"``).
    rotation_seed : int | None
        Seed for the random rotation matrix used by TurboQuant dtypes.
        If ``None`` and a TurboQuant dtype is selected, a random seed is
        generated automatically.
    """
    return VectorStore(
        path=path,
        dim=dim,
        backend=backend or NpyBackend(),
        load_mode=load_mode,
        dtype=dtype,
        metric=metric,
        device=device,
        rotation_seed=rotation_seed,
    )
