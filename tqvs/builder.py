"""Fluent builder for :class:`~tqvs.store.VectorStore`."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Self

from tqvs._types import LoadMode, StoreDtype
from tqvs.backends.base import Backend
from tqvs.backends.npy import NpyBackend
from tqvs.metrics import cosine_similarity
from tqvs.store import VectorStore


class VectorStoreBuilder:
    """Compose a :class:`VectorStore` via a fluent chain.

    Example::

        store = (
            VectorStoreBuilder()
            .at("./my_store")
            .dim(768)
            .dtype(StoreDtype.INT8_SYM)
            .load_mode(LoadMode.MMAP)
            .build()
        )
    """

    def __init__(self) -> None:
        self._path: str | Path | None = None
        self._dim: int | None = None
        self._backend: Backend | None = None
        self._load_mode: LoadMode = LoadMode.EAGER
        self._dtype: StoreDtype = StoreDtype.FLOAT32
        self._metric: Callable = cosine_similarity
        self._device: str | None = None

    # -- setters (return self for chaining) -----------------------------------

    def at(self, path: str | Path) -> Self:
        """Set the store directory path."""
        self._path = path
        return self

    def with_dim(self, dim: int) -> Self:
        """Set vector dimensionality."""
        self._dim = dim
        return self

    def with_backend(self, backend: Backend) -> Self:
        """Set the persistence backend."""
        self._backend = backend
        return self

    def with_load_mode(self, mode: LoadMode) -> Self:
        """Set load mode (EAGER, LAZY, MMAP)."""
        self._load_mode = mode
        return self

    def with_dtype(self, dtype: StoreDtype) -> Self:
        """Set the storage dtype / quantisation level."""
        self._dtype = dtype
        return self

    def with_metric(self, metric: Callable) -> Self:
        """Set the default similarity metric."""
        self._metric = metric
        return self

    def with_device(self, device: str) -> Self:
        """Set the torch device for accelerated scoring (e.g. ``"cuda"``)."""
        self._device = device
        return self

    # -- build ----------------------------------------------------------------

    def build(self) -> VectorStore:
        """Validate parameters and construct the :class:`VectorStore`."""
        if self._path is None:
            raise ValueError("Store path is required — call .at(path)")
        if self._dim is None:
            raise ValueError("Vector dimension is required — call .with_dim(dim)")
        if self._dim <= 0:
            raise ValueError(f"Dimension must be positive, got {self._dim}")

        backend = self._backend or NpyBackend()

        return VectorStore(
            path=self._path,
            dim=self._dim,
            backend=backend,
            load_mode=self._load_mode,
            dtype=self._dtype,
            metric=self._metric,
            device=self._device,
        )
