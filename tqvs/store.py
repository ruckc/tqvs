"""VectorStore — the main user-facing class."""

from __future__ import annotations

import bisect
from pathlib import Path
from typing import Callable, Iterator, Literal, overload

import numpy as np
from numpy.typing import NDArray

from tqvs._locking import StoreLock
from tqvs._types import LoadMode, Metadata, QueryResult, StoreDtype
from tqvs.backends.base import Backend, Manifest
from tqvs.metrics import cosine_similarity
from tqvs.quantize import dequantize, make_rotation_matrix, quantize
from tqvs import query as _qe

# ---------------------------------------------------------------------------
# Sorted prefix index for fast prefix lookups
# ---------------------------------------------------------------------------

class _PrefixIndex:
    """Maintain a sorted list of (key, position) for O(log n) prefix queries."""

    __slots__ = ("_sorted_keys", "_sorted_indices")

    def __init__(self) -> None:
        self._sorted_keys: list[str] = []
        self._sorted_indices: list[int] = []

    def build(self, keys: list[str]) -> None:
        pairs = sorted(enumerate(keys), key=lambda p: p[1])
        self._sorted_indices = [p[0] for p in pairs]
        self._sorted_keys = [p[1] for p in pairs]

    def add(self, key: str, idx: int) -> None:
        pos = bisect.bisect_left(self._sorted_keys, key)
        self._sorted_keys.insert(pos, key)
        self._sorted_indices.insert(pos, idx)

    def remove(self, key: str, idx: int) -> None:
        pos = bisect.bisect_left(self._sorted_keys, key)
        while pos < len(self._sorted_keys) and self._sorted_keys[pos] == key:
            if self._sorted_indices[pos] == idx:
                del self._sorted_keys[pos]
                del self._sorted_indices[pos]
                return
            pos += 1

    def update_index(self, key: str, old_idx: int, new_idx: int) -> None:
        pos = bisect.bisect_left(self._sorted_keys, key)
        while pos < len(self._sorted_keys) and self._sorted_keys[pos] == key:
            if self._sorted_indices[pos] == old_idx:
                self._sorted_indices[pos] = new_idx
                return
            pos += 1

    def prefix_indices(self, prefix: str) -> list[int]:
        lo = bisect.bisect_left(self._sorted_keys, prefix)
        # Upper bound: prefix with last char incremented
        hi_key = prefix[:-1] + chr(ord(prefix[-1]) + 1) if prefix else ""
        hi = bisect.bisect_left(self._sorted_keys, hi_key) if hi_key else len(self._sorted_keys)
        return self._sorted_indices[lo:hi]

    def keys_with_prefix(self, prefix: str) -> list[str]:
        lo = bisect.bisect_left(self._sorted_keys, prefix)
        hi_key = prefix[:-1] + chr(ord(prefix[-1]) + 1) if prefix else ""
        hi = bisect.bisect_left(self._sorted_keys, hi_key) if hi_key else len(self._sorted_keys)
        return self._sorted_keys[lo:hi]


class VectorStore:
    """A keyed vector store with pluggable persistence and dtype.

    Do not instantiate directly — use :func:`tqvs.create_vector_store` or
    :class:`tqvs.VectorStoreBuilder`.
    """

    def __init__(
        self,
        path: str | Path,
        dim: int,
        backend: Backend,
        *,
        load_mode: LoadMode = LoadMode.EAGER,
        dtype: StoreDtype = StoreDtype.FLOAT32,
        metric: Callable = cosine_similarity,
        device: str | None = None,
        rotation_seed: int | None = None,
    ) -> None:
        self._path = Path(path)
        self._dim = dim
        self._backend = backend
        self._load_mode = load_mode
        self._dtype = dtype
        self._metric = metric
        self._device = device

        self._vectors: np.ndarray | None = None
        self._quant_params: np.ndarray | None = None
        self._vec_capacity: int = 0  # allocated rows in the buffer
        self._vec_len: int = 0  # logical number of vectors
        self._keys: list[str] = []
        self._key_index: dict[str, int] = {}
        self._prefix_index = _PrefixIndex()
        self._metadata: dict[str, Metadata] = {}
        self._dirty = False

        self._lock = StoreLock(self._path)

        # TurboQuant rotation state
        self._rotation_seed: int | None = rotation_seed
        self._rotation_matrix: np.ndarray | None = None
        if self._dtype.is_turboquant:
            if self._rotation_seed is None:
                self._rotation_seed = int(np.random.default_rng().integers(0, 2**31))
            self._rotation_matrix = make_rotation_matrix(dim, self._rotation_seed)

        # Load from disk if the store already exists.
        if self._backend.exists(self._path):
            self._load_from_disk()

    # -- properties -----------------------------------------------------------

    @property
    def path(self) -> Path:
        return self._path

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def dtype(self) -> StoreDtype:
        return self._dtype

    # -- CRUD -----------------------------------------------------------------

    def add(
        self,
        key: str,
        vector: NDArray[np.floating],
        metadata: Metadata | None = None,
    ) -> None:
        """Add a single vector.  *vector* is auto-quantized to the store's dtype."""
        with self._lock.write_lock():
            self._check_no_duplicate(key)
            vec = self._validate_vector(vector)
            data, qp = quantize(vec, self._dtype, rotation_matrix=self._rotation_matrix)
            self._append(key, data[0], qp[0] if qp is not None else None, metadata)

    def add_many(
        self,
        keys: list[str],
        vectors: NDArray[np.floating],
        metadata: list[Metadata | None] | None = None,
    ) -> None:
        """Batch add.  *vectors* shape ``(n, dim)``."""
        with self._lock.write_lock():
            vecs = np.asarray(vectors, dtype=np.float32)
            if vecs.ndim == 1:
                vecs = vecs[np.newaxis, :]
            if vecs.shape[0] != len(keys):
                raise ValueError(
                    f"keys length ({len(keys)}) != vectors rows ({vecs.shape[0]})"
                )
            for key in keys:
                self._check_no_duplicate(key)
            self._validate_dim(vecs.shape[1])

            data, qp = quantize(vecs, self._dtype, rotation_matrix=self._rotation_matrix)

            n_new = data.shape[0]
            self._detach_mmap()
            start_idx = self._vec_len

            # Grow buffer to fit new data
            self._ensure_capacity(self._vec_len + n_new, data.dtype,
                                  data.shape[1],
                                  qp.shape[1] if qp is not None else 0,
                                  qp.dtype if qp is not None else None)
            assert self._vectors is not None  # guaranteed by _ensure_capacity

            self._vectors[self._vec_len:self._vec_len + n_new] = data
            if qp is not None and self._quant_params is not None:
                self._quant_params[self._vec_len:self._vec_len + n_new] = qp
            self._vec_len += n_new

            self._keys.extend(keys)
            for i, key in enumerate(keys):
                self._key_index[key] = start_idx + i
                self._prefix_index.add(key, start_idx + i)

            if metadata:
                for i, key in enumerate(keys):
                    md = metadata[i]
                    if md is not None:
                        self._metadata[key] = md

            self._dirty = True

    @overload
    def get(
        self,
        key: str,
        *,
        raw: Literal[False] = ...,
    ) -> tuple[np.ndarray, Metadata | None]: ...

    @overload
    def get(
        self,
        key: str,
        *,
        raw: Literal[True],
    ) -> tuple[np.ndarray, np.ndarray | None, Metadata | None]: ...

    def get(
        self,
        key: str,
        *,
        raw: bool = False,
    ) -> tuple[np.ndarray, np.ndarray | None, Metadata | None] | tuple[np.ndarray, Metadata | None]:
        """Retrieve a vector by key.

        Parameters
        ----------
        raw : bool
            If ``False`` (default) returns ``(float32_vector, metadata)``.
            If ``True`` returns ``(stored_data, quant_params_row_or_None, metadata)``.
        """
        with self._lock.read_lock():
            self._ensure_loaded()
            idx = self._key_index.get(key)
            if idx is None:
                raise KeyError(key)

            row = self._vectors[idx]  # type: ignore[index]
            qp_row: np.ndarray | None = None
            if self._quant_params is not None:
                qp_row = self._quant_params[idx]

            md = self._metadata.get(key)

            if raw:
                return row.copy(), qp_row.copy() if qp_row is not None else None, md

            # Dequantize
            if self._dtype.is_quantized or self._dtype is StoreDtype.BFLOAT16:
                float_vec = dequantize(
                    row[np.newaxis, :],
                    self._dtype,
                    qp_row[np.newaxis, :] if qp_row is not None else None,
                    self._dim,
                    rotation_matrix=self._rotation_matrix,
                )[0]
            else:
                float_vec = row.astype(np.float32, copy=True)
            return float_vec, md

    def update(
        self,
        key: str,
        vector: NDArray[np.floating] | None = None,
        metadata: Metadata | None = None,
    ) -> None:
        """Update a vector and/or metadata in-place."""
        with self._lock.write_lock():
            self._ensure_loaded()
            idx = self._key_index.get(key)
            if idx is None:
                raise KeyError(key)

            if vector is not None:
                vec = self._validate_vector(vector)
                data, qp = quantize(vec, self._dtype, rotation_matrix=self._rotation_matrix)
                self._detach_mmap()
                self._vectors[idx] = data[0]  # type: ignore[index]
                if qp is not None and self._quant_params is not None:
                    self._quant_params[idx] = qp[0]
                self._dirty = True

            if metadata is not None:
                self._metadata[key] = metadata
                self._dirty = True

    def delete(self, key: str) -> None:
        """Remove a vector by key (swap-with-last, O(1))."""
        with self._lock.write_lock():
            self._ensure_loaded()
            idx = self._key_index.pop(key, None)
            if idx is None:
                raise KeyError(key)

            last = self._vec_len - 1
            self._detach_mmap()

            # Remove from prefix index
            self._prefix_index.remove(key, idx)

            if idx != last:
                # Swap with last element
                last_key = self._keys[last]
                self._vectors[idx] = self._vectors[last]  # type: ignore[index]
                if self._quant_params is not None:
                    self._quant_params[idx] = self._quant_params[last]
                self._keys[idx] = last_key
                self._key_index[last_key] = idx
                self._prefix_index.update_index(last_key, last, idx)

            # Truncate logical length (keep buffer capacity)
            self._keys.pop()
            self._vec_len -= 1
            self._metadata.pop(key, None)
            self._dirty = True

    # -- enumeration ----------------------------------------------------------

    def __contains__(self, key: str) -> bool:
        with self._lock.read_lock():
            return key in self._key_index

    def __len__(self) -> int:
        with self._lock.read_lock():
            return self._vec_len

    def keys(self, prefix: str | None = None) -> Iterator[str]:
        """Iterate over keys, optionally filtered by *prefix*."""
        with self._lock.read_lock():
            if prefix is not None:
                yield from self._prefix_index.keys_with_prefix(prefix)
            else:
                yield from self._keys

    # -- query ----------------------------------------------------------------

    def query(
        self,
        vector: NDArray[np.floating],
        k: int = 10,
        *,
        prefix: str | None = None,
        metric: Callable | None = None,
    ) -> list[QueryResult]:
        """Return the *k* most-similar entries."""
        with self._lock.read_lock():
            self._ensure_loaded()
            prefix_idx = self._prefix_index.prefix_indices(prefix) if prefix else None
            return _qe.top_k(
                np.asarray(vector, dtype=np.float32),
                self._active_vectors,
                self._keys,
                self._metadata,
                k,
                metric or self._metric,
                self._dtype,
                self._active_quant_params,
                self._dim,
                prefix=prefix,
                prefix_indices=prefix_idx,
                device=self._device,
                rotation_matrix=self._rotation_matrix,
            )

    def score(
        self,
        vector: NDArray[np.floating],
        *,
        prefix: str | None = None,
        metric: Callable | None = None,
    ) -> list[QueryResult]:
        """Score a query against all (or prefix-filtered) vectors."""
        with self._lock.read_lock():
            self._ensure_loaded()
            prefix_idx = self._prefix_index.prefix_indices(prefix) if prefix else None
            return _qe.score_all(
                np.asarray(vector, dtype=np.float32),
                self._active_vectors,
                self._keys,
                self._metadata,
                metric or self._metric,
                self._dtype,
                self._active_quant_params,
                self._dim,
                prefix=prefix,
                prefix_indices=prefix_idx,
                device=self._device,
                rotation_matrix=self._rotation_matrix,
            )

    def score_array(
        self,
        vector: NDArray[np.floating],
        *,
        prefix: str | None = None,
        metric: Callable | None = None,
    ) -> np.ndarray:
        """Score a query against all vectors, returning a raw float32 array.

        Unlike :meth:`score`, this returns a plain ``np.ndarray`` in insertion
        order without wrapping each entry in a :class:`QueryResult`.
        """
        with self._lock.read_lock():
            self._ensure_loaded()
            prefix_idx = self._prefix_index.prefix_indices(prefix) if prefix else None
            return _qe.score_array_raw(
                np.asarray(vector, dtype=np.float32),
                self._active_vectors,
                self._keys,
                self._metadata,
                metric or self._metric,
                self._dtype,
                self._active_quant_params,
                self._dim,
                prefix=prefix,
                prefix_indices=prefix_idx,
                device=self._device,
                rotation_matrix=self._rotation_matrix,
            )

    def score_many(
        self,
        vectors: NDArray[np.floating],
        *,
        metric: Callable | None = None,
    ) -> np.ndarray:
        """Score multiple queries at once, returning an (N, M) score matrix.

        Transfers candidate vectors to GPU only once when a torch device is
        configured.  Prefix filtering is not supported; use the full store.
        """
        with self._lock.read_lock():
            self._ensure_loaded()
            return _qe.score_batch(
                np.asarray(vectors, dtype=np.float32),
                self._active_vectors,
                self._keys,
                metric or self._metric,
                self._dtype,
                self._active_quant_params,
                self._dim,
                device=self._device,
                rotation_matrix=self._rotation_matrix,
            )

    @property
    def vectors(self) -> np.ndarray:
        """Return the dequantized float32 active vector buffer (shape ``(n, dim)``)."""
        from tqvs.quantize import dequantize as _dequantize
        with self._lock.read_lock():
            self._ensure_loaded()
            active = self._active_vectors
            if active is None:
                return np.empty((0, self._dim), dtype=np.float32)
            if self._dtype.is_quantized or self._dtype is StoreDtype.BFLOAT16:
                return _dequantize(
                    active, self._dtype, self._active_quant_params, self._dim,
                    rotation_matrix=self._rotation_matrix,
                )
            return active.astype(np.float32, copy=False)

    # -- persistence ----------------------------------------------------------

    def save(self) -> None:
        """Persist current state to disk."""
        with self._lock.write_lock():
            if self._vectors is None or self._vec_len == 0:
                return
            manifest: Manifest = {
                "dim": self._dim,
                "dtype": self._dtype.value,
                "keys": self._keys,
            }
            if self._rotation_seed is not None:
                manifest["rotation_seed"] = self._rotation_seed
            active_vecs = self._active_vectors
            assert active_vecs is not None  # checked self._vectors above
            active_qp = self._active_quant_params
            self._backend.save(
                self._path,
                active_vecs,
                active_qp,
                manifest,
                self._metadata,
            )
            self._dirty = False

    def reload(self) -> None:
        """Re-read store from disk, discarding unsaved in-memory changes."""
        with self._lock.write_lock():
            self._load_from_disk()
            self._dirty = False

    # -- internals ------------------------------------------------------------

    def _load_from_disk(self) -> None:
        vectors, qp, manifest = self._backend.load(self._path, self._load_mode)
        metadata = self._backend.load_metadata(self._path)
        self._dim = manifest["dim"]
        self._dtype = StoreDtype(manifest["dtype"])
        self._keys = list(manifest["keys"])
        self._key_index = {k: i for i, k in enumerate(self._keys)}
        self._metadata = metadata
        self._vectors = vectors
        self._quant_params = qp

        # Sync buffer tracking
        if vectors is not None:
            self._vec_len = vectors.shape[0]
            self._vec_capacity = vectors.shape[0]
        else:
            self._vec_len = len(self._keys)
            self._vec_capacity = 0

        # Build prefix index
        self._prefix_index = _PrefixIndex()
        self._prefix_index.build(self._keys)

        # Restore TurboQuant rotation state
        if "rotation_seed" in manifest:
            seed: int = manifest["rotation_seed"]
            self._rotation_seed = seed
            self._rotation_matrix = make_rotation_matrix(
                self._dim, seed
            )

    def _ensure_loaded(self) -> None:
        """Eagerly load vectors when the store was opened with LAZY mode."""
        if self._vectors is None and self._backend.exists(self._path):
            vectors, qp, _ = self._backend.load(self._path, LoadMode.EAGER)
            self._vectors = vectors
            self._quant_params = qp
            if vectors is not None:
                self._vec_len = vectors.shape[0]
                self._vec_capacity = vectors.shape[0]

    def _validate_vector(self, vector: NDArray[np.floating]) -> NDArray[np.float32]:
        vec = np.asarray(vector, dtype=np.float32)
        if vec.ndim == 1:
            self._validate_dim(vec.shape[0])
            vec = vec[np.newaxis, :]
        elif vec.ndim == 2:
            self._validate_dim(vec.shape[1])
        else:
            raise ValueError(f"Expected 1-D or 2-D vector, got shape {vec.shape}")
        return vec

    def _validate_dim(self, d: int) -> None:
        if d != self._dim:
            raise ValueError(
                f"Dimension mismatch: store expects {self._dim}, got {d}"
            )

    def _check_no_duplicate(self, key: str) -> None:
        if key in self._key_index:
            raise KeyError(f"Duplicate key: {key!r}")

    def _append(
        self,
        key: str,
        data_row: np.ndarray,
        qp_row: np.ndarray | None,
        md: Metadata | None,
    ) -> None:
        self._detach_mmap()
        idx = self._vec_len

        # Ensure buffer has space
        self._ensure_capacity(
            idx + 1, data_row.dtype, data_row.shape[0],
            qp_row.shape[0] if qp_row is not None else 0,
            qp_row.dtype if qp_row is not None else None,
        )
        assert self._vectors is not None  # guaranteed by _ensure_capacity

        self._keys.append(key)
        self._key_index[key] = idx
        self._prefix_index.add(key, idx)

        self._vectors[idx] = data_row
        if qp_row is not None and self._quant_params is not None:
            self._quant_params[idx] = qp_row

        self._vec_len += 1
        if md is not None:
            self._metadata[key] = md
        self._dirty = True

    def _ensure_capacity(
        self,
        needed: int,
        vec_dtype: np.dtype,
        vec_cols: int,
        qp_cols: int,
        qp_dtype: np.dtype | None,
    ) -> None:
        """Grow the pre-allocated vector buffer geometrically if needed."""
        if self._vectors is not None and self._vec_capacity >= needed:
            return

        new_cap = max(needed, self._vec_capacity * 2, 64)

        if self._vectors is None:
            self._vectors = np.empty((new_cap, vec_cols), dtype=vec_dtype)
            if qp_cols > 0 and qp_dtype is not None:
                self._quant_params = np.empty((new_cap, qp_cols), dtype=qp_dtype)
        else:
            new_buf = np.empty((new_cap, self._vectors.shape[1]), dtype=self._vectors.dtype)
            new_buf[:self._vec_len] = self._vectors[:self._vec_len]
            self._vectors = new_buf
            if self._quant_params is not None:
                new_qp = np.empty((new_cap, self._quant_params.shape[1]), dtype=self._quant_params.dtype)
                new_qp[:self._vec_len] = self._quant_params[:self._vec_len]
                self._quant_params = new_qp

        self._vec_capacity = new_cap

    @property
    def _active_vectors(self) -> np.ndarray | None:
        """Return a view over only the filled portion of the vector buffer."""
        if self._vectors is None:
            return None
        return self._vectors[:self._vec_len]

    @property
    def _active_quant_params(self) -> np.ndarray | None:
        """Return a view over only the filled portion of the quant params buffer."""
        if self._quant_params is None:
            return None
        return self._quant_params[:self._vec_len]

    def _detach_mmap(self) -> None:
        """If vectors are memory-mapped, copy to a writable in-memory array."""
        if self._vectors is not None and not self._vectors.flags.writeable:
            self._vectors = self._vectors.copy()
        if self._quant_params is not None and not self._quant_params.flags.writeable:
            self._quant_params = self._quant_params.copy()
