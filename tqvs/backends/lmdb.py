"""LMDB persistence backend."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    import lmdb
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The lmdb package is required for LmdbBackend. "
        "Install it with:  pip install tqvs[lmdb]"
    ) from exc

from tqvs._types import LoadMode, Metadata
from tqvs.backends.base import Manifest

_VECTORS_KEY = b"vectors"
_QUANT_PARAMS_KEY = b"quant_params"
_MANIFEST_KEY = b"manifest"
_METADATA_KEY = b"metadata"


class LmdbBackend:
    """Persistence backend that stores vectors in an LMDB database."""

    def __init__(self) -> None:
        # Holds LMDB env references for MMAP-mode arrays to prevent GC.
        self._open_envs: dict[str, lmdb.Environment] = {}

    # -- Backend protocol -----------------------------------------------------

    def exists(self, path: Path) -> bool:
        return (path / "data.mdb").exists()

    def _env(self, path: Path, *, readonly: bool = True) -> lmdb.Environment:
        """Return an open env, reusing one kept alive for MMAP mode."""
        key = str(path)
        if key in self._open_envs:
            return self._open_envs[key]
        return lmdb.open(key, readonly=readonly, lock=False)

    def load(
        self,
        path: Path,
        load_mode: LoadMode,
    ) -> tuple[np.ndarray | None, np.ndarray | None, Manifest]:
        env = self._env(path)
        try:
            with env.begin(buffers=True) as txn:
                manifest = self._get_json(txn, _MANIFEST_KEY)

                vectors: np.ndarray | None = None
                quant_params: np.ndarray | None = None

                if load_mode is not LoadMode.LAZY:
                    raw_vectors = txn.get(_VECTORS_KEY)
                    if raw_vectors is None:
                        raise FileNotFoundError(
                            f"Missing key {_VECTORS_KEY!r} in LMDB database"
                        )
                    vectors = self._buf_to_array(
                        raw_vectors,
                        manifest["vectors_dtype"],
                        manifest["vectors_shape"],
                    )
                    raw_qp = txn.get(_QUANT_PARAMS_KEY)
                    if raw_qp is not None:
                        quant_params = self._buf_to_array(
                            raw_qp,
                            manifest["qp_dtype"],
                            manifest["qp_shape"],
                        )

                    if load_mode is LoadMode.EAGER:
                        # Detach from LMDB mmap so env can be closed
                        vectors = vectors.copy()
                        if quant_params is not None:
                            quant_params = quant_params.copy()
                    else:
                        # MMAP mode – mark read-only (already backed by mmap)
                        vectors.flags.writeable = False
                        if quant_params is not None:
                            quant_params.flags.writeable = False
        finally:
            if load_mode is LoadMode.MMAP:
                # Keep env alive so the mmap-backed arrays stay valid.
                self._open_envs[str(path)] = env
            elif str(path) not in self._open_envs:
                env.close()

        return vectors, quant_params, manifest

    def load_metadata(self, path: Path) -> dict[str, Metadata]:
        env = self._env(path)
        try:
            with env.begin(buffers=True) as txn:
                raw = txn.get(_METADATA_KEY)
                if raw is None:
                    return {}
                return json.loads(bytes(raw))
        finally:
            if str(path) not in self._open_envs:
                env.close()

    def save(
        self,
        path: Path,
        vectors: np.ndarray,
        quant_params: np.ndarray | None,
        manifest: Manifest,
        metadata: dict[str, Metadata],
    ) -> None:
        path.mkdir(parents=True, exist_ok=True)

        # Close any read-only env kept alive for MMAP mode.
        key = str(path)
        old_env = self._open_envs.pop(key, None)
        if old_env is not None:
            old_env.close()

        # Augment manifest with array reconstruction info
        manifest = dict(manifest)  # don't mutate caller's dict
        manifest["vectors_dtype"] = str(vectors.dtype)
        manifest["vectors_shape"] = list(vectors.shape)
        if quant_params is not None:
            manifest["qp_dtype"] = str(quant_params.dtype)
            manifest["qp_shape"] = list(quant_params.shape)

        # Compute map_size: 2× total data size + 10 MB headroom
        data_size = vectors.nbytes
        if quant_params is not None:
            data_size += quant_params.nbytes
        manifest_bytes = json.dumps(manifest, ensure_ascii=False).encode("utf-8")
        metadata_bytes = json.dumps(metadata, ensure_ascii=False).encode("utf-8")
        data_size += len(manifest_bytes) + len(metadata_bytes)
        map_size = data_size * 2 + 10 * 1024 * 1024

        env = lmdb.open(str(path), map_size=map_size)
        try:
            with env.begin(write=True) as txn:
                txn.put(_VECTORS_KEY, vectors.tobytes())
                if quant_params is not None:
                    txn.put(_QUANT_PARAMS_KEY, quant_params.tobytes())
                else:
                    txn.delete(_QUANT_PARAMS_KEY)
                txn.put(_MANIFEST_KEY, manifest_bytes)
                txn.put(_METADATA_KEY, metadata_bytes)
        finally:
            env.close()

    # -- helpers --------------------------------------------------------------

    @staticmethod
    def _buf_to_array(
        buf: bytes | memoryview,
        dtype_str: str,
        shape: list[int],
    ) -> np.ndarray:
        return np.frombuffer(buf, dtype=np.dtype(dtype_str)).reshape(shape)

    @staticmethod
    def _get_json(txn: lmdb.Transaction, key: bytes) -> dict:
        raw = txn.get(key)
        if raw is None:
            raise FileNotFoundError(f"Missing key {key!r} in LMDB database")
        return json.loads(bytes(raw))
