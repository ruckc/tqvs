""".npy persistence backend."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np

from tqvs._types import LoadMode, Metadata
from tqvs.backends.base import Manifest

_VECTORS_FILE = "vectors.npy"
_QUANT_PARAMS_FILE = "quant_params.npy"
_MANIFEST_FILE = "manifest.json"
_METADATA_FILE = "metadata.json"


class NpyBackend:
    """Persistence backend that stores vectors as ``.npy`` files."""

    # -- Backend protocol -----------------------------------------------------

    def exists(self, path: Path) -> bool:
        return (path / _VECTORS_FILE).exists()

    def load(
        self,
        path: Path,
        load_mode: LoadMode,
    ) -> tuple[np.ndarray | None, np.ndarray | None, Manifest]:
        manifest = self._read_json(path / _MANIFEST_FILE)

        vectors: np.ndarray | None = None
        quant_params: np.ndarray | None = None

        if load_mode is not LoadMode.LAZY:
            mmap_mode = "r" if load_mode is LoadMode.MMAP else None
            vectors = np.load(path / _VECTORS_FILE, mmap_mode=mmap_mode)

            qp_path = path / _QUANT_PARAMS_FILE
            if qp_path.exists():
                quant_params = np.load(qp_path, mmap_mode=mmap_mode)

        return vectors, quant_params, manifest

    def load_metadata(self, path: Path) -> dict[str, Metadata]:
        meta_path = path / _METADATA_FILE
        if meta_path.exists():
            return self._read_json(meta_path)
        return {}

    def save(
        self,
        path: Path,
        vectors: np.ndarray,
        quant_params: np.ndarray | None,
        manifest: Manifest,
        metadata: dict[str, Metadata],
    ) -> None:
        path.mkdir(parents=True, exist_ok=True)
        # Write to temp files, then rename for atomicity.
        self._atomic_npy(path / _VECTORS_FILE, vectors)
        if quant_params is not None:
            self._atomic_npy(path / _QUANT_PARAMS_FILE, quant_params)
        else:
            # Remove stale quant_params if dtype changed to float
            qp = path / _QUANT_PARAMS_FILE
            if qp.exists():
                qp.unlink()
        self._atomic_json(path / _MANIFEST_FILE, manifest)
        self._atomic_json(path / _METADATA_FILE, metadata)

    # -- helpers --------------------------------------------------------------

    @staticmethod
    def _atomic_npy(target: Path, array: np.ndarray) -> None:
        # Use .npy suffix so np.save doesn't append another .npy
        fd, tmp = tempfile.mkstemp(dir=target.parent, suffix=".npy")
        os.close(fd)  # close immediately; np.save will reopen by path
        try:
            np.save(tmp, array)
            os.replace(tmp, target)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    @staticmethod
    def _atomic_json(target: Path, data: dict) -> None:
        fd, tmp = tempfile.mkstemp(dir=target.parent, suffix=".json.tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            os.replace(tmp, target)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    @staticmethod
    def _read_json(path: Path) -> dict:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
