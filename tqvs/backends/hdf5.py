"""HDF5 persistence backend."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np

try:
    import h5py
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The h5py package is required for Hdf5Backend. "
        "Install it with:  pip install tqvs[hdf5]"
    ) from exc

from tqvs._types import LoadMode, Metadata
from tqvs.backends.base import Manifest

_STORE_FILE = "store.h5"


class Hdf5Backend:
    """Persistence backend that stores vectors in an HDF5 file."""

    # -- Backend protocol -----------------------------------------------------

    def exists(self, path: Path) -> bool:
        return (path / _STORE_FILE).exists()

    def load(
        self,
        path: Path,
        load_mode: LoadMode,
    ) -> tuple[np.ndarray | None, np.ndarray | None, Manifest]:
        filepath = path / _STORE_FILE
        with h5py.File(filepath, "r") as f:
            manifest: Manifest = json.loads(str(f.attrs["manifest"]))

            vectors: np.ndarray | None = None
            quant_params: np.ndarray | None = None

            if load_mode is not LoadMode.LAZY:
                vectors = np.array(f["vectors"])
                if "quant_params" in f:
                    quant_params = np.array(f["quant_params"])

                if load_mode is LoadMode.MMAP:
                    # HDF5 doesn't support true OS mmap; simulate by
                    # returning read-only copies to match the NPY contract.
                    vectors.flags.writeable = False
                    if quant_params is not None:
                        quant_params.flags.writeable = False

        return vectors, quant_params, manifest

    def load_metadata(self, path: Path) -> dict[str, Metadata]:
        filepath = path / _STORE_FILE
        if not filepath.exists():
            return {}
        with h5py.File(filepath, "r") as f:
            raw = f.attrs.get("metadata")
            if raw is None:
                return {}
            return json.loads(raw)

    def save(
        self,
        path: Path,
        vectors: np.ndarray,
        quant_params: np.ndarray | None,
        manifest: Manifest,
        metadata: dict[str, Metadata],
    ) -> None:
        path.mkdir(parents=True, exist_ok=True)
        target = path / _STORE_FILE

        # Write to temp file, then rename for atomicity.
        fd, tmp = tempfile.mkstemp(dir=path, suffix=".h5.tmp")
        os.close(fd)
        try:
            with h5py.File(tmp, "w") as f:
                f.create_dataset("vectors", data=vectors)
                if quant_params is not None:
                    f.create_dataset("quant_params", data=quant_params)
                f.attrs["manifest"] = json.dumps(
                    manifest, ensure_ascii=False
                )
                f.attrs["metadata"] = json.dumps(
                    metadata, ensure_ascii=False
                )
            os.replace(tmp, target)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
