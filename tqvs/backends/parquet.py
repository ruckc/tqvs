"""Parquet (Arrow IPC) persistence backend."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The pyarrow package is required for ParquetBackend. "
        "Install it with:  pip install tqvs[parquet]"
    ) from exc

from tqvs._types import LoadMode, Metadata
from tqvs.backends.base import Manifest

_STORE_FILE = "store.parquet"
_META_KEY_MANIFEST = b"tqvs_manifest"
_META_KEY_METADATA = b"tqvs_metadata"


class ParquetBackend:
    """Persistence backend that stores vectors in a Parquet file."""

    # -- Backend protocol -----------------------------------------------------

    def exists(self, path: Path) -> bool:
        return (path / _STORE_FILE).exists()

    def load(
        self,
        path: Path,
        load_mode: LoadMode,
    ) -> tuple[np.ndarray | None, np.ndarray | None, Manifest]:
        filepath = path / _STORE_FILE
        pf = pq.ParquetFile(str(filepath))
        schema_meta = pf.schema_arrow.metadata or {}
        manifest: Manifest = json.loads(schema_meta[_META_KEY_MANIFEST])

        vectors: np.ndarray | None = None
        quant_params: np.ndarray | None = None

        if load_mode is not LoadMode.LAZY:
            table = pf.read()
            vec_col = table.column("vectors")
            n = table.num_rows
            if n > 0:
                vec_dim = vec_col.type.list_size
                flat = vec_col.combine_chunks().values.to_numpy(zero_copy_only=False)
                vectors = flat.reshape(n, vec_dim)
            else:
                vectors = np.empty((0, 0))

            if "quant_params" in table.schema.names:
                qp_col = table.column("quant_params")
                if n > 0:
                    qp_dim = qp_col.type.list_size
                    flat_qp = qp_col.combine_chunks().values.to_numpy(zero_copy_only=False)
                    quant_params = flat_qp.reshape(n, qp_dim)
                else:
                    quant_params = None

            if load_mode is LoadMode.MMAP:
                # Parquet requires decompression — no true mmap.
                # Return read-only copies to match the contract.
                vectors = np.array(vectors)
                vectors.flags.writeable = False
                if quant_params is not None:
                    quant_params = np.array(quant_params)
                    quant_params.flags.writeable = False
            else:
                # EAGER — ensure writable contiguous copies
                vectors = np.ascontiguousarray(vectors)
                if quant_params is not None:
                    quant_params = np.ascontiguousarray(quant_params)

        return vectors, quant_params, manifest

    def load_metadata(self, path: Path) -> dict[str, Metadata]:
        filepath = path / _STORE_FILE
        if not filepath.exists():
            return {}
        pf = pq.ParquetFile(str(filepath))
        schema_meta = pf.schema_arrow.metadata or {}
        raw = schema_meta.get(_META_KEY_METADATA)
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

        # Build Arrow table with FixedSizeList columns
        vec_dim = vectors.shape[1]
        flat_vecs = vectors.reshape(-1)
        vec_arr = pa.FixedSizeListArray.from_arrays(
            pa.array(flat_vecs, type=_numpy_to_arrow_type(vectors.dtype)),
            list_size=vec_dim,
        )

        fields = [pa.field("vectors", vec_arr.type)]
        arrays = [vec_arr]

        if quant_params is not None:
            qp_dim = quant_params.shape[1]
            flat_qp = quant_params.reshape(-1)
            qp_arr = pa.FixedSizeListArray.from_arrays(
                pa.array(flat_qp, type=_numpy_to_arrow_type(quant_params.dtype)),
                list_size=qp_dim,
            )
            fields.append(pa.field("quant_params", qp_arr.type))
            arrays.append(qp_arr)

        # Store manifest and metadata in schema-level metadata
        schema_meta = {
            _META_KEY_MANIFEST: json.dumps(manifest, ensure_ascii=False).encode("utf-8"),
            _META_KEY_METADATA: json.dumps(metadata, ensure_ascii=False).encode("utf-8"),
        }
        schema = pa.schema(fields, metadata=schema_meta)
        table = pa.table(
            {f.name: a for f, a in zip(fields, arrays)},
            schema=schema,
        )

        # Atomic write via temp file + rename
        fd, tmp = tempfile.mkstemp(dir=path, suffix=".parquet.tmp")
        os.close(fd)
        try:
            pq.write_table(table, tmp)
            os.replace(tmp, target)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _numpy_to_arrow_type(dtype: np.dtype) -> pa.DataType:
    """Map numpy dtype to Arrow type."""
    mapping = {
        np.dtype(np.float32): pa.float32(),
        np.dtype(np.float64): pa.float64(),
        np.dtype(np.float16): pa.float16(),
        np.dtype(np.int8): pa.int8(),
        np.dtype(np.uint8): pa.uint8(),
        np.dtype(np.int16): pa.int16(),
        np.dtype(np.uint16): pa.uint16(),
        np.dtype(np.int32): pa.int32(),
        np.dtype(np.uint32): pa.uint32(),
    }
    if dtype in mapping:
        return mapping[dtype]
    raise ValueError(f"Unsupported numpy dtype for Arrow: {dtype}")
