from __future__ import annotations

import enum
from typing import Any, NamedTuple, Protocol

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

Metadata = dict[str, Any]

# ---------------------------------------------------------------------------
# Load mode
# ---------------------------------------------------------------------------

LoadMode = enum.StrEnum("LoadMode", ["EAGER", "LAZY", "MMAP"])

# ---------------------------------------------------------------------------
# Store dtype
# ---------------------------------------------------------------------------


class StoreDtype(enum.Enum):
    """Specifies how vectors are stored (and whether quantisation is applied)."""

    FLOAT64 = "float64"
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    INT8_SYM = "int8_sym"
    INT8_ASYM = "int8_asym"
    FP4 = "fp4"
    INT4 = "int4"
    INT3 = "int3"
    TURBO_2BIT = "turbo_2bit"
    TURBO_3BIT = "turbo_3bit"
    TURBO_4BIT = "turbo_4bit"

    # -- helpers --------------------------------------------------------------

    @property
    def is_quantized(self) -> bool:
        return self in _QUANTIZED

    @property
    def is_turboquant(self) -> bool:
        return self in _TURBOQUANT

    @property
    def numpy_dtype(self) -> np.dtype:
        """The numpy dtype used for the stored array on disk / in RAM."""
        return _NUMPY_DTYPES[self]


_QUANTIZED: frozenset[StoreDtype] = frozenset(
    {
        StoreDtype.INT8_SYM,
        StoreDtype.INT8_ASYM,
        StoreDtype.FP4,
        StoreDtype.INT4,
        StoreDtype.INT3,
        StoreDtype.TURBO_2BIT,
        StoreDtype.TURBO_3BIT,
        StoreDtype.TURBO_4BIT,
    }
)

_TURBOQUANT: frozenset[StoreDtype] = frozenset(
    {StoreDtype.TURBO_2BIT, StoreDtype.TURBO_3BIT, StoreDtype.TURBO_4BIT}
)

_NUMPY_DTYPES: dict[StoreDtype, np.dtype] = {
    StoreDtype.FLOAT64: np.dtype(np.float64),
    StoreDtype.FLOAT32: np.dtype(np.float32),
    StoreDtype.FLOAT16: np.dtype(np.float16),
    StoreDtype.BFLOAT16: np.dtype(np.uint16),  # reinterpreted via view
    StoreDtype.INT8_SYM: np.dtype(np.int8),
    StoreDtype.INT8_ASYM: np.dtype(np.int8),
    StoreDtype.FP4: np.dtype(np.uint8),  # packed: 2 values per byte (E2M1)
    StoreDtype.INT4: np.dtype(np.uint8),  # packed: 2 values per byte
    StoreDtype.INT3: np.dtype(np.uint8),  # packed: 8 values per 3 bytes
    StoreDtype.TURBO_2BIT: np.dtype(np.uint8),  # packed: 4 values per byte
    StoreDtype.TURBO_3BIT: np.dtype(np.uint8),  # packed: 8 values per 3 bytes
    StoreDtype.TURBO_4BIT: np.dtype(np.uint8),  # packed: 2 values per byte
}

# ---------------------------------------------------------------------------
# Metric function signature
# ---------------------------------------------------------------------------


class MetricFn(Protocol):
    """Callable that scores a single query vector against a candidate matrix.

    Parameters
    ----------
    query : 1-D array (dim,)
    candidates : 2-D array (n, dim)

    Returns
    -------
    scores : 1-D array (n,)  – higher = more similar.
    """

    def __call__(
        self,
        query: NDArray[np.floating],
        candidates: NDArray[np.floating],
    ) -> NDArray[np.floating]: ...


# ---------------------------------------------------------------------------
# Query result
# ---------------------------------------------------------------------------


class QueryResult(NamedTuple):
    key: str
    score: float
    metadata: Metadata | None
