"""Backend protocol – the interface every persistence backend must implement."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from tqvs._types import LoadMode, Metadata, StoreDtype

# ---------------------------------------------------------------------------
# Manifest type
# ---------------------------------------------------------------------------

Manifest = dict[str, Any]  # {"dim": int, "dtype": str, "keys": list[str]}


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Backend(Protocol):
    """Persistence backend for a vector store."""

    def exists(self, path: Path) -> bool:
        """Return ``True`` if a store has been previously saved at *path*."""
        ...

    def load(
        self,
        path: Path,
        load_mode: LoadMode,
    ) -> tuple[np.ndarray | None, np.ndarray | None, Manifest]:
        """Load stored data from *path*.

        Returns
        -------
        vectors : np.ndarray | None
            The raw vector data (``None`` when *load_mode* is ``LAZY``).
        quant_params : np.ndarray | None
            Per-vector quantization parameters, or ``None`` if the store
            uses a float dtype.
        manifest : Manifest
            Dict with at least ``"dim"``, ``"dtype"`` (str name of
            :class:`StoreDtype`), ``"keys"`` (list[str]).
        """
        ...

    def load_metadata(self, path: Path) -> dict[str, Metadata]:
        """Load *only* the metadata mapping from *path*."""
        ...

    def save(
        self,
        path: Path,
        vectors: np.ndarray,
        quant_params: np.ndarray | None,
        manifest: Manifest,
        metadata: dict[str, Metadata],
    ) -> None:
        """Persist all store data to *path* atomically."""
        ...
