"""Built-in similarity / distance metrics with optional torch dispatch."""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray


def cosine_similarity(
    query: NDArray[np.floating],
    candidates: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Cosine similarity.  Higher = more similar."""
    query = np.asarray(query, dtype=np.float32)
    candidates = np.asarray(candidates, dtype=np.float32)

    dots = candidates @ query  # (n,)
    q_norm = np.linalg.norm(query)
    if q_norm == 0:
        return np.zeros(candidates.shape[0], dtype=np.float32)
    c_norms = np.linalg.norm(candidates, axis=1)
    c_norms = np.where(c_norms == 0, 1.0, c_norms)
    return dots / (q_norm * c_norms)


def dot_product(
    query: NDArray[np.floating],
    candidates: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Dot-product similarity.  Higher = more similar."""
    query = np.asarray(query, dtype=np.float32)
    candidates = np.asarray(candidates, dtype=np.float32)
    return candidates @ query


def euclidean_distance(
    query: NDArray[np.floating],
    candidates: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Negative Euclidean distance (higher = closer = more similar).

    Uses the identity ||x-q||^2 = ||x||^2 + ||q||^2 - 2*x·q to avoid
    materializing an (n, dim) difference array.
    """
    query = np.asarray(query, dtype=np.float32)
    candidates = np.asarray(candidates, dtype=np.float32)
    dots = candidates @ query  # (n,)
    q_sq = np.dot(query, query)  # scalar
    c_sq = np.einsum("ij,ij->i", candidates, candidates)  # (n,)
    dist_sq = c_sq + q_sq - 2.0 * dots
    # Clamp to zero to avoid sqrt of tiny negative from float rounding
    np.maximum(dist_sq, 0.0, out=dist_sq)
    return -np.sqrt(dist_sq)


# ---------------------------------------------------------------------------
# Torch-accelerated variants
# ---------------------------------------------------------------------------

def _has_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def cosine_similarity_torch(
    query: NDArray[np.floating],
    candidates: NDArray[np.floating],
    device: str = "cpu",
) -> NDArray[np.floating]:
    import torch

    q = torch.as_tensor(query, dtype=torch.float32, device=device)
    c = torch.as_tensor(candidates, dtype=torch.float32, device=device)
    sim = torch.nn.functional.cosine_similarity(q.unsqueeze(0), c, dim=1)
    return sim.cpu().numpy()


def dot_product_torch(
    query: NDArray[np.floating],
    candidates: NDArray[np.floating],
    device: str = "cpu",
) -> NDArray[np.floating]:
    import torch

    q = torch.as_tensor(query, dtype=torch.float32, device=device)
    c = torch.as_tensor(candidates, dtype=torch.float32, device=device)
    return (c @ q).cpu().numpy()


def euclidean_distance_torch(
    query: NDArray[np.floating],
    candidates: NDArray[np.floating],
    device: str = "cpu",
) -> NDArray[np.floating]:
    import torch

    q = torch.as_tensor(query, dtype=torch.float32, device=device)
    c = torch.as_tensor(candidates, dtype=torch.float32, device=device)
    return -torch.norm(c - q, dim=1).cpu().numpy()


# -- Batch torch variants (transfer candidates once) -----------------------

def cosine_similarity_batch_torch(
    queries: NDArray[np.floating],
    candidates: NDArray[np.floating],
    device: str = "cpu",
) -> NDArray[np.floating]:
    """Batched cosine similarity: (N, dim) queries × (M, dim) candidates → (N, M)."""
    import torch

    q = torch.as_tensor(queries, dtype=torch.float32, device=device)   # (N, D)
    c = torch.as_tensor(candidates, dtype=torch.float32, device=device)  # (M, D)
    q_norm = q / q.norm(dim=1, keepdim=True).clamp(min=1e-8)
    c_norm = c / c.norm(dim=1, keepdim=True).clamp(min=1e-8)
    return (q_norm @ c_norm.T).cpu().numpy()  # (N, M)


def dot_product_batch_torch(
    queries: NDArray[np.floating],
    candidates: NDArray[np.floating],
    device: str = "cpu",
) -> NDArray[np.floating]:
    """Batched dot product: (N, dim) × (M, dim) → (N, M)."""
    import torch

    q = torch.as_tensor(queries, dtype=torch.float32, device=device)
    c = torch.as_tensor(candidates, dtype=torch.float32, device=device)
    return (q @ c.T).cpu().numpy()


def euclidean_distance_batch_torch(
    queries: NDArray[np.floating],
    candidates: NDArray[np.floating],
    device: str = "cpu",
) -> NDArray[np.floating]:
    """Batched neg-euclidean: (N, dim) × (M, dim) → (N, M)."""
    import torch

    q = torch.as_tensor(queries, dtype=torch.float32, device=device)
    c = torch.as_tensor(candidates, dtype=torch.float32, device=device)
    return -torch.cdist(q, c).cpu().numpy()


# Map numpy metrics → torch equivalents
_TORCH_DISPATCH: dict = {
    cosine_similarity: cosine_similarity_torch,
    dot_product: dot_product_torch,
    euclidean_distance: euclidean_distance_torch,
}

_TORCH_BATCH_DISPATCH: dict = {
    cosine_similarity: cosine_similarity_batch_torch,
    dot_product: dot_product_batch_torch,
    euclidean_distance: euclidean_distance_batch_torch,
}


def resolve_metric(
    metric: Callable,
    device: str | None,
) -> tuple[Callable, str | None]:
    """Return (metric_fn, device) with torch dispatch when appropriate.

    If *device* is set and torch is available and *metric* has a known torch
    variant, return the torch variant.  Otherwise return the original metric.
    """
    if device and _has_torch() and metric in _TORCH_DISPATCH:
        return _TORCH_DISPATCH[metric], device
    return metric, None


def resolve_batch_metric(
    metric: Callable,
    device: str | None,
) -> tuple[Callable, str | None]:
    """Like :func:`resolve_metric` but returns the *batch* torch variant."""
    if device and _has_torch() and metric in _TORCH_BATCH_DISPATCH:
        return _TORCH_BATCH_DISPATCH[metric], device
    return metric, None
