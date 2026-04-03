"""Query engine – top-k, score-all, prefix filtering, native-dtype scoring."""

from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from tqvs._types import Metadata, QueryResult, StoreDtype
from tqvs.quantize import (
    dequantize,
    _get_turbo_codebook,
    _unpack_turbo2,
    _unpack_turbo3,
    _unpack_turbo4,
    unpack_int4,
    unpack_int3,
)
from tqvs.metrics import cosine_similarity, dot_product, resolve_metric, resolve_batch_metric

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def top_k(
    query_vec: NDArray[np.floating],
    vectors: np.ndarray | None,
    keys: list[str],
    metadata: dict[str, Metadata],
    k: int,
    metric: Callable,
    dtype: StoreDtype,
    quant_params: np.ndarray | None,
    dim: int,
    *,
    prefix: str | None = None,
    prefix_indices: list[int] | None = None,
    device: str | None = None,
    rotation_matrix: np.ndarray | None = None,
) -> list[QueryResult]:
    """Return the *k* most-similar entries (descending score).

    Uses ``np.argpartition`` to select the top-k in O(n) rather than
    sorting all scores in O(n log n).
    """
    scores = _score_vectors(
        query_vec, vectors, keys, metadata, metric, dtype,
        quant_params, dim, prefix=prefix,
        prefix_indices=prefix_indices, device=device,
        rotation_matrix=rotation_matrix,
    )
    if scores is None:
        return []

    score_arr, sub_keys = scores

    n = len(score_arr)
    actual_k = min(k, n)

    if actual_k >= n:
        # Need all results — just argsort
        top_idx = np.argsort(score_arr)[::-1]
    else:
        # O(n) partition to find top-k, then sort only those k
        top_idx = np.argpartition(score_arr, -actual_k)[-actual_k:]
        top_idx = top_idx[np.argsort(score_arr[top_idx])[::-1]]

    # Build QueryResult only for top-k entries
    return [
        QueryResult(
            key=sub_keys[i],
            score=float(score_arr[i]),
            metadata=metadata.get(sub_keys[i]),
        )
        for i in top_idx
    ]


def score_all(
    query_vec: NDArray[np.floating],
    vectors: np.ndarray | None,
    keys: list[str],
    metadata: dict[str, Metadata],
    metric: Callable,
    dtype: StoreDtype,
    quant_params: np.ndarray | None,
    dim: int,
    *,
    prefix: str | None = None,
    prefix_indices: list[int] | None = None,
    device: str | None = None,
    rotation_matrix: np.ndarray | None = None,
) -> list[QueryResult]:
    """Score a query against all (or prefix-matched) vectors."""
    result = _score_vectors(
        query_vec, vectors, keys, metadata, metric, dtype,
        quant_params, dim, prefix=prefix,
        prefix_indices=prefix_indices, device=device,
        rotation_matrix=rotation_matrix,
    )
    if result is None:
        return []

    score_arr, sub_keys = result

    return [
        QueryResult(
            key=sub_keys[i],
            score=float(score_arr[i]),
            metadata=metadata.get(sub_keys[i]),
        )
        for i in range(len(score_arr))
    ]


def score_array_raw(
    query_vec: NDArray[np.floating],
    vectors: np.ndarray | None,
    keys: list[str],
    metadata: dict[str, Metadata],
    metric: Callable,
    dtype: StoreDtype,
    quant_params: np.ndarray | None,
    dim: int,
    *,
    prefix: str | None = None,
    prefix_indices: list[int] | None = None,
    device: str | None = None,
    rotation_matrix: np.ndarray | None = None,
) -> np.ndarray:
    """Score query against all vectors, returning raw float32 score array.

    Unlike :func:`score_all`, this returns a plain numpy array in insertion
    order (no QueryResult wrapping).  When *prefix* is set the returned array
    covers only the matching subset.
    """
    result = _score_vectors(
        query_vec, vectors, keys, metadata, metric, dtype,
        quant_params, dim, prefix=prefix,
        prefix_indices=prefix_indices, device=device,
        rotation_matrix=rotation_matrix,
    )
    if result is None:
        return np.empty(0, dtype=np.float32)
    return result[0]


def score_batch(
    query_vecs: NDArray[np.floating],
    vectors: np.ndarray | None,
    keys: list[str],
    metric: Callable,
    dtype: StoreDtype,
    quant_params: np.ndarray | None,
    dim: int,
    *,
    device: str | None = None,
    rotation_matrix: np.ndarray | None = None,
) -> np.ndarray:
    """Score multiple queries in one shot, returning (N, M) score matrix.

    Uses batched torch metric when available to transfer candidates to GPU
    only once.  Falls back to per-query loop otherwise.

    Prefix filtering is not supported here — callers should pre-filter if
    needed.
    """
    query_vecs = np.asarray(query_vecs, dtype=np.float32)
    n = query_vecs.shape[0]

    if vectors is None or len(keys) == 0:
        return np.empty((n, 0), dtype=np.float32)

    # -- prepare candidates (dequantize once) ---------------------------------
    fast_scores_first = _try_score_quantized(
        query_vecs[0], vectors, dtype, quant_params, dim, metric, rotation_matrix,
    )
    if fast_scores_first is not None:
        # Quantized fast path exists — use per-query loop
        m = len(fast_scores_first)
        out = np.empty((n, m), dtype=np.float32)
        out[0] = fast_scores_first
        for i in range(1, n):
            out[i] = _try_score_quantized(
                query_vecs[i], vectors, dtype, quant_params, dim, metric, rotation_matrix,
            )
        return out

    candidates = _prepare_candidates(
        vectors, dtype, quant_params, dim, rotation_matrix=rotation_matrix,
    )

    # -- try batched torch dispatch -------------------------------------------
    batch_metric, resolved_device = resolve_batch_metric(metric, device)
    if resolved_device is not None:
        return batch_metric(query_vecs, candidates, device=resolved_device)

    # -- CPU fallback: per-query loop -----------------------------------------
    m = candidates.shape[0]
    out = np.empty((n, m), dtype=np.float32)
    for i in range(n):
        out[i] = metric(query_vecs[i], candidates)
    return out


def _score_vectors(
    query_vec: NDArray[np.floating],
    vectors: np.ndarray | None,
    keys: list[str],
    metadata: dict[str, Metadata],
    metric: Callable,
    dtype: StoreDtype,
    quant_params: np.ndarray | None,
    dim: int,
    *,
    prefix: str | None = None,
    prefix_indices: list[int] | None = None,
    device: str | None = None,
    rotation_matrix: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str]] | None:
    """Compute scores, returning (scores_array, sub_keys) or None if empty."""
    if vectors is None or len(keys) == 0:
        return None

    # -- prefix filtering (use pre-built index when available) ----------------
    if prefix is not None:
        if prefix_indices is not None:
            # Fast path: pre-built prefix index
            idx_arr = np.array(prefix_indices, dtype=np.intp)
            if len(idx_arr) == 0:
                return None
            sub_keys: list[str] = [keys[i] for i in prefix_indices]
            sub_vectors = vectors[idx_arr]
            sub_qp = quant_params[idx_arr] if quant_params is not None else None
        else:
            mask = np.array([k.startswith(prefix) for k in keys], dtype=bool)
            if not mask.any():
                return None
            sub_keys = [keys[i] for i, m in enumerate(mask) if m]
            sub_vectors = vectors[mask]
            sub_qp = quant_params[mask] if quant_params is not None else None
    else:
        sub_keys = keys
        sub_vectors = vectors
        sub_qp = quant_params

    # -- prepare candidates ---------------------------------------------------
    query = np.asarray(query_vec, dtype=np.float32)

    # -- try quantized-domain fast path ---------------------------------------
    fast_scores = _try_score_quantized(
        query, sub_vectors, dtype, sub_qp, dim, metric, rotation_matrix,
    )
    if fast_scores is not None:
        scores: np.ndarray = fast_scores
    else:
        candidates = _prepare_candidates(
            sub_vectors, dtype, sub_qp, dim, rotation_matrix=rotation_matrix,
        )

        # -- resolve torch dispatch -------------------------------------------
        resolved_metric, resolved_device = resolve_metric(metric, device)

        # -- score ------------------------------------------------------------
        if resolved_device is not None:
            scores = resolved_metric(query, candidates, device=resolved_device)
        else:
            scores = resolved_metric(query, candidates)

    return scores, sub_keys


# ---------------------------------------------------------------------------
# Internal – native-dtype scoring preparation
# ---------------------------------------------------------------------------


def _prepare_candidates(
    vectors: np.ndarray,
    dtype: StoreDtype,
    quant_params: np.ndarray | None,
    dim: int,
    *,
    rotation_matrix: np.ndarray | None = None,
) -> NDArray[np.float32]:
    """Convert stored vectors to float32 for metric evaluation.

    For float dtypes this is a simple cast; for quantised dtypes the
    vectors are dequantised first.
    """
    if dtype.is_quantized:
        return dequantize(vectors, dtype, quant_params, dim, rotation_matrix=rotation_matrix)

    if dtype is StoreDtype.BFLOAT16:
        return dequantize(vectors, dtype, None, dim)

    # float64 / float32 / float16
    return vectors.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Quantized-domain fast paths
# ---------------------------------------------------------------------------

# Set of metrics that support quantized fast paths
_QUANT_FAST_METRICS = {cosine_similarity, dot_product}


def _try_score_quantized(
    query: NDArray[np.float32],
    vectors: np.ndarray,
    dtype: StoreDtype,
    quant_params: np.ndarray | None,
    dim: int,
    metric: Callable,
    rotation_matrix: np.ndarray | None,
) -> np.ndarray | None:
    """Attempt to score directly on quantized data, returning None on fallback.

    Returns scores (shape ``(n,)``) if a fast path applies, else ``None``.
    """
    if metric not in _QUANT_FAST_METRICS:
        return None

    if dtype is StoreDtype.INT8_SYM:
        return _score_int8_sym(query, vectors, quant_params, metric)

    if dtype is StoreDtype.INT8_ASYM:
        return _score_int8_asym(query, vectors, quant_params, dim, metric)

    if dtype is StoreDtype.INT4:
        return _score_int4(query, vectors, quant_params, dim, metric)

    if dtype is StoreDtype.INT3:
        return _score_int3(query, vectors, quant_params, dim, metric)

    if dtype.is_turboquant:
        bits = {
            StoreDtype.TURBO_2BIT: 2,
            StoreDtype.TURBO_3BIT: 3,
            StoreDtype.TURBO_4BIT: 4,
        }[dtype]
        return _score_turbo_adc(
            query, vectors, quant_params, dim, bits, rotation_matrix, metric,
        )

    return None


# ---------------------------------------------------------------------------
# INT8_SYM: direct scoring without dequantization
# ---------------------------------------------------------------------------


def _score_int8_sym(
    query: NDArray[np.float32],
    data: np.ndarray,
    quant_params: np.ndarray | None,
    metric: Callable,
) -> np.ndarray:
    """Score INT8 symmetric quantized data against a float32 query."""
    # Compute integer dot products — let numpy promote int8×float32 → float32
    dots = np.dot(data, query)  # (n,) — avoids explicit .astype(float32) copy

    if metric is cosine_similarity:
        q_norm = np.linalg.norm(query)
        if q_norm == 0:
            return np.zeros(data.shape[0], dtype=np.float32)
        # Compute norms of int8 rows efficiently via squared-sum
        d_norms_sq = np.einsum("ij,ij->i", data.astype(np.float32), data.astype(np.float32))
        d_norms = np.sqrt(d_norms_sq)
        d_norms = np.where(d_norms == 0, 1.0, d_norms)
        return dots / (q_norm * d_norms)

    # dot_product: scale back
    assert quant_params is not None
    scales = quant_params[:, 0]  # (n,)
    return dots * scales


# ---------------------------------------------------------------------------
# INT8_ASYM: direct scoring without full dequantization
# ---------------------------------------------------------------------------


def _score_int8_asym(
    query: NDArray[np.float32],
    data: np.ndarray,
    quant_params: np.ndarray | None,
    dim: int,
    metric: Callable,
) -> np.ndarray:
    """Score INT8 asymmetric quantized data against a float32 query."""
    assert quant_params is not None
    scale = quant_params[:, 0:1]  # (n, 1)
    zero_point = quant_params[:, 1:2]  # (n, 1)

    # Dequantize: float_val = (uint8_val - zero_point) * scale
    # dot(q, x) = scale * (uint8_data @ q - zero_point * sum(q))
    uint_data = data.view(np.uint8).astype(np.float32)  # (n, dim)
    raw_dots = uint_data @ query  # (n,)
    q_sum = query.sum()
    adjusted_dots = (raw_dots - zero_point[:, 0] * q_sum) * scale[:, 0]

    if metric is cosine_similarity:
        q_norm = np.linalg.norm(query)
        if q_norm == 0:
            return np.zeros(data.shape[0], dtype=np.float32)
        # Reconstruct float vectors for norms (cheaper than full dequant+matmul)
        float_vecs = (uint_data - zero_point) * scale
        c_norms = np.linalg.norm(float_vecs, axis=1)
        c_norms = np.where(c_norms == 0, 1.0, c_norms)
        return adjusted_dots / (q_norm * c_norms)

    return adjusted_dots


# ---------------------------------------------------------------------------
# INT4: direct scoring without full dequantization
# ---------------------------------------------------------------------------


def _score_int4(
    query: NDArray[np.float32],
    packed: np.ndarray,
    quant_params: np.ndarray | None,
    dim: int,
    metric: Callable,
) -> np.ndarray:
    """Score INT4 quantized data against a float32 query."""
    assert quant_params is not None
    scale = quant_params[:, 0:1]  # (n, 1)
    unpacked = unpack_int4(packed, dim).astype(np.float32)  # (n, dim)
    dots = unpacked @ query  # (n,)

    if metric is cosine_similarity:
        q_norm = np.linalg.norm(query)
        if q_norm == 0:
            return np.zeros(packed.shape[0], dtype=np.float32)
        d_norms = np.linalg.norm(unpacked, axis=1)
        d_norms = np.where(d_norms == 0, 1.0, d_norms)
        return (dots * scale[:, 0]) / (q_norm * d_norms * scale[:, 0])

    return dots * scale[:, 0]


# ---------------------------------------------------------------------------
# INT3: direct scoring without full dequantization
# ---------------------------------------------------------------------------


def _score_int3(
    query: NDArray[np.float32],
    packed: np.ndarray,
    quant_params: np.ndarray | None,
    dim: int,
    metric: Callable,
) -> np.ndarray:
    """Score INT3 quantized data against a float32 query."""
    assert quant_params is not None
    scale = quant_params[:, 0:1]  # (n, 1)
    unpacked = unpack_int3(packed, dim).astype(np.float32)  # (n, dim)
    dots = unpacked @ query  # (n,)

    if metric is cosine_similarity:
        q_norm = np.linalg.norm(query)
        if q_norm == 0:
            return np.zeros(packed.shape[0], dtype=np.float32)
        d_norms = np.linalg.norm(unpacked, axis=1)
        d_norms = np.where(d_norms == 0, 1.0, d_norms)
        return (dots * scale[:, 0]) / (q_norm * d_norms * scale[:, 0])

    return dots * scale[:, 0]


# ---------------------------------------------------------------------------
# TurboQuant ADC: Asymmetric Distance Computation via lookup table
# ---------------------------------------------------------------------------


def _score_turbo_adc(
    query: NDArray[np.float32],
    packed: np.ndarray,
    quant_params: np.ndarray | None,
    dim: int,
    bits: int,
    rotation_matrix: np.ndarray | None,
    metric: Callable,
) -> np.ndarray:
    """Score TurboQuant vectors using ADC (no full dequantization).

    Instead of reconstructing all float vectors, we:
    1. Rotate the query into the quantized space.
    2. Build a lookup table: LUT[j, c] = rotated_query[j] * centroid[c]
       for each dimension j and code c.
    3. For each stored vector, unpack its code indices and sum the
       corresponding LUT entries to get the dot product.
    4. Apply norms for cosine, or scale for dot product.
    """
    assert rotation_matrix is not None
    assert quant_params is not None

    n = packed.shape[0]
    norms = quant_params[:, 0]  # (n,) stored norms

    # 1. Rotate query into quantized space
    rot = rotation_matrix.astype(np.float32)
    q_rotated = query @ rot.T  # (dim,)

    # 2. Build codebook and LUT
    codebook = _get_turbo_codebook(bits, dim).astype(np.float32)  # (num_levels,)
    num_levels = len(codebook)
    # LUT[c] for each level c, applied per-dimension via q_rotated
    # lut_flat[c] = codebook[c]; contribution = q_rotated[j] * lut_flat[indices[i,j]]
    # Precompute per-level dot contribution: for each code c, partial[c] = codebook[c]
    # dots[i] = sum_j q_rotated[j] * codebook[indices[i,j]]
    # Factor: accumulate per-level across dimensions to avoid (n, dim) intermediate
    lut = q_rotated[:, np.newaxis] * codebook[np.newaxis, :]  # (dim, num_levels)

    # 3. Unpack indices: (n, dim) of uint8 code indices
    if bits == 2:
        indices = _unpack_turbo2(packed, dim)
    elif bits == 3:
        indices = _unpack_turbo3(packed, dim)
    else:  # bits == 4
        indices = _unpack_turbo4(packed, dim)

    # 4. Sum LUT entries per-level to avoid large (n, dim) temporary.
    #    For each level c: dots += q_rotated[mask_dims] . 1  for all dims where indices == c
    #    Equivalent but uses level-accumulation: O(num_levels * n) vs O(n * dim)
    if n * dim > 500_000:
        # For large arrays, accumulate per-level to reduce peak memory
        dots = np.zeros(n, dtype=np.float32)
        for c in range(num_levels):
            # mask: (n, dim) bool — but we sum over dim immediately
            # Per dim j, contribution = q_rotated[j] * codebook[c] if indices[:,j]==c
            # = codebook[c] * sum_j(q_rotated[j] * (indices[:,j]==c))
            mask = (indices == c)  # (n, dim) bool
            dots += codebook[c] * (mask.astype(np.float32) @ q_rotated)
    else:
        dim_idx = np.arange(dim)
        dots = lut[dim_idx, indices].sum(axis=1)  # (n,)

    # 5. The dot products are in the rotated space of unit-norm vectors.
    if metric is cosine_similarity:
        q_norm = np.linalg.norm(query)
        if q_norm == 0:
            return np.zeros(n, dtype=np.float32)
        codebook_sq = codebook ** 2  # (num_levels,)
        recon_norms_sq = codebook_sq[indices].sum(axis=1)  # (n,)
        recon_norms = np.sqrt(recon_norms_sq)
        recon_norms = np.where(recon_norms == 0, 1.0, recon_norms)
        return dots / (q_norm * recon_norms)

    # dot_product: real_dot = norm * dot_in_unit_space
    return norms * dots
