"""Tests for quantize / dequantize round-trips across all StoreDtype levels."""

import math

import numpy as np
import pytest

from tqvs._types import StoreDtype
from tqvs.quantize import (
    dequantize,
    make_rotation_matrix,
    pack_int3,
    pack_int4,
    quantize,
    unpack_int3,
    unpack_int4,
)

RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Float dtype round-trips (lossless or near-lossless)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype",
    [StoreDtype.FLOAT64, StoreDtype.FLOAT32, StoreDtype.FLOAT16, StoreDtype.BFLOAT16],
)
def test_float_round_trip(dtype: StoreDtype) -> None:
    vecs = RNG.standard_normal((8, 64)).astype(np.float32)
    data, qp = quantize(vecs, dtype)
    assert qp is None
    restored = dequantize(data, dtype, None, 64)
    if dtype in (StoreDtype.FLOAT64, StoreDtype.FLOAT32):
        np.testing.assert_allclose(restored, vecs, atol=1e-6)
    elif dtype is StoreDtype.FLOAT16:
        np.testing.assert_allclose(restored, vecs, atol=1e-2)
    else:  # BFLOAT16 — truncation loses ~8 bits of mantissa
        np.testing.assert_allclose(restored, vecs, atol=2e-2)


# ---------------------------------------------------------------------------
# INT8 symmetric
# ---------------------------------------------------------------------------


def test_int8_sym_round_trip() -> None:
    vecs = RNG.standard_normal((16, 128)).astype(np.float32)
    data, qp = quantize(vecs, StoreDtype.INT8_SYM)
    assert data.dtype == np.int8
    assert qp is not None and qp.shape == (16, 1)
    restored = dequantize(data, StoreDtype.INT8_SYM, qp, 128)
    # Per-element error bounded by scale = max(|v|)/127
    max_err = (np.max(np.abs(vecs), axis=1, keepdims=True) / 127.0)
    assert np.all(np.abs(restored - vecs) <= max_err + 1e-6)


# ---------------------------------------------------------------------------
# INT8 asymmetric
# ---------------------------------------------------------------------------


def test_int8_asym_round_trip() -> None:
    vecs = RNG.uniform(-3, 5, (16, 128)).astype(np.float32)
    data, qp = quantize(vecs, StoreDtype.INT8_ASYM)
    assert qp is not None and qp.shape == (16, 2)
    restored = dequantize(data, StoreDtype.INT8_ASYM, qp, 128)
    scale = qp[:, 0:1]
    assert np.all(np.abs(restored - vecs) <= scale + 1e-5)


# ---------------------------------------------------------------------------
# INT4
# ---------------------------------------------------------------------------


def test_int4_round_trip() -> None:
    vecs = RNG.standard_normal((8, 33)).astype(np.float32)  # odd dim
    data, qp = quantize(vecs, StoreDtype.INT4)
    assert data.dtype == np.uint8
    assert qp is not None and qp.shape == (8, 1)
    restored = dequantize(data, StoreDtype.INT4, qp, 33)
    assert restored.shape == (8, 33)
    # Error bound: scale = max(|v|)/7
    scale = qp[:, 0:1]
    assert np.all(np.abs(restored - vecs) <= scale + 1e-5)


def test_pack_unpack_int4() -> None:
    vals = np.array([[-8, 7, 0, -1, 3, -4, 6, 2, -3]], dtype=np.int8)
    packed = pack_int4(vals)
    unpacked = unpack_int4(packed, 9)
    np.testing.assert_array_equal(unpacked, vals)


def test_pack_unpack_int4_even() -> None:
    vals = RNG.integers(-8, 8, size=(4, 16), dtype=np.int8)
    packed = pack_int4(vals)
    unpacked = unpack_int4(packed, 16)
    np.testing.assert_array_equal(unpacked, vals)


# ---------------------------------------------------------------------------
# INT3
# ---------------------------------------------------------------------------


def test_int3_round_trip() -> None:
    vecs = RNG.standard_normal((8, 25)).astype(np.float32)  # not multiple of 8
    data, qp = quantize(vecs, StoreDtype.INT3)
    assert data.dtype == np.uint8
    assert qp is not None and qp.shape == (8, 1)
    restored = dequantize(data, StoreDtype.INT3, qp, 25)
    assert restored.shape == (8, 25)
    scale = qp[:, 0:1]
    assert np.all(np.abs(restored - vecs) <= scale + 1e-5)


def test_pack_unpack_int3() -> None:
    vals = np.array([[-4, 3, 0, -1, 2, -3, 1, -2]], dtype=np.int8)
    packed = pack_int3(vals)
    unpacked = unpack_int3(packed, 8)
    np.testing.assert_array_equal(unpacked, vals)


def test_pack_unpack_int3_non_multiple() -> None:
    vals = RNG.integers(-4, 4, size=(4, 11), dtype=np.int8)
    packed = pack_int3(vals)
    unpacked = unpack_int3(packed, 11)
    np.testing.assert_array_equal(unpacked, vals)


# ---------------------------------------------------------------------------
# 1-D input
# ---------------------------------------------------------------------------


def test_quantize_1d_input() -> None:
    vec = RNG.standard_normal(64).astype(np.float32)
    data, qp = quantize(vec, StoreDtype.INT8_SYM)
    assert data.shape == (1, 64)


# ---------------------------------------------------------------------------
# TurboQuant round-trips
# ---------------------------------------------------------------------------

DIM_TURBO = 128  # Moderate dimension for reliable Beta → Gaussian convergence


@pytest.fixture(scope="module")
def rotation() -> np.ndarray:
    return make_rotation_matrix(DIM_TURBO, seed=42)


@pytest.mark.parametrize(
    "dtype, bits, max_mse",
    [
        (StoreDtype.TURBO_2BIT, 2, 0.20),
        (StoreDtype.TURBO_3BIT, 3, 0.06),
        (StoreDtype.TURBO_4BIT, 4, 0.02),
    ],
)
def test_turbo_round_trip(dtype: StoreDtype, bits: int, max_mse: float, rotation: np.ndarray) -> None:
    vecs = RNG.standard_normal((8, DIM_TURBO)).astype(np.float32)
    data, qp = quantize(vecs, dtype, rotation_matrix=rotation)
    assert data.dtype == np.uint8
    assert qp is not None and qp.shape == (8, 1)
    restored = dequantize(data, dtype, qp, DIM_TURBO, rotation_matrix=rotation)
    assert restored.shape == (8, DIM_TURBO)
    # Check MSE on unit-norm vectors (TurboQuant's primary guarantee)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit_orig = vecs / norms
    unit_rest = restored / norms  # same scale
    mse = np.mean(np.sum((unit_orig - unit_rest) ** 2, axis=1))
    assert mse < max_mse, f"{bits}-bit MSE {mse:.4f} > {max_mse}"
    # Cosine similarity should be high for 3+ bits
    cosine = np.sum(vecs * restored, axis=1) / (
        np.linalg.norm(vecs, axis=1) * np.linalg.norm(restored, axis=1) + 1e-10
    )
    if bits >= 3:
        assert np.all(cosine > 0.9), f"Cosine too low for {bits}-bit: {cosine}"


def test_turbo_zero_vector(rotation: np.ndarray) -> None:
    vecs = np.zeros((2, DIM_TURBO), dtype=np.float32)
    data, qp = quantize(vecs, StoreDtype.TURBO_4BIT, rotation_matrix=rotation)
    restored = dequantize(data, StoreDtype.TURBO_4BIT, qp, DIM_TURBO, rotation_matrix=rotation)
    np.testing.assert_allclose(restored, 0.0, atol=1e-6)


def test_turbo_1d_input(rotation: np.ndarray) -> None:
    vec = RNG.standard_normal(DIM_TURBO).astype(np.float32)
    data, qp = quantize(vec, StoreDtype.TURBO_3BIT, rotation_matrix=rotation)
    assert data.shape[0] == 1
    assert qp is not None and qp.shape == (1, 1)


def test_turbo_requires_rotation() -> None:
    vecs = RNG.standard_normal((2, DIM_TURBO)).astype(np.float32)
    with pytest.raises(ValueError, match="rotation_matrix"):
        quantize(vecs, StoreDtype.TURBO_2BIT)


def test_rotation_matrix_deterministic() -> None:
    r1 = make_rotation_matrix(64, seed=123)
    r2 = make_rotation_matrix(64, seed=123)
    np.testing.assert_array_equal(r1, r2)


def test_rotation_matrix_orthogonal() -> None:
    dim = 64
    r = make_rotation_matrix(dim, seed=99)
    product = r @ r.T
    np.testing.assert_allclose(product, np.eye(dim), atol=1e-10)
