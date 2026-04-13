# tqvs

Composable vector store with pluggable persistence, dtype quantization, and similarity search.

`tqvs` gives you a keyed vector store that works entirely in-process — no
server, no network calls. Vectors are quantized on the fly to the storage dtype
you choose, persisted through a swappable backend, and queried with brute-force
similarity search (with optional CUDA acceleration via PyTorch).

## Installation

```bash
pip install tqvs              # core (NumPy only, includes the npy backend)
pip install tqvs[hdf5]        # + HDF5 backend  (h5py)
pip install tqvs[lmdb]        # + LMDB backend  (lmdb)
pip install tqvs[lance]       # + Lance backend (pylance)
pip install tqvs[parquet]     # + Parquet backend (pyarrow)
pip install tqvs[torch]       # + GPU-accelerated metrics
pip install tqvs[all]         # all optional backends
```

Requires Python 3.12+.

## Quick start

```python
import numpy as np
from tqvs import create_vector_store, StoreDtype

store = create_vector_store("./my_store", dim=768, dtype=StoreDtype.INT8_SYM)

store.add("doc/1", np.random.rand(768).astype(np.float32), metadata={"title": "Hello"})
store.add("doc/2", np.random.rand(768).astype(np.float32))

results = store.query(np.random.rand(768).astype(np.float32), k=5)
for r in results:
    print(r.key, r.score)

store.save()
```

### Fluent builder

```python
from tqvs import VectorStoreBuilder, StoreDtype, Hdf5Backend, LoadMode

store = (
    VectorStoreBuilder()
    .at("./my_store")
    .with_dim(768)
    .with_dtype(StoreDtype.FLOAT16)
    .with_backend(Hdf5Backend())
    .with_load_mode(LoadMode.MMAP)
    .build()
)
```

### Batch insert

```python
keys = [f"vec/{i}" for i in range(10_000)]
vectors = np.random.rand(10_000, 768).astype(np.float32)
store.add_many(keys, vectors)
```

### Prefix-filtered queries

```python
# Search only keys starting with "doc/"
results = store.query(q, k=10, prefix="doc/")

# Iterate matching keys
for key in store.keys(prefix="img/"):
    print(key)
```

## Storage dtypes

Vectors are quantized at insert time and dequantized at query time. Smaller
dtypes reduce memory and disk usage at the cost of some precision.

| Dtype | Bits/value | Description |
|---|---|---|
| `FLOAT64` | 64 | Full double precision |
| `FLOAT32` | 32 | Single precision (default) |
| `FLOAT16` | 16 | IEEE half precision |
| `BFLOAT16` | 16 | Brain floating point |
| `INT8_SYM` | 8 | Symmetric linear quantization |
| `INT8_ASYM` | 8 | Asymmetric linear quantization |
| `FP4` | 4 | 4-bit mini-float (E2M1, 2 per byte) |
| `INT4` | 4 | 4-bit packed (2 per byte) |
| `INT3` | 3 | 3-bit packed (8 per 3 bytes) |
| `TURBO_2BIT` | 2 | TurboQuant with random rotation |
| `TURBO_3BIT` | 3 | TurboQuant with random rotation |
| `TURBO_4BIT` | 4 | TurboQuant with random rotation |

TurboQuant dtypes apply a random rotation before quantization to spread
quantization error more evenly across dimensions.

## Persistence backends

Five backends are available. They all share the same in-memory store — the
backend only determines how data is serialized to and from disk.

| Backend | Install extra | Dependency | Format on disk |
|---|---|---|---|
| `NpyBackend` | *(none)* | NumPy (core) | `.npy` files + JSON manifest |
| `Hdf5Backend` | `hdf5` | h5py | Single `.h5` file |
| `LmdbBackend` | `lmdb` | lmdb | LMDB database directory |
| `LanceBackend` | `lance` | pylance | Lance columnar dataset |
| `ParquetBackend` | `parquet` | pyarrow | `.parquet` files |

### Performance comparison

Benchmarked with 1,000,000 128-dimensional vectors, averaged across all 12
storage dtypes. In-memory operations (add, query, key lookup) are
backend-independent since the backend is only involved during save/reload.

#### Save throughput (vectors/sec, 1M vectors)

| Rank | Backend | Avg vec/s | Relative |
|---|---|---|---|
| 1 | **HDF5** | 7,840,000 | 1.0× |
| 2 | LMDB | 5,980,000 | 0.76× |
| 3 | Lance | 5,530,000 | 0.71× |
| 4 | NumPy | 4,530,000 | 0.58× |
| 5 | Parquet | 715,000 | 0.09× |

#### Reload throughput (vectors/sec, 1M vectors)

| Rank | Backend | Avg vec/s | Relative |
|---|---|---|---|
| 1 | **NumPy** | 1,860,000 | 1.0× |
| 2 | HDF5 | 1,700,000 | 0.91× |
| 3 | LMDB | 1,630,000 | 0.88× |
| 4 | Lance | 1,490,000 | 0.80× |
| 5 | Parquet | 523,000 | 0.28× |

#### In-memory operations (all backends equivalent)

| Operation | Avg vec/s |
|---|---|
| Key prefix scan | ~70,000,000 |
| Prefix query (brute-force) | ~4,000,000 |

### Choosing a backend

- **HDF5** — Best all-round choice. Fastest saves, near-fastest reloads. Single
  `.h5` file is easy to manage. Good default for most workloads.
- **NumPy (npy)** — Zero optional dependencies. Fastest reloads, solid saves.
  Choose this if you want to avoid extra packages.
- **LMDB** — Comparable I/O to HDF5/NumPy, and supports transactional
  semantics. Useful when concurrent read access matters.
- **Lance** — Slightly slower reloads than the top tier but still competitive.
  Best suited when you already use the Lance ecosystem and want a unified
  format.
- **Parquet** — Roughly 10× slower saves and 3–4× slower reloads than HDF5.
  Choose only when interop with Spark, DuckDB, Pandas, or other columnar
  tools is a priority.

### Dtype impact on I/O

Smaller quantized dtypes save and reload faster because there is less data
to write and read:

| Dtype | HDF5 save (s) | HDF5 reload (s) |
|---|---|---|
| `TURBO_2BIT` | 0.08 | 0.59 |
| `INT3` | 0.08 | 0.55 |
| `TURBO_3BIT` | 0.08 | 0.54 |
| `INT4` | 0.10 | 0.67 |
| `TURBO_4BIT` | 0.09 | 0.52 |
| `INT8_SYM` | 0.14 | 0.60 |
| `INT8_ASYM` | 0.14 | 0.65 |
| `BFLOAT16` | 0.21 | 0.54 |
| `FLOAT16` | 0.22 | 0.55 |
| `FLOAT32` | 0.39 | 0.62 |
| `FLOAT64` | 0.73 | 0.70 |

*(1M vectors, dim=128)*

### Disk usage

On-disk size depends primarily on the storage dtype and secondarily on the
backend format. NpyBackend and Hdf5Backend are the most space-efficient,
storing close to the raw vector data with minimal overhead. LMDB adds ~2×
overhead due to its B-tree page structure. Lance and Parquet add modest
columnar metadata.

*(100K vectors, dim=128)*

| Dtype | Npy / HDF5 | Lance | Parquet | LMDB |
|---|---|---|---|---|
| `FLOAT64` | 1,036 B/vec (98.8 MB) | 1,048 B/vec | 817 B/vec | 2,177 B/vec |
| `FLOAT32` | 524 B/vec (49.9 MB) | 536 B/vec | 547 B/vec | 1,153 B/vec |
| `FLOAT16` | 268 B/vec (25.6 MB) | 280 B/vec | 269 B/vec | 641 B/vec |
| `BFLOAT16` | 268 B/vec (25.6 MB) | 280 B/vec | 232 B/vec | 641 B/vec |
| `INT8_SYM` | 144 B/vec (13.7 MB) | 156 B/vec | 163 B/vec | 393 B/vec |
| `INT8_ASYM` | 148 B/vec (14.1 MB) | 160 B/vec | 165 B/vec | 401 B/vec |
| `FP4` | 80 B/vec (7.6 MB) | 92 B/vec | 98 B/vec | 265 B/vec |
| `INT4` | 80 B/vec (7.6 MB) | 92 B/vec | 98 B/vec | 265 B/vec |
| `INT3` | 64 B/vec (6.1 MB) | 76 B/vec | 82 B/vec | 233 B/vec |
| `TURBO_2BIT` | 48 B/vec (4.6 MB) | 60 B/vec | 66 B/vec | 201 B/vec |
| `TURBO_3BIT` | 64 B/vec (6.1 MB) | 76 B/vec | 82 B/vec | 233 B/vec |
| `TURBO_4BIT` | 80 B/vec (7.6 MB) | 92 B/vec | 98 B/vec | 265 B/vec |

Using `TURBO_2BIT` instead of `FLOAT32` reduces on-disk size by ~11× (50 MB →
4.6 MB per 100K vectors). LMDB is consistently ~2× larger than the other
backends due to its B-tree page overhead, but this is the cost of its
transactional and concurrent-read capabilities.

### TurboQuant performance overhead

TurboQuant dtypes apply a random orthogonal rotation before scalar
quantization, which spreads quantization error evenly but adds computational
cost. The overhead is most visible during queries, where dequantization sits
in the hot loop.

#### Insert overhead (add_many, 1M vectors)

TurboQuant inserts are only marginally slower than standard quantization at the
same bit-width — the rotation matrix multiply is amortized across the batch:

| Dtype | Avg vec/s | vs standard equivalent |
|---|---|---|
| `TURBO_2BIT` | ~66K | 0.97× vs `INT3` |
| `TURBO_3BIT` | ~64K | 0.90× vs `INT4` |
| `TURBO_4BIT` | ~63K | 0.89× vs `INT8_SYM` |

#### Query overhead (query_prefix, 1M vectors)

Queries are significantly slower because each scored vector requires
unpacking, codebook lookup, and inverse rotation:

| Dtype | Avg vec/s | vs `FLOAT32` | vs standard equivalent |
|---|---|---|---|
| `FLOAT32` | 7.74M | 1.0× | — |
| `INT8_SYM` | 5.35M | 0.69× | — |
| `INT4` | 4.10M | 0.53× | — |
| `INT3` | 2.86M | 0.37× | — |
| `TURBO_2BIT` | 2.32M | 0.30× | 0.81× vs `INT3` |
| `TURBO_3BIT` | 1.29M | 0.17× | 0.31× vs `INT4` |
| `TURBO_4BIT` | 933K | 0.12× | 0.17× vs `INT8_SYM` |

`TURBO_2BIT` queries at ~0.30× FLOAT32 speed — reasonable given the 11×
storage reduction. `TURBO_3BIT` and `TURBO_4BIT` are notably slower because the
sub-byte unpacking for 3- and 4-bit codes is more complex. If query speed
matters more than maximum compression, prefer `INT4` or `INT8_SYM` over the
higher-bit TurboQuant variants.

## Similarity metrics

Three built-in metrics are provided:

- `cosine_similarity` — default, higher = more similar
- `dot_product` — raw dot product
- `euclidean_distance` — negative L2 distance (higher = closer)

```python
from tqvs import create_vector_store, euclidean_distance

store = create_vector_store("./store", dim=128, metric=euclidean_distance)
```

## GPU acceleration with PyTorch

Install the `torch` extra to unlock CUDA-accelerated similarity scoring:

```bash
pip install tqvs[torch]
```

When `device` is set, `query()` and `score()` automatically dispatch to
PyTorch equivalents of the three built-in metrics. Vectors are transferred to
the target device, scored using `torch` tensor ops, and results are returned as
NumPy arrays.

```python
store = create_vector_store("./store", dim=768, device="cuda")

# query() and score() now run on GPU
results = store.query(q, k=10)
```

This is most beneficial when:

- **You use `cosine_similarity`** (the default) — the normalization step adds
  enough compute to offset the CPU→GPU→CPU data transfer overhead. For
  `dot_product` and `euclidean_distance`, NumPy is already fast enough that
  the transfer cost makes CUDA slower in practice.
- **Vector count is large** (10k+) — the cost of host-to-device transfer is
  amortized over a larger scoring pass.
- **Dimensionality is high** (768, 1024, 1536) — GPU parallelism shines on
  wide matrix operations.

#### CUDA speedup (cosine_similarity, RTX 5090)

| Vectors | dim=128 | dim=768 | dim=1536 |
|---|---|---|---|
| 1,000 | 0.2× (slower) | 1.9× | 2.5× |
| 10,000 | 1.7× | 4.0× | 4.4× |
| 100,000 | 2.7× | 4.5× | 4.5× |
| 1,000,000 | 3.2× | **4.9×** | OOM |

For `dot_product` and `euclidean_distance`, CUDA is 0.1–0.9× (slower than
NumPy) across all tested configurations because the operations are dominated by
a single matrix-vector product that NumPy (MKL/OpenBLAS) handles efficiently
without data transfer overhead.

**Recommendation:** Use `device="cuda"` with the default `cosine_similarity`
metric when you have 10k+ vectors at dim ≥ 768. For other metrics or smaller
stores, leave `device` unset.

For small stores (under ~10k vectors) or low dimensions, NumPy on CPU is
already fast enough and the data-transfer overhead makes GPU dispatch slower.
The `device` parameter also accepts `"cpu"` to use PyTorch's CPU kernels
(useful for MKL-backed PyTorch builds), though in most cases plain NumPy is
comparable.

> **Note:** PyTorch acceleration applies only to the scoring step of
> `query()` and `score()`. Insert, delete, save, reload, and key operations
> are always NumPy-based and unaffected by the `device` setting.

## Load modes

| Mode | Behavior |
|---|---|
| `EAGER` | Load all data into memory on open (default) |
| `LAZY` | Defer loading until first access |
| `MMAP` | Memory-map the vector array from disk |

## License

See [pyproject.toml](pyproject.toml) for package metadata.