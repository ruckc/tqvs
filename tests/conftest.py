"""Shared fixtures and performance-report hook for the test suite."""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Performance results collector
# ---------------------------------------------------------------------------

PERF_RESULTS: list[dict] = []


def record_perf(
    backend: str,
    dtype: str,
    n: int,
    operation: str,
    elapsed: float,
) -> None:
    """Append a single timing row to the global results list."""
    PERF_RESULTS.append(
        {
            "backend": backend,
            "dtype": dtype,
            "n": n,
            "operation": operation,
            "elapsed": elapsed,
        }
    )


# ---------------------------------------------------------------------------
# Marker registration
# ---------------------------------------------------------------------------


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "perf: performance benchmark tests (deselected by default, run with -m perf)",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Auto-deselect ``@pytest.mark.perf`` tests unless ``-m perf`` is given."""
    marker_expr = config.getoption("-m", default="")
    if "perf" in str(marker_expr):
        return  # user explicitly requested perf tests
    perf_marker = pytest.mark.skip(reason="perf tests deselected (run with -m perf)")
    for item in items:
        if item.get_closest_marker("perf"):
            item.add_marker(perf_marker)


# ---------------------------------------------------------------------------
# Session-finish: print the performance report
# ---------------------------------------------------------------------------


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if not PERF_RESULTS:
        return

    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is None:
        return

    _write = reporter.write_line

    # Group by operation
    operations = sorted({r["operation"] for r in PERF_RESULTS})

    _write("")
    _write("=" * 90)
    _write("  PERFORMANCE REPORT")
    _write("=" * 90)

    for op in operations:
        rows = [r for r in PERF_RESULTS if r["operation"] == op]
        rows.sort(key=lambda r: (r["backend"], r["dtype"], r["n"]))

        _write("")
        _write(f"  {op}")
        _write("-" * 90)
        _write(
            f"  {'Backend':<12} {'Dtype':<14} {'N':>10}  {'Time (s)':>12}  {'vec/s':>14}"
        )
        _write("-" * 90)

        for r in rows:
            vps = r["n"] / r["elapsed"] if r["elapsed"] > 0 else float("inf")
            _write(
                f"  {r['backend']:<12} {r['dtype']:<14} {r['n']:>10,}  "
                f"{r['elapsed']:>12.4f}  {vps:>14,.0f}"
            )

    _write("")
    _write("=" * 90)
