"""Thread-level read/write lock and process-level file lock."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from filelock import FileLock


class RWLock:
    """Multiple-reader / single-writer lock using :class:`threading.Condition`."""

    def __init__(self) -> None:
        self._cond = threading.Condition(threading.Lock())
        self._readers = 0
        self._writer = False

    @contextmanager
    def read_lock(self) -> Generator[None]:
        with self._cond:
            while self._writer:
                self._cond.wait()
            self._readers += 1
        try:
            yield
        finally:
            with self._cond:
                self._readers -= 1
                if self._readers == 0:
                    self._cond.notify_all()

    @contextmanager
    def write_lock(self) -> Generator[None]:
        with self._cond:
            while self._writer or self._readers > 0:
                self._cond.wait()
            self._writer = True
        try:
            yield
        finally:
            with self._cond:
                self._writer = False
                self._cond.notify_all()


class StoreLock:
    """Combined thread + process lock for a store directory.

    * **Reads** – thread-level RWLock only (no file lock).
    * **Writes** – process-level FileLock first, then thread-level write lock.
    """

    def __init__(self, store_path: Path) -> None:
        self._rw = RWLock()
        self._file_lock = FileLock(store_path / ".tqvs.lock")

    @contextmanager
    def read_lock(self) -> Generator[None]:
        with self._rw.read_lock():
            yield

    @contextmanager
    def write_lock(self) -> Generator[None]:
        with self._file_lock:
            with self._rw.write_lock():
                yield
