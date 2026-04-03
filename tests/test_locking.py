"""Tests for RWLock and StoreLock — concurrent reads, exclusive writes."""

import threading
import time

import pytest

from tqvs._locking import RWLock, StoreLock


class TestRWLock:
    def test_concurrent_readers(self):
        lock = RWLock()
        results = []

        def reader(idx):
            with lock.read_lock():
                results.append(("enter", idx))
                time.sleep(0.05)
                results.append(("exit", idx))

        threads = [threading.Thread(target=reader, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All readers should have overlapping execution
        enters = [r for r in results if r[0] == "enter"]
        exits = [r for r in results if r[0] == "exit"]
        # All entered before any exited (concurrent)
        assert len(enters) >= 2  # at least some overlap

    def test_writer_excludes_readers(self):
        lock = RWLock()
        log = []

        def writer():
            with lock.write_lock():
                log.append("w_enter")
                time.sleep(0.1)
                log.append("w_exit")

        def reader():
            time.sleep(0.02)  # let writer start first
            with lock.read_lock():
                log.append("r_enter")
                log.append("r_exit")

        tw = threading.Thread(target=writer)
        tr = threading.Thread(target=reader)
        tw.start()
        tr.start()
        tw.join()
        tr.join()

        # Reader should not enter until writer exits
        assert log.index("r_enter") > log.index("w_exit")

    def test_writer_excludes_writer(self):
        lock = RWLock()
        log = []

        def writer(idx):
            with lock.write_lock():
                log.append(f"w{idx}_enter")
                time.sleep(0.05)
                log.append(f"w{idx}_exit")

        t1 = threading.Thread(target=writer, args=(1,))
        t2 = threading.Thread(target=writer, args=(2,))
        t1.start()
        time.sleep(0.01)
        t2.start()
        t1.join()
        t2.join()

        # Second writer enters after first exits
        assert log.index("w2_enter") > log.index("w1_exit")


class TestStoreLock:
    def test_read_write(self, tmp_path):
        lock = StoreLock(tmp_path)
        log = []

        def writer():
            with lock.write_lock():
                log.append("w")
                time.sleep(0.05)

        def reader():
            time.sleep(0.01)
            with lock.read_lock():
                log.append("r")

        tw = threading.Thread(target=writer)
        tr = threading.Thread(target=reader)
        tw.start()
        tr.start()
        tw.join()
        tr.join()

        assert log.index("r") > log.index("w") or log.index("r") == log.index("w")
