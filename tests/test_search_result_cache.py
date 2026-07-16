"""Tests for search_symbols result cache (Feature 5)."""

import pytest
import threading
from pathlib import Path
from typing import Optional

from tests.conftest_helpers import create_mini_index


# Import the cache functions we expect to exist (will fail until GREEN)
def _import_cache_functions():
    """Lazily import cache functions from search_symbols."""
    from jcodemunch_mcp.tools.search_symbols import (
        _result_cache_get,
        _result_cache_put,
        _result_cache,
        _result_cache_lock,
        _RESULT_CACHE_MAX,
        result_cache_invalidate_repo,
    )
    return (
        _result_cache_get,
        _result_cache_put,
        _result_cache,
        _result_cache_lock,
        _RESULT_CACHE_MAX,
        result_cache_invalidate_repo,
    )


@pytest.fixture(autouse=True)
def _clear_result_cache():
    """Clear the search result cache before/after each test."""
    try:
        from jcodemunch_mcp.tools.search_symbols import _result_cache, _result_cache_lock
        with _result_cache_lock:
            _result_cache.clear()
    except ImportError:
        pass
    yield
    try:
        from jcodemunch_mcp.tools.search_symbols import _result_cache, _result_cache_lock
        with _result_cache_lock:
            _result_cache.clear()
    except ImportError:
        pass


class TestSearchResultCacheBasics:
    """Basic get/put operations."""

    def test_cache_put_and_get(self):
        """Put a key-value into cache, get it back."""
        (
            _result_cache_get,
            _result_cache_put,
            _result_cache,
            _result_cache_lock,
            _RESULT_CACHE_MAX,
            result_cache_invalidate_repo,
        ) = _import_cache_functions()
        key = ("owner/repo", "indexed_at_ts", "query", "standard", None, None, None, 10, False, False, 0.4, 2, "relevance", 0.5, False)
        value = {"result_count": 1, "results": [{"id": "test"}], "_meta": {"timing_ms": 1.0}}
        _result_cache_put(key, value)
        got = _result_cache_get(key)
        assert got is not None
        assert got["result_count"] == 1

    def test_cache_miss_returns_none(self):
        """Get nonexistent key returns None."""
        (
            _result_cache_get,
            _result_cache_put,
            _result_cache,
            _result_cache_lock,
            _RESULT_CACHE_MAX,
            result_cache_invalidate_repo,
        ) = _import_cache_functions()
        key = ("owner/repo", "indexed_at_ts", "nonexistent_query", "standard", None, None, None, 10, False, False, 0.4, 2, "relevance", 0.5, False)
        got = _result_cache_get(key)
        assert got is None

    def test_cache_evicts_oldest_when_full(self):
        """Fill past _RESULT_CACHE_MAX, verify oldest evicted."""
        (
            _result_cache_get,
            _result_cache_put,
            _result_cache,
            _result_cache_lock,
            _RESULT_CACHE_MAX,
            result_cache_invalidate_repo,
        ) = _import_cache_functions()
        # Fill to capacity
        for i in range(_RESULT_CACHE_MAX):
            key = ("repo", f"ts{i}", f"query{i}", "standard", None, None, None, 10, False, False, 0.4, 2, "relevance", 0.5, False)
            _result_cache_put(key, {"i": i})
        # Oldest (i=0) should still be present before overflow
        oldest_key = ("repo", "ts0", "query0", "standard", None, None, None, 10, False, False, 0.4, 2, "relevance", 0.5, False)
        assert _result_cache_get(oldest_key) is not None
        # Add one more to trigger eviction
        overflow_key = ("repo", "overflow_ts", "overflow_query", "standard", None, None, None, 10, False, False, 0.4, 2, "relevance", 0.5, False)
        _result_cache_put(overflow_key, {"overflow": True})
        # Cache size should stay at max
        with _result_cache_lock:
            assert len(_result_cache) == _RESULT_CACHE_MAX

    def test_invalidate_repo_removes_matching(self):
        """Invalidate repoA entries, repoB untouched."""
        (
            _result_cache_get,
            _result_cache_put,
            _result_cache,
            _result_cache_lock,
            _RESULT_CACHE_MAX,
            result_cache_invalidate_repo,
        ) = _import_cache_functions()
        key_a = ("owner/repoA", "ts1", "query", "standard", None, None, None, 10, False, False, 0.4, 2, "relevance", 0.5, False)
        key_b = ("owner/repoB", "ts2", "query", "standard", None, None, None, 10, False, False, 0.4, 2, "relevance", 0.5, False)
        _result_cache_put(key_a, {"repo": "A"})
        _result_cache_put(key_b, {"repo": "B"})
        result_cache_invalidate_repo("owner/repoA")
        assert _result_cache_get(key_a) is None
        assert _result_cache_get(key_b) is not None


class TestSearchCacheIntegration:
    """Integration tests with search_symbols."""

    def test_search_cache_hit_returns_same(self, tmp_path: Path):
        """Two identical search_symbols calls return same results (cache hit)."""
        from jcodemunch_mcp.tools.search_symbols import search_symbols
        repo, storage_path = create_mini_index(tmp_path)
        # First call populates cache
        result1 = search_symbols(repo=repo, query="my_func", storage_path=storage_path)
        # Second call should hit cache
        result2 = search_symbols(repo=repo, query="my_func", storage_path=storage_path)
        assert result1["result_count"] == result2["result_count"]
        # Second result should have cache_hit flag
        assert result2["_meta"].get("cache_hit") is True

    def test_cache_hit_records_savings_into_meter(self, tmp_path: Path, monkeypatch):
        """A cache hit re-records the cached per-call savings into the lifetime
        meter. Without this, every repeat query (identical avoided read) is
        invisible to the persistent meter + community counter — a systematic
        undercount on repetitive workloads."""
        import jcodemunch_mcp.tools.search_symbols as ss
        repo, storage_path = create_mini_index(tmp_path)

        calls: list = []

        def _spy(tokens_saved, *a, **k):
            calls.append(tokens_saved)
            return sum(calls)

        monkeypatch.setattr(ss, "record_savings", _spy)

        r1 = ss.search_symbols(repo=repo, query="my_func", storage_path=storage_path)
        saved = r1["_meta"].get("tokens_saved", 0)
        n_after_miss = len(calls)

        r2 = ss.search_symbols(repo=repo, query="my_func", storage_path=storage_path)
        assert r2["_meta"].get("cache_hit") is True

        if saved:
            # The hit recorded exactly the cached per-call figure (not zero).
            assert len(calls) == n_after_miss + 1
            assert calls[-1] == saved
            assert r2["_meta"]["total_tokens_saved"] == sum(calls)
        else:
            # Degenerate tiny index with no savings: the hit must not
            # fabricate a recording either.
            assert len(calls) == n_after_miss

    def test_cache_skipped_for_debug(self, tmp_path: Path):
        """search_symbols(debug=True) should NOT populate cache."""
        (
            _result_cache_get,
            _result_cache_put,
            _result_cache,
            _result_cache_lock,
            _RESULT_CACHE_MAX,
            result_cache_invalidate_repo,
        ) = _import_cache_functions()
        from jcodemunch_mcp.tools.search_symbols import search_symbols
        repo, storage_path = create_mini_index(tmp_path)
        # Clear cache first
        with _result_cache_lock:
            _result_cache.clear()
        # debug=True should skip cache
        result = search_symbols(repo=repo, query="my_func", debug=True, storage_path=storage_path)
        # Cache should remain empty (debug bypasses cache)
        with _result_cache_lock:
            assert len(_result_cache) == 0

    def test_cache_skipped_for_semantic(self, tmp_path: Path):
        """search_symbols(semantic=True) should NOT populate cache."""
        (
            _result_cache_get,
            _result_cache_put,
            _result_cache,
            _result_cache_lock,
            _RESULT_CACHE_MAX,
            result_cache_invalidate_repo,
        ) = _import_cache_functions()
        from jcodemunch_mcp.tools.search_symbols import search_symbols
        repo, storage_path = create_mini_index(tmp_path)
        # Clear cache first
        with _result_cache_lock:
            _result_cache.clear()
        # semantic=True should skip cache (will return error but shouldn't cache)
        result = search_symbols(repo=repo, query="my_func", semantic=True, storage_path=storage_path)
        # Cache should remain empty (semantic bypasses cache)
        with _result_cache_lock:
            assert len(_result_cache) == 0


class TestSearchCacheThreadSafety:
    """Thread safety tests."""

    def test_thread_safety(self):
        """Concurrent put/get operations don't crash."""
        (
            _result_cache_get,
            _result_cache_put,
            _result_cache,
            _result_cache_lock,
            _RESULT_CACHE_MAX,
            result_cache_invalidate_repo,
        ) = _import_cache_functions()
        errors = []

        def writer(i: int):
            try:
                for j in range(100):
                    key = ("repo", f"ts{i}_{j}", f"query{i}_{j}", "standard", None, None, None, 10, False, False, 0.4, 2, "relevance", 0.5, False)
                    _result_cache_put(key, {"i": i, "j": j})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0