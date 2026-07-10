"""Unified retrieval verdict — one honesty contract across the search tools.

An empty or weak retrieval result is positive, token-saving evidence: grounded
symbolic retrieval can prove "this is not here" where nearest-neighbour search
always returns its closest something. ``build_verdict`` centralises the logic
that ``search_symbols`` and ``get_ranked_context`` previously duplicated, and
extends it to ``search_text``.

The result carries two things:

* ``verdict`` — the unified ``_meta.verdict`` dict with a complete taxonomy
  (``ok`` / ``low_confidence`` / ``absent`` / ``degraded``), the scan counts that
  back an absence claim, per-channel status, and near-miss suggestions.
* ``negative_evidence`` — the legacy dict (or ``None``) with the same trigger and
  shape as before, so existing consumers and the injected agent policy keep
  working unchanged.
"""

from __future__ import annotations

from typing import Optional, Sequence

# Emitted as verdict["state"].
STATE_OK = "ok"
STATE_LOW_CONFIDENCE = "low_confidence"
STATE_ABSENT = "absent"
STATE_DEGRADED = "degraded"

_NOTES = {
    STATE_OK: "Confident matches returned.",
    STATE_LOW_CONFIDENCE: (
        "Matches are below the confidence threshold; verify before relying on them."
    ),
    STATE_ABSENT: (
        "No match found after scanning the index. Treat this as strong evidence the "
        "target is not present; do not reformulate the same query expecting a hit."
    ),
    STATE_DEGRADED: (
        "A requested retrieval channel was unavailable or the scan was cut short. "
        "Results are partial and absence is NOT proven."
    ),
}


def _semantic_provider_available() -> bool:
    """Return True when an embedding provider is actually configured.

    Reuses ``embed_repo``'s live detection so we do not drift from the encoder the
    semantic path would really use. Called only when semantic was requested.
    """
    try:
        from ..tools.embed_repo import _detect_provider

        detected = _detect_provider()
        if isinstance(detected, tuple):
            return bool(detected and detected[0])
        return bool(detected)
    except Exception:
        return False


def _did_you_mean(
    source_files: Optional[Sequence[str]],
    query_terms: Optional[Sequence[str]],
    cap: int = 5,
) -> list:
    """Files whose basename contains a query term (near-miss candidates)."""
    if not source_files or not query_terms:
        return []
    out: list = []
    seen: set = set()
    for f in source_files:
        base = f.lower().replace("\\", "/").rsplit("/", 1)[-1]
        if any(t in base for t in query_terms):
            if f not in seen:
                seen.add(f)
                out.append(f)
                if len(out) >= cap:
                    break
    return out


def build_verdict(
    *,
    result_count: int,
    scanned_symbols: int = 0,
    scanned_files: int = 0,
    best_score: Optional[float] = None,
    threshold: Optional[float] = None,
    query_terms: Optional[Sequence[str]] = None,
    source_files: Optional[Sequence[str]] = None,
    semantic_requested: bool = False,
    index_stale: bool = False,
    timed_out: bool = False,
) -> dict:
    """Compute the unified verdict plus the legacy negative_evidence dict.

    Returns ``{"verdict": <_meta.verdict>, "negative_evidence": <dict|None>}``.

    Backward compatibility: ``negative_evidence`` fires on exactly the historical
    trigger (empty result, or best score below threshold) with the historical keys
    and verdict names, so existing tests and the agent policy are unaffected. The
    new ``verdict`` is purely additive.
    """
    terms = [t for t in (query_terms or []) if t]
    did_you_mean = _did_you_mean(source_files, terms)

    semantic_available = _semantic_provider_available() if semantic_requested else True
    below_threshold = (
        threshold is not None and best_score is not None and best_score < threshold
    )

    # --- unified state (degraded takes precedence: partial scans can't prove absence) ---
    if timed_out:
        state = STATE_DEGRADED
    elif semantic_requested and not semantic_available:
        state = STATE_DEGRADED
    elif result_count == 0:
        state = STATE_ABSENT
    elif below_threshold:
        state = STATE_LOW_CONFIDENCE
    else:
        state = STATE_OK

    if semantic_requested and not semantic_available:
        semantic_channel = "unavailable"
    elif semantic_requested:
        semantic_channel = "ok"
    else:
        semantic_channel = "off"

    verdict = {
        "state": state,
        "scanned": {"symbols": int(scanned_symbols), "files": int(scanned_files)},
        "best_score": round(best_score, 3) if best_score is not None else None,
        "channels": {
            "lexical": "ok",
            "semantic": semantic_channel,
            "index": "stale" if index_stale else "fresh",
        },
        "note": _NOTES[state],
    }
    if did_you_mean:
        verdict["did_you_mean"] = did_you_mean

    # --- legacy negative_evidence: unchanged trigger + shape ---
    negative_evidence = None
    if result_count == 0 or below_threshold:
        negative_evidence = {
            "verdict": (
                "no_implementation_found" if result_count == 0 else "low_confidence_matches"
            ),
            "scanned_symbols": int(scanned_symbols),
            "scanned_files": int(scanned_files),
            "best_match_score": round(best_score, 3) if best_score else 0.0,
        }
        if did_you_mean:
            negative_evidence["related_existing"] = did_you_mean

    return {"verdict": verdict, "negative_evidence": negative_evidence}
