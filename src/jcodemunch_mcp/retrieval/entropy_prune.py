"""Structural, model-free context compression for token-budgeted assembly.

Renovates entropy/relevance-guided context selection into jCodeMunch's read-only,
local idiom. Given an oversized symbol body and a per-item token cap, this ranks
lines by a structural information signal (normalized Shannon entropy of the
line's tokens, weighted by length), keeps the highest-signal lines, and ALWAYS
keeps "keystone" lines — control-flow, returns/raises, signatures, and
decision/constraint/negation cues whose removal could silently flip meaning.
Everything else is elided, replaced by an honest ``… N low-signal line(s)
elided …`` marker so the caller can never mistake a pruned view for the full body.

No model, no network, no state: the signal is pure structure (entropy of the
token distribution), so this runs headless and deterministic. The result is a
labeled VIEW, never an edit — consistent with the read-only charter. Callers opt
in explicitly (``get_ranked_context(compress=True)``); default paths never invoke
it, so their output is byte-identical.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass

# Lines whose removal could silently change behavior — never elided. Covers
# control flow, exits, boolean/comparison operators, and natural-language
# constraint cues that carry the "what must hold" of a block.
_KEYSTONE_RE = re.compile(
    r"\b(return|raise|yield|assert|if|elif|else|for|while|try|except|finally|"
    r"with|def|class|async|await|break|continue|throw|match|case|goto|guard)\b"
    r"|=>|->|:=|==|!=|<=|>=|&&|\|\|"
    r"|\b(?:must|only if|unless|never|always|require|ensure|shall)\b",
    re.IGNORECASE,
)
# Definition/signature lines — the contract of the symbol, always kept.
_SIGNATURE_RE = re.compile(
    r"^\s*(?:@|def |class |func |function |fn |public |private |protected |"
    r"static |export |const |var |let |type |interface |struct |impl )"
)
# A decorator/annotation line is a keystone too (already covered by ``@`` above).

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\S")


@dataclass
class PruneResult:
    """Outcome of a keystone-protected prune. ``text`` is a labeled view."""

    text: str
    total_lines: int
    kept_lines: int
    elided_lines: int
    is_pruned: bool


def _token_entropy(text: str) -> float:
    """Normalized Shannon entropy (0..1) of the line's token distribution."""
    toks = _TOKEN_RE.findall(text)
    n = len(toks)
    if n <= 1:
        return 0.0
    counts = Counter(toks)
    ent = -sum((c / n) * math.log2(c / n) for c in counts.values())
    return ent / math.log2(n)


def line_signal(line: str) -> float:
    """Per-line information signal: token entropy weighted by content length.

    Dense, varied lines (real logic) outrank repetitive or near-empty ones. A
    long line of distinct tokens scores highest; boilerplate and padding lowest.
    """
    stripped = line.strip()
    if not stripped:
        return 0.0
    return _token_entropy(line) * math.log2(len(stripped) + 2)


def is_keystone(line: str) -> bool:
    """True if the line must be preserved regardless of its information signal."""
    stripped = line.strip()
    if not stripped:
        return False
    if _SIGNATURE_RE.match(line):
        return True
    return bool(_KEYSTONE_RE.search(stripped))


def _emit(lines: list[str], keep: list[bool]) -> str:
    """Reconstruct the kept lines, collapsing dropped runs into elision markers."""
    out: list[str] = []
    run = 0
    for i, ln in enumerate(lines):
        if keep[i]:
            if run:
                out.append(f"    # … {run} low-signal line(s) elided …")
                run = 0
            out.append(ln)
        else:
            run += 1
    if run:
        out.append(f"    # … {run} low-signal line(s) elided …")
    return "\n".join(out)


def prune_source(source: str, max_tokens: int, count_tokens) -> PruneResult:
    """Keystone-protected prune of ``source`` toward ``max_tokens``.

    Keeps every keystone line plus the highest-signal remaining lines that fit
    the budget. ``max_tokens`` is a SOFT per-item cap: keystone protection wins
    over the budget (correctness over compression), and elision markers add a
    few tokens, so the returned view may modestly exceed the cap — the caller's
    outer greedy packer still enforces the hard total. Returns the full source
    unchanged (``is_pruned=False``) when it already fits.

    ``count_tokens`` is the same token estimator the caller packs with, so the
    prune and the pack agree on cost. Cost is measured once per line (O(L)).
    """
    lines = source.splitlines()
    if not lines:
        return PruneResult(source, 0, 0, 0, False)
    if count_tokens(source) <= max_tokens:
        return PruneResult(source, len(lines), len(lines), 0, False)

    line_cost = [max(1, count_tokens(ln)) for ln in lines]
    keep = [False] * len(lines)
    used = 0
    optional: list[tuple[float, int, int]] = []  # (-signal, index, index) for stable sort
    for i, ln in enumerate(lines):
        if is_keystone(ln):
            keep[i] = True
            used += line_cost[i]
        else:
            # Blank lines are near-free structure; rank them lowest so real
            # content is admitted first, but let them fill leftover budget.
            optional.append((-line_signal(ln), i, i))

    optional.sort()  # highest signal first (negated), ties by original order
    for _, i, _ in optional:
        if used + line_cost[i] > max_tokens:
            continue
        keep[i] = True
        used += line_cost[i]

    kept = sum(keep)
    elided = len(lines) - kept
    if elided == 0:
        return PruneResult(source, len(lines), kept, 0, False)
    return PruneResult(_emit(lines, keep), len(lines), kept, elided, True)
