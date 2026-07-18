"""Query-shape classification: detect source-shaped tokens in a retrieval query.

A query like "how does SqliteStore.incremental_save handle deltas?" carries a
token the author lifted straight out of the source. That token deserves an
exact-symbol lookup ahead of any ranked scoring — BM25 tokenization splits
``SqliteStore.incremental_save`` into fragments and dilutes the one identifier
the caller actually named. This module classifies query tokens into shapes:

- ``qualified``: ``Store::get`` / ``config.load_config`` — identifier segments
  joined by ``::`` or ``.`` (file-extension tails are excluded; a filename is
  not a symbol name)
- ``camel``: ``FreshnessProbe`` / ``getUser`` — an interior case change only an
  identifier has
- ``snake``: ``get_ranked_context`` — underscore-joined identifier

Plain prose words never match any shape, so a pure natural-language query
yields an empty list and downstream behavior is byte-identical.
"""

import re

# Identifier segment: letter/underscore head, word tail.
_IDENT = r"[A-Za-z_][A-Za-z0-9_]*"

_QUALIFIED_RE = re.compile(rf"^{_IDENT}(?:(?:::|\.){_IDENT})+$")
_CAMEL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9]*$")
_INTERIOR_CASE_RE = re.compile(r"[a-z0-9][A-Z]|[A-Z]{2,}[a-z]")
_SNAKE_RE = re.compile(rf"^{_IDENT}$")

# Dotted tails that mean "this token is a filename, not a qualified symbol".
_FILE_EXT_TAILS = frozenset({
    "py", "js", "jsx", "ts", "tsx", "mjs", "cjs", "mts", "cts", "md", "txt",
    "json", "jsonc", "yaml", "yml", "toml", "xml", "html", "css", "scss",
    "rs", "go", "java", "kt", "rb", "php", "c", "h", "cpp", "hpp", "cc",
    "cs", "swift", "sql", "sh", "bash", "ps1", "lua", "dart", "vue", "svelte",
    "astro", "erl", "ex", "exs", "scala", "clj", "zig", "gz", "db", "sqlite",
})

# Strip surrounding punctuation a prose sentence wraps around a pasted
# identifier: quotes, backticks, parens, trailing sentence punctuation.
_TRIM_RE = re.compile(r"^[\s'\"`(\[{<]+|[\s'\"`)\]}>.,;:!?]+$")

_MAX_TOKENS = 3  # seed at most this many distinct source-shaped tokens


def source_shaped_tokens(query: str) -> list[dict]:
    """Extract source-shaped tokens from *query*, first-seen order, deduped.

    Returns a list of ``{"token", "name", "parent", "shape"}`` dicts where
    ``name`` is the identifier to match symbol names against (the tail segment
    for qualified tokens) and ``parent`` is the qualifying segment before the
    tail (``None`` for bare identifiers). Capped at ``_MAX_TOKENS`` entries.
    """
    out: list[dict] = []
    seen: set[str] = set()
    for raw in query.split():
        tok = _TRIM_RE.sub("", raw)
        # A trailing call spelling (``foo()``, or ``foo(`` after the trim ate
        # the closing paren) is the same identifier.
        tok = re.sub(r"\(\)?$", "", tok)
        if len(tok) < 3 or tok in seen:
            continue
        shape = _classify(tok)
        if shape is None:
            continue
        seen.add(tok)
        if shape == "qualified":
            parts = re.split(r"::|\.", tok)
            entry = {"token": tok, "name": parts[-1], "parent": parts[-2], "shape": shape}
        else:
            entry = {"token": tok, "name": tok, "parent": None, "shape": shape}
        out.append(entry)
        if len(out) >= _MAX_TOKENS:
            break
    return out


def _classify(tok: str) -> str | None:
    if _QUALIFIED_RE.match(tok):
        if tok.rsplit(".", 1)[-1].lower() in _FILE_EXT_TAILS and "::" not in tok:
            return None  # filename, not a qualified symbol
        return "qualified"
    if "_" in tok and _SNAKE_RE.match(tok) and not tok.startswith("__"):
        return "snake"
    if tok.startswith("__") and tok.endswith("__") and _SNAKE_RE.match(tok):
        return "snake"  # dunder is still an exact identifier
    if _CAMEL_RE.match(tok) and _INTERIOR_CASE_RE.search(tok):
        return "camel"
    return None
