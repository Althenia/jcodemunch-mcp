"""Native .mjs / .cjs indexing as JavaScript (#365, reported by @oderwat).

Before this, `.mjs`/`.cjs` files resolved to no language and were dropped at
discovery as `wrong_extension` (the import-graph/dead-code/hook layers already
recognised them; only the core extension->language map didn't), so agents had to
fall back to raw scans to read them.
"""

from jcodemunch_mcp.parser.languages import (
    LANGUAGE_EXTENSIONS,
    get_language_for_path,
)
from jcodemunch_mcp.parser.extractor import parse_file


def test_mjs_cjs_registered_as_javascript():
    assert LANGUAGE_EXTENSIONS[".mjs"] == "javascript"
    assert LANGUAGE_EXTENSIONS[".cjs"] == "javascript"


def test_get_language_for_path_mjs_cjs():
    assert get_language_for_path("src/app.mjs") == "javascript"
    assert get_language_for_path("lib/legacy.cjs") == "javascript"
    # ESM/CJS variants don't disturb the plain .js / .jsx mappings
    assert get_language_for_path("a.js") == "javascript"
    assert get_language_for_path("a.jsx") == "javascript"


def test_mjs_cjs_yield_symbols():
    src = (
        "export function greet(name) { return name; }\n"
        "export const VERSION = 1;\n"
        "function helper(a, b) { return a + b; }\n"
    )
    for filename in ("hooks.mjs", "config.cjs"):
        lang = get_language_for_path(filename)
        symbols = parse_file(src, filename, lang)
        names = {s.name for s in symbols}
        assert {"greet", "VERSION", "helper"} <= names, filename
