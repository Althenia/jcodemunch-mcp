"""v1.108.103 — audit WS-6 parser correctness (V7 + V8).

V7: nested / closure functions were misclassified `kind='method'` because the
generic walker promoted a child function to a method whenever it had ANY parent
symbol, not only when that parent was a type/class container. The dead
`container_node_types` spec field is now the promotion gate. Qualified names
also chain through the full parent path so same-named symbols nested under
different scopes stay distinct.

V8: a syntax error anywhere in a symbol's subtree dropped the whole symbol via a
blanket `node.has_error` bail. For a class with one broken method that erased
the entire class AND declassed the surviving methods to top-level functions.
Extraction now keys off a cleanly-extractable name, so a body-level error no
longer deletes the cleanly-defined symbols around it.
"""

from __future__ import annotations

from jcodemunch_mcp.parser import parse_file


def _by_name(code, lang="python", fn="t.py"):
    return {s.qualified_name: s for s in parse_file(code, fn, lang)}


# ---------------------------------------------------------------- V7

def test_v7_nested_function_is_function_not_method():
    syms = _by_name(
        "def outer():\n"
        "    def inner():\n"
        "        return 1\n"
        "    return inner\n"
    )
    assert syms["outer"].kind == "function"
    # nested def keeps kind='function' and chains under its lexical parent
    assert "outer.inner" in syms
    assert syms["outer.inner"].kind == "function"


def test_v7_method_in_class_still_method():
    syms = _by_name(
        "class C:\n"
        "    def m(self):\n"
        "        return 1\n"
    )
    assert syms["C"].kind == "class"
    assert syms["C.m"].kind == "method"


def test_v7_function_nested_in_method_is_function():
    syms = _by_name(
        "class C:\n"
        "    def m(self):\n"
        "        def helper():\n"
        "            return 2\n"
        "        return helper\n"
    )
    assert syms["C.m"].kind == "method"
    # a closure inside a method is a plain function, fully qualified
    assert syms["C.m.helper"].kind == "function"


def test_v7_deeply_nested_same_named_classes_are_distinct():
    syms = _by_name(
        "class A:\n"
        "    class Mid:\n"
        "        class Inner:\n"
        "            pass\n"
        "class B:\n"
        "    class Mid:\n"
        "        class Inner:\n"
        "            pass\n"
    )
    # full-chain qualified names disambiguate the two Inner classes
    assert "A.Mid.Inner" in syms
    assert "B.Mid.Inner" in syms
    assert syms["A.Mid.Inner"].id != syms["B.Mid.Inner"].id


def test_v7_rust_trait_method_promoted():
    syms = _by_name(
        "trait T {\n"
        "    fn m(&self) -> i32 { 1 }\n"
        "}\n",
        lang="rust", fn="t.rs",
    )
    # trait_item is a container_node_type → its (default-bodied) fn is a method
    assert syms["T.m"].kind == "method"


# ---------------------------------------------------------------- V8

def test_v8_class_with_broken_method_keeps_class_and_siblings():
    syms = _by_name(
        "class C:\n"
        "    def a(:\n"            # syntax error in one method
        "        return 1\n"
        "    def b(self):\n"
        "        return 2\n"
        "    def c(self):\n"
        "        return 3\n"
    )
    # the class survives and its clean methods stay methods, not top-level fns
    assert syms["C"].kind == "class"
    assert syms["C.b"].kind == "method"
    assert syms["C.c"].kind == "method"


def test_v8_broken_middle_keeps_trailing():
    syms = _by_name(
        "def a():\n"
        "    return 1\n"
        "def b(:\n"                # broken middle
        "    return 2\n"
        "def c():\n"
        "    return 3\n"
    )
    assert "a" in syms
    assert "c" in syms  # trailing valid function is not erased


def test_v8_clean_file_unaffected():
    # A file with no errors must extract exactly what it did before (no phantom
    # symbols from loosening the has_error guard).
    syms = _by_name(
        "def a():\n"
        "    return 1\n"
        "class C:\n"
        "    def m(self):\n"
        "        return 2\n"
    )
    assert set(syms) == {"a", "C", "C.m"}
    assert syms["a"].kind == "function"
    assert syms["C"].kind == "class"
    assert syms["C.m"].kind == "method"
