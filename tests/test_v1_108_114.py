"""v1.108.114 — route() gains mutate + stateful intent buckets.

The leakage finding: route() had no bucket for edit/execute commands or for
session/recent-change queries, so those intents fell through to search_symbols
or NO_MATCH. This adds both, on-charter: mutate is recognize-and-redirect to a
READ-ONLY prep tool (jcm never performs the edit), stateful maps to the
session/delta tools.
"""
import re

from jcodemunch_mcp import counter

# A catalog superset covering the new bucket targets + the pre-existing ones.
NAMES = {
    "search_symbols", "search_text", "get_file_outline", "get_repo_health",
    "get_call_hierarchy", "find_references", "get_blast_radius",
    "get_class_hierarchy", "get_dependency_graph", "find_dead_code",
    "assemble_task_context",
    # new bucket targets
    "get_changed_symbols", "get_session_context",
    "check_rename_safe", "check_delete_safe", "plan_refactoring", "check_edit_safe",
}

MUTATE_PREP_TOOLS = {"check_rename_safe", "check_delete_safe",
                     "plan_refactoring", "check_edit_safe"}
STATEFUL_TOOLS = {"get_changed_symbols", "get_session_context"}


def _top(task):
    recs = counter.classify_intent(task, NAMES)
    return recs[0]["action"] if recs else None


# --- mutate bucket --------------------------------------------------------- #

def test_mutate_commands_route_to_readonly_prep():
    cases = {
        "rename UserRepository to AccountRepository everywhere": "check_rename_safe",
        "delete the unused helper functions": "check_delete_safe",
        "remove the dead config loader": "check_delete_safe",
        "refactor the auth module": "plan_refactoring",
        "extract this into a separate method": "plan_refactoring",
        "move the config loading into its own module": "plan_refactoring",
        "inline the temporary variable": "plan_refactoring",
        "add a retry to the http client": "check_edit_safe",
        "fix the bug in parse_config": "check_edit_safe",
        "implement pagination for the results endpoint": "check_edit_safe",
        "update the docstring on createSession": "check_edit_safe",
        "change the timeout to 60 seconds": "check_edit_safe",
        "write a function that validates emails": "check_edit_safe",
        "generate a migration for the new column": "check_edit_safe",
        "apply the fix and run the tests": "check_edit_safe",
    }
    for task, expected in cases.items():
        assert _top(task) == expected, f"{task!r} -> {_top(task)} (want {expected})"


def test_mutate_recommendations_are_readonly_charter_safe():
    """The mutate bucket must only ever recommend read-only prep tools —
    never a state-changing or exec/write action. This is the charter guard."""
    for tool in MUTATE_PREP_TOOLS:
        assert not counter.is_state_changing(tool), f"{tool} is state-changing"
        assert counter.forbidden_reason(tool) is None, f"{tool} trips the exec/write tripwire"


def test_mutate_why_declares_readonly():
    recs = counter.classify_intent("rename Foo to Bar", NAMES)
    assert recs and "read-only" in recs[0]["why"].lower()


def test_edit_question_is_not_captured_as_mutate():
    """Leading-verb anchor: a QUESTION about an edit must fall through to the
    impact rules, not the mutate bucket (which is for imperative commands)."""
    assert _top("what would be affected if I rename UserRepository") not in MUTATE_PREP_TOOLS
    assert _top("what breaks if I change the signature of createSession") not in MUTATE_PREP_TOOLS


# --- stateful bucket ------------------------------------------------------- #

def test_stateful_change_queries_route_to_delta_tool():
    for task in [
        "what did I just change",
        "show me my uncommitted edits",
        "what symbols changed since the last commit",
        "what's different from main",
        "which functions did my last edit affect",
        "list files changed today",
        "what's in my working tree that's new",
        "show recently modified symbols",
        "the classes I renamed just now",
        "what edits are pending",
    ]:
        assert _top(task) in STATEFUL_TOOLS, f"{task!r} -> {_top(task)}"


def test_stateful_session_queries_route_to_session_tool():
    for task in [
        "what have we touched in this session",
        "pick up where we left off",
        "recap the changes from this branch",
        "what did we work on earlier",
        "the file I was editing a minute ago",
    ]:
        assert _top(task) in STATEFUL_TOOLS, f"{task!r} -> {_top(task)}"


def test_stateful_does_not_steal_semantic_session_word():
    """'where does session state get persisted' is a semantic code query, not a
    stateful session recap — the bare word 'session' must not trigger the bucket."""
    assert _top("where does session state get persisted") not in STATEFUL_TOOLS


# --- no regression to existing routes -------------------------------------- #

def test_existing_routes_unchanged():
    assert _top("who calls this function") == "get_call_hierarchy"
    assert _top("find dead code in the project") == "find_dead_code"
    assert _top("search for the string TODO in the code") == "search_text"
    assert _top("find the UserRepository class") == "search_symbols"


def test_mutate_prep_not_auto_executed():
    """Mutate prep tools have no _QUERY_ARG entry -> route recommends, never
    auto-executes a mutation flow from a free-form task."""
    for tool in MUTATE_PREP_TOOLS:
        assert counter.shape_execute_args(tool, "some/repo", "rename Foo to Bar") is None
