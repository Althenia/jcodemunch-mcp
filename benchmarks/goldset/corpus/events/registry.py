"""Tiny event bus used by the handler scenarios."""

_HANDLERS = {}


def on(event_name):
    """Register a handler for a named event."""
    def _wrap(fn):
        _HANDLERS.setdefault(event_name, []).append(fn)
        return fn
    return _wrap


def route(path):
    """Register a web route (unrelated to the event bus)."""
    def _wrap(fn):
        return fn
    return _wrap


def task(name):
    """Register a background task."""
    def _wrap(fn):
        return fn
    return _wrap
