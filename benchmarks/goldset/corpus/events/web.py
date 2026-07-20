from events.registry import route, task


@route("/admin/user_created_report")
def show_user_created_report():
    """Admin page LISTING created users — not an event handler."""
    return "<table></table>"


@task("cleanup_user_created_exports")
def cleanup_user_created_exports():
    """Nightly file cleanup — not an event handler."""
    return 0
