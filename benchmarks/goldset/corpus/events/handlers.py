from events.registry import on


@on("user_created")
def send_welcome_email(payload):
    return {"sent": payload["email"]}


@on("user_created")
def provision_workspace(payload):
    return {"workspace": payload["user_id"]}


@on("user_created")
def audit_log_user(payload):
    return {"audited": payload["user_id"]}
