"""
flag_for_human tool — escalates a conversation for human review.
Called by the agent when confidence is low, topic is sensitive, or user is frustrated.
"""
import json
from app.memory.base import BaseMemory


def flag_for_human(user_id: str, session_id: str, reason: str, memory: BaseMemory) -> str:
    """
    Logs a human-review flag to the DB. Human reviewer can query GET /flags.
    """
    memory.log_flag(user_id=user_id, session_id=session_id, reason=reason)
    return json.dumps({
        "flagged": True,
        "user_id": user_id,
        "session_id": session_id,
        "reason": reason,
        "message": "Conversation flagged for human review. A team member will follow up.",
    })
