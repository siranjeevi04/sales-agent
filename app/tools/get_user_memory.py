"""
get_user_memory tool — retrieves structured memory for a user from the DB.
The agent calls this tool to recall past context about the user.
"""
import json
from app.memory.base import BaseMemory


def get_user_memory(user_id: str, memory: BaseMemory) -> str:
    """
    Returns the user's stored memory summary and message stats.
    Real DB query — not hallucinated context.
    """
    user = memory.get_or_create_user(user_id)
    message_count = memory.get_message_count(user_id)
    eval_records = memory.get_evals_for_user(user_id)

    avg_confidence = None
    if eval_records:
        avg_confidence = round(
            sum(e.confidence for e in eval_records) / len(eval_records), 2
        )

    return json.dumps({
        "user_id": user_id,
        "memory_summary": user.memory_summary or "No summary yet — this may be the user's first session.",
        "total_messages_stored": message_count,
        "total_eval_records": len(eval_records),
        "avg_confidence_score": avg_confidence,
        "member_since": user.created_at.isoformat() if user.created_at else None,
    }, indent=2)
