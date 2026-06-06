"""
Eval aggregation service — computes stats across all eval records for a user.
"""
from sqlalchemy.orm import Session as DBSession
from app.memory.sqlite import SQLiteMemory


def get_user_eval_aggregate(user_id: str, db: DBSession) -> dict:
    memory = SQLiteMemory(db)
    records = memory.get_evals_for_user(user_id)

    if not records:
        return {
            "user_id": user_id,
            "total_responses": 0,
            "avg_groundedness": 0.0,
            "avg_relevance": 0.0,
            "avg_confidence": 0.0,
            "flagged_count": 0,
            "high_confidence_pct": 0.0,
            "records": [],
        }

    n = len(records)
    avg_g = sum(r.groundedness for r in records) / n
    avg_r = sum(r.relevance for r in records) / n
    avg_c = sum(r.confidence for r in records) / n
    flagged = sum(1 for r in records if r.flagged)
    high_conf = sum(1 for r in records if r.confidence >= 0.85)

    return {
        "user_id": user_id,
        "total_responses": n,
        "avg_groundedness": round(avg_g, 3),
        "avg_relevance": round(avg_r, 3),
        "avg_confidence": round(avg_c, 3),
        "flagged_count": flagged,
        "high_confidence_pct": round(high_conf / n * 100, 1),
        "records": [
            {
                "groundedness": r.groundedness,
                "relevance": r.relevance,
                "confidence": r.confidence,
                "flagged": r.flagged,
                "reasoning": r.reasoning,
                "session_id": r.session_id,
                "created_at": r.created_at,
            }
            for r in records
        ],
    }
