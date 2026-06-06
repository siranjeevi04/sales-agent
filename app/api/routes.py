import json
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session as DBSession

from app.db.session import get_db
from app.db.models import User, Message, FlaggedLog
from app.models.schemas import (
    ChatRequest, ChatResponse, EvalBlock,
    HistoryResponse, MessageOut,
    EvalAggregateResponse, FlaggedLogOut,
)
from app.memory.sqlite import SQLiteMemory
from app.services.chat import handle_chat
from app.services.eval_service import get_user_eval_aggregate

router = APIRouter()

_CATALOG_PATH = Path(__file__).parent.parent.parent / "catalog.json"


# ---------------------------------------------------------------------------
# Core endpoints
# ---------------------------------------------------------------------------

@router.post("/chat/{user_id}", response_model=ChatResponse)
def chat(user_id: str, req: ChatRequest, db: DBSession = Depends(get_db)):
    result = handle_chat(
        user_id=user_id,
        message=req.message,
        session_id=req.session_id,
        db=db,
    )
    return ChatResponse(
        session_id=result["session_id"],
        response=result["response"],
        eval=EvalBlock(**result["eval"]),
        tools_called=result["tools_called"],
    )


@router.get("/chat/{user_id}/history", response_model=HistoryResponse)
def get_history(user_id: str, db: DBSession = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail=f"User '{user_id}' not found")

    msgs = (
        db.query(Message)
        .filter(Message.user_id == user_id)
        .order_by(Message.created_at.asc())
        .all()
    )

    sessions = list(dict.fromkeys(m.session_id for m in msgs))

    messages_out = [
        MessageOut(
            role=m.role,
            content=m.content,
            tool_calls=m.tool_calls,
            tool_call_id=m.tool_call_id,
            tool_name=m.tool_name,
            created_at=m.created_at,
        )
        for m in msgs
    ]

    return HistoryResponse(user_id=user_id, sessions=sessions, messages=messages_out)


@router.delete("/chat/{user_id}/memory", status_code=200)
def delete_memory(user_id: str, db: DBSession = Depends(get_db)):
    memory = SQLiteMemory(db)
    memory.wipe_user_memory(user_id)
    return {"message": f"Memory for user '{user_id}' wiped.", "user_id": user_id}


# ---------------------------------------------------------------------------
# Bonus endpoints
# ---------------------------------------------------------------------------

@router.get("/chat/{user_id}/evals", response_model=EvalAggregateResponse)
def get_evals(user_id: str, db: DBSession = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail=f"User '{user_id}' not found")
    return get_user_eval_aggregate(user_id=user_id, db=db)


@router.get("/flags", response_model=list[FlaggedLogOut])
def get_flags(reviewed: bool = None, db: DBSession = Depends(get_db)):
    """Human reviewer endpoint — returns flagged conversations."""
    memory = SQLiteMemory(db)
    logs = memory.get_flagged_logs(reviewed=reviewed)
    return [
        FlaggedLogOut(
            id=log.id,
            user_id=log.user_id,
            session_id=log.session_id,
            reason=log.reason,
            reviewed=log.reviewed,
            created_at=log.created_at,
        )
        for log in logs
    ]


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

@router.get("/catalog")
def get_catalog():
    with open(_CATALOG_PATH, encoding="utf-8") as f:
        return json.load(f)


@router.get("/health")
def health():
    return {"status": "ok"}
