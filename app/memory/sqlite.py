from typing import Optional
from sqlalchemy.orm import Session as DBSession
from app.memory.base import BaseMemory
from app.db.models import User, Message, EvalRecord, FlaggedLog


class SQLiteMemory(BaseMemory):
    """
    SQLite-backed memory. Swap to PostgresMemory or Mem0Memory by implementing BaseMemory
    and updating the one-line factory in services/chat.py.
    """

    def __init__(self, db: DBSession):
        self.db = db

    def get_or_create_user(self, user_id: str) -> User:
        user = self.db.query(User).filter(User.id == user_id).first()
        if not user:
            user = User(id=user_id)
            self.db.add(user)
            self.db.commit()
            self.db.refresh(user)
        return user

    def save_message(
        self,
        user_id: str,
        session_id: str,
        role: str,
        content: Optional[str],
        tool_calls: Optional[list] = None,
        tool_call_id: Optional[str] = None,
        tool_name: Optional[str] = None,
    ) -> None:
        msg = Message(
            user_id=user_id,
            session_id=session_id,
            role=role,
            content=content,
            tool_calls=tool_calls,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
        )
        self.db.add(msg)
        self.db.commit()

    def get_session_messages(self, user_id: str, session_id: str, last_n: int = 20) -> list[dict]:
        rows = (
            self.db.query(Message)
            .filter(Message.user_id == user_id, Message.session_id == session_id)
            .order_by(Message.created_at.desc())
            .limit(last_n)
            .all()
        )
        rows = list(reversed(rows))
        return [self._to_groq_message(m) for m in rows]

    def get_user_summary(self, user_id: str) -> Optional[str]:
        user = self.db.query(User).filter(User.id == user_id).first()
        return user.memory_summary if user else None

    def update_user_summary(self, user_id: str, summary: str) -> None:
        user = self.get_or_create_user(user_id)
        user.memory_summary = summary
        self.db.commit()

    def get_message_count(self, user_id: str) -> int:
        return self.db.query(Message).filter(Message.user_id == user_id).count()

    def get_unsummarized_messages(self, user_id: str, exclude_last_n: int = 20) -> tuple[list[dict], int]:
        """Return only messages not yet in the summary, excluding the most recent N."""
        user = self.get_or_create_user(user_id)
        last_id = user.summarized_up_to_id or 0

        total = self.get_message_count(user_id)
        if total <= exclude_last_n:
            return [], 0

        # All messages after last summarized id, excluding last N
        all_rows = (
            self.db.query(Message)
            .filter(Message.user_id == user_id, Message.id > last_id)
            .order_by(Message.id.asc())
            .all()
        )
        # Exclude last N
        rows_to_summarize = all_rows[:-exclude_last_n] if len(all_rows) > exclude_last_n else []
        if not rows_to_summarize:
            return [], 0

        last_msg_id = rows_to_summarize[-1].id
        return [self._to_groq_message(m) for m in rows_to_summarize], last_msg_id

    def update_summarized_up_to(self, user_id: str, message_id: int) -> None:
        user = self.get_or_create_user(user_id)
        user.summarized_up_to_id = message_id
        self.db.commit()

    def wipe_user_memory(self, user_id: str) -> None:
        self.db.query(Message).filter(Message.user_id == user_id).delete()
        self.db.query(EvalRecord).filter(EvalRecord.user_id == user_id).delete()
        self.db.query(FlaggedLog).filter(FlaggedLog.user_id == user_id).delete()
        user = self.db.query(User).filter(User.id == user_id).first()
        if user:
            user.memory_summary = None
        self.db.commit()

    def save_eval(self, user_id: str, session_id: str, eval_data: dict) -> None:
        record = EvalRecord(
            user_id=user_id,
            session_id=session_id,
            groundedness=eval_data.get("groundedness", 0.5),
            relevance=eval_data.get("relevance", 0.5),
            confidence=eval_data.get("confidence", 0.5),
            flagged=eval_data.get("flagged", False),
            reasoning=eval_data.get("reasoning"),
        )
        self.db.add(record)
        self.db.commit()

    def get_evals_for_user(self, user_id: str) -> list:
        return (
            self.db.query(EvalRecord)
            .filter(EvalRecord.user_id == user_id)
            .order_by(EvalRecord.created_at.desc())
            .all()
        )

    def log_flag(self, user_id: str, session_id: str, reason: str) -> None:
        entry = FlaggedLog(user_id=user_id, session_id=session_id, reason=reason)
        self.db.add(entry)
        self.db.commit()

    def get_flagged_logs(self, reviewed: Optional[bool] = None) -> list:
        q = self.db.query(FlaggedLog).order_by(FlaggedLog.created_at.desc())
        if reviewed is not None:
            q = q.filter(FlaggedLog.reviewed == reviewed)
        return q.all()

    # ------------------------------------------------------------------
    def _to_groq_message(self, m: Message) -> dict:
        """Convert a DB row to Groq API message dict."""
        if m.role == "user":
            return {"role": "user", "content": m.content or ""}
        elif m.role == "assistant":
            msg: dict = {"role": "assistant", "content": m.content}
            if m.tool_calls:
                msg["tool_calls"] = m.tool_calls
            return msg
        elif m.role == "tool":
            return {
                "role": "tool",
                "tool_call_id": m.tool_call_id or "",
                "content": m.content or "",
            }
        return {"role": m.role, "content": m.content or ""}
