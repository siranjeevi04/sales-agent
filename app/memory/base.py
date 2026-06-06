from abc import ABC, abstractmethod
from typing import Optional


class BaseMemory(ABC):
    """Abstract memory interface. Swap SQLiteMemory → PostgresMemory → Mem0Memory by changing one file."""

    @abstractmethod
    def get_or_create_user(self, user_id: str) -> object:
        """Return User ORM row, creating if absent."""

    @abstractmethod
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
        """Persist a single message turn."""

    @abstractmethod
    def get_session_messages(self, user_id: str, session_id: str, last_n: int = 20) -> list[dict]:
        """Return last N messages for a session in Groq API format."""

    @abstractmethod
    def get_user_summary(self, user_id: str) -> Optional[str]:
        """Return the stored memory summary for a user."""

    @abstractmethod
    def update_user_summary(self, user_id: str, summary: str) -> None:
        """Store / overwrite the user's memory summary."""

    @abstractmethod
    def get_message_count(self, user_id: str) -> int:
        """Total messages stored across all sessions for user."""

    @abstractmethod
    def get_unsummarized_messages(self, user_id: str, exclude_last_n: int = 20) -> tuple[list[dict], int]:
        """Return (messages not yet summarized excluding last N, last_message_id)."""

    @abstractmethod
    def wipe_user_memory(self, user_id: str) -> None:
        """GDPR-style: delete all messages and reset summary for user."""

    @abstractmethod
    def save_eval(self, user_id: str, session_id: str, eval_data: dict) -> None:
        """Persist an evaluation record."""

    @abstractmethod
    def get_evals_for_user(self, user_id: str) -> list:
        """Return all eval records for a user."""

    @abstractmethod
    def log_flag(self, user_id: str, session_id: str, reason: str) -> None:
        """Log a human-review flag."""

    @abstractmethod
    def get_flagged_logs(self, reviewed: Optional[bool] = None) -> list:
        """Return flagged log entries, optionally filtered by reviewed status."""
