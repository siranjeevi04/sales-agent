from datetime import datetime
from sqlalchemy import Column, String, Text, DateTime, Integer, Float, Boolean, JSON, ForeignKey
from sqlalchemy.orm import relationship
from app.db.session import Base


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True)  # user_id from path
    memory_summary = Column(Text, nullable=True)
    summarized_up_to_id = Column(Integer, nullable=True)  # last message.id included in summary
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    messages = relationship("Message", back_populates="user", order_by="Message.created_at")
    eval_records = relationship("EvalRecord", back_populates="user", order_by="EvalRecord.created_at")


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    session_id = Column(String, nullable=False, index=True)
    role = Column(String, nullable=False)       # user | assistant | tool
    content = Column(Text, nullable=True)       # None for assistant-with-tool-calls turns
    tool_calls = Column(JSON, nullable=True)    # list of Groq tool_call objects
    tool_call_id = Column(String, nullable=True)  # for role=tool
    tool_name = Column(String, nullable=True)   # for role=tool
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="messages")


class EvalRecord(Base):
    __tablename__ = "eval_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    session_id = Column(String, nullable=False)
    groundedness = Column(Float, nullable=False)
    relevance = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    flagged = Column(Boolean, default=False)
    reasoning = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="eval_records")


class FlaggedLog(Base):
    __tablename__ = "flagged_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, nullable=False, index=True)
    session_id = Column(String, nullable=False)
    reason = Column(Text, nullable=False)
    reviewed = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
