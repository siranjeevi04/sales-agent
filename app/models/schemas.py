from pydantic import BaseModel
from typing import Optional, Any
from datetime import datetime


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class EvalBlock(BaseModel):
    groundedness: float
    relevance: float
    confidence: float
    flagged: bool
    reasoning: str


class ChatResponse(BaseModel):
    session_id: str
    response: str
    eval: EvalBlock
    tools_called: list[str]


class MessageOut(BaseModel):
    role: str
    content: Optional[str]
    tool_calls: Optional[Any] = None
    tool_call_id: Optional[str] = None
    tool_name: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class HistoryResponse(BaseModel):
    user_id: str
    sessions: list[str]
    messages: list[MessageOut]


class EvalOut(BaseModel):
    groundedness: float
    relevance: float
    confidence: float
    flagged: bool
    reasoning: Optional[str]
    session_id: str
    created_at: datetime

    class Config:
        from_attributes = True


class EvalAggregateResponse(BaseModel):
    user_id: str
    total_responses: int
    avg_groundedness: float
    avg_relevance: float
    avg_confidence: float
    flagged_count: int
    high_confidence_pct: float  # % with confidence >= 0.85
    records: list[EvalOut]


class FlaggedLogOut(BaseModel):
    id: int
    user_id: str
    session_id: str
    reason: str
    reviewed: bool
    created_at: datetime

    class Config:
        from_attributes = True
