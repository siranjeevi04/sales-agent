"""
Chat service — orchestrates agent loop, eval, memory summarization, and flagging.
One place to swap memory backend: change SQLiteMemory → PostgresMemory here.
"""
import uuid
from pathlib import Path
from groq import Groq
from sqlalchemy.orm import Session as DBSession

from app.config import settings
from app.memory.sqlite import SQLiteMemory
from app.agents.loop import run_agent
from app.agents.eval import evaluate_response

client = Groq(api_key=settings.groq_api_key_2)

_PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"
_SUMMARY_SYSTEM = (_PROMPTS_DIR / "summary_system.txt").read_text(encoding="utf-8").strip()


def handle_chat(
    user_id: str,
    message: str,
    session_id: str | None,
    db: DBSession,
) -> dict:
    # Memory backend — swap this one line to change storage
    memory = SQLiteMemory(db)

    # Ensure user exists
    memory.get_or_create_user(user_id)

    # Create or continue session
    if not session_id:
        session_id = str(uuid.uuid4())

    # Run agentic loop
    reply, tools_called = run_agent(
        user_id=user_id,
        user_message=message,
        session_id=session_id,
        memory=memory,
    )

    # Self-evaluation
    eval_data = evaluate_response(
        user_message=message,
        assistant_response=reply,
        tools_called=tools_called,
    )

    # Persist eval
    memory.save_eval(user_id, session_id, eval_data)

    # Auto-flag if eval says flagged
    if eval_data.get("flagged"):
        memory.log_flag(
            user_id=user_id,
            session_id=session_id,
            reason=f"Auto-flag: {eval_data.get('reasoning', 'low confidence')}",
        )

    # Memory summarization — compress old messages if threshold exceeded
    total_msgs = memory.get_message_count(user_id)
    if total_msgs >= settings.summarize_after_messages:
        _try_summarize(user_id, memory)

    return {
        "session_id": session_id,
        "response": reply,
        "eval": eval_data,
        "tools_called": tools_called,
    }


def _try_summarize(user_id: str, memory: SQLiteMemory) -> None:
    """
    Rolling summary: merge existing summary + new older messages → updated summary.
    Runs every time total messages crosses summarize_after_messages threshold.
    Old summary is never thrown away — it becomes input to the next summarization.
    """
    new_msgs, last_msg_id = memory.get_unsummarized_messages(user_id, exclude_last_n=settings.max_history_messages)
    if not new_msgs:
        return

    new_conversation = "\n".join(
        f"{m['role'].upper()}: {m.get('content') or '[tool call]'}"
        for m in new_msgs
        if m.get("content")
    )

    # Rolling: merge existing summary + only NEW unsummarized messages
    existing_summary = memory.get_user_summary(user_id)
    if existing_summary:
        user_content = (
            f"PREVIOUS SUMMARY:\n{existing_summary}\n\n"
            f"NEW MESSAGES TO INCORPORATE:\n{new_conversation}"
        )
    else:
        user_content = new_conversation

    try:
        resp = client.chat.completions.create(
            model=settings.eval_model,
            max_tokens=300,
            messages=[
                {"role": "system", "content": _SUMMARY_SYSTEM},
                {"role": "user", "content": user_content},
            ],
        )
        summary = resp.choices[0].message.content.strip()
        memory.update_user_summary(user_id, summary)
        memory.update_summarized_up_to(user_id, last_msg_id)  # mark checkpoint
    except Exception as e:
        print(f"[summarizer] error: {e}")
