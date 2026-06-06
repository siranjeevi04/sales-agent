"""
Groq agentic loop. Handles multi-turn tool use until the model reaches stop.
Tool calls are executed locally; results are fed back into the context.
"""
import json
from pathlib import Path
from groq import Groq
from app.config import settings
from app.memory.base import BaseMemory
from app.agents.tool_defs import TOOL_DEFINITIONS
from app.tools.search_catalog import search_catalog
from app.tools.get_user_memory import get_user_memory
from app.tools.flag_for_human import flag_for_human

client = Groq(api_key=settings.groq_api_key_1)

_PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"
_SYSTEM_PROMPT = (_PROMPTS_DIR / "agent_system.txt").read_text(encoding="utf-8").strip()


def _dispatch(tool_name: str, args: dict, user_id: str, session_id: str, memory: BaseMemory) -> str:
    if tool_name == "search_catalog":
        return search_catalog(**args)
    elif tool_name == "get_user_memory":
        return get_user_memory(user_id=user_id, memory=memory)
    elif tool_name == "flag_for_human":
        return flag_for_human(
            user_id=user_id,
            session_id=session_id,
            reason=args.get("reason", "Unspecified"),
            memory=memory,
        )
    return json.dumps({"error": f"Unknown tool: {tool_name}"})


def run_agent(
    user_id: str,
    user_message: str,
    session_id: str,
    memory: BaseMemory,
) -> tuple[str, list[str]]:
    """
    Run the agentic loop for one user turn.
    Persists all messages (user, assistant, tool results) to DB.
    Returns (final_reply_text, tools_called_list).
    """
    # Persist the incoming user message
    memory.save_message(user_id, session_id, "user", user_message)

    # Build context: system prompt + optional memory summary
    user_summary = memory.get_user_summary(user_id)
    system_content = _SYSTEM_PROMPT
    if user_summary:
        system_content += f"\n\n## User Memory\n{user_summary}"

    # Retrieve recent session history from DB
    history = memory.get_session_messages(user_id, session_id, last_n=settings.max_history_messages)

    messages = [{"role": "system", "content": system_content}] + history
    tools_called: list[str] = []

    while True:
        response = client.chat.completions.create(
            model=settings.agent_model,
            max_tokens=settings.max_tokens,
            tools=TOOL_DEFINITIONS,
            tool_choice="auto",
            messages=messages,
        )

        choice = response.choices[0]
        msg = choice.message

        # No tool calls → done
        if choice.finish_reason == "stop" or not msg.tool_calls:
            final_text = msg.content or ""
            memory.save_message(user_id, session_id, "assistant", final_text)
            return final_text.strip(), tools_called

        # Serialize tool_calls for DB storage (Groq objects → dicts)
        tc_dicts = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in msg.tool_calls
        ]

        # Persist assistant turn with tool_calls
        memory.save_message(
            user_id, session_id, "assistant",
            content=msg.content,  # may be None
            tool_calls=tc_dicts,
        )

        # Append to in-memory messages for next loop iteration
        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": tc_dicts,
        })

        # Execute tools and append results
        for tc in msg.tool_calls:
            tools_called.append(tc.function.name)
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}

            result = _dispatch(tc.function.name, args, user_id, session_id, memory)

            # Persist tool result
            memory.save_message(
                user_id, session_id, "tool",
                content=result,
                tool_call_id=tc.id,
                tool_name=tc.function.name,
            )

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })
