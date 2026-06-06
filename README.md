# Persistent Sales Assistant Agent

A hosted conversational API for B2B SaaS sales — with cross-session memory, real tool use, and self-evaluation on every response.

**Live URL:** `https://<your-project>.up.railway.app`

---

## Architecture

```
User Request
     │
     ▼
POST /chat/{user_id}          ← FastAPI route handler (app/api/routes.py)
     │
     ▼
ChatService.handle_chat()     ← app/services/chat.py
     │
     ├─► SQLiteMemory         ← app/memory/sqlite.py  (implements BaseMemory ABC)
     │   └─► SQLite DB        ← /data/sales_agent.db  (Railway Volume)
     │
     ▼
AgentLoop.run_agent()         ← app/agents/loop.py
     │
     ├─► [Tool: get_user_memory]    ← queries DB via BaseMemory
     ├─► [Tool: search_catalog]     ← semantic search with all-MiniLM-L12-v2
     └─► [Tool: flag_for_human]     ← logs to FlaggedLog table
     │
     ▼
LLM: Groq llama3-groq-70b (tool use)
     │
     ▼
EvalService.evaluate_response()   ← app/agents/eval.py
     │   Groq llama-3.1-8b-instant → groundedness / relevance / confidence
     │
     ▼
Response:
{
  "session_id": "uuid",
  "response": "...",
  "eval": { "groundedness": 0.92, "relevance": 0.88, "confidence": 0.91, "flagged": false, "reasoning": "..." },
  "tools_called": ["get_user_memory", "search_catalog"]
}
```

---

## Quick Start

```bash
cp .env.example .env
# Set GROQ_API_KEY in .env
pip install -r requirements.txt
uvicorn main:app --reload
```

---

## Cross-Session Memory Demo (curl)

These two calls use the **same `user_id`** but are completely separate HTTP requests — no session_id shared in call 2. The agent recalls context from call 1 via the DB.

**Call 1 — establish context:**
```bash
curl -X POST https://<your-project>.up.railway.app/chat/user-demo-001 \
  -H "Content-Type: application/json" \
  -d '{"message": "Hi, I have a team of 30 people and need SSO. What do you recommend?"}'
```

Expected: agent calls `search_catalog` + `get_user_memory`, recommends Enterprise plan.

**Call 2 — new session, same user:**
```bash
curl -X POST https://<your-project>.up.railway.app/chat/user-demo-001 \
  -H "Content-Type: application/json" \
  -d '{"message": "Does that plan include audit logs too?"}'
```

Expected: agent calls `get_user_memory`, knows user asked about SSO/30 people, confirms Enterprise includes audit logs. **No context re-sent in the request** — retrieved from DB.

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/chat/{user_id}` | Send message, get response + eval |
| GET | `/chat/{user_id}/history` | Full conversation history |
| DELETE | `/chat/{user_id}/memory` | GDPR-style memory wipe |
| GET | `/chat/{user_id}/evals` | Aggregated eval scores |
| GET | `/catalog` | Product catalog |
| GET | `/flags` | Human-review queue |
| GET | `/health` | Health check |

---

## Memory Design

**What's stored:** Every message (user, assistant, tool calls, tool results) is persisted to SQLite with `user_id` + `session_id` keys.

**How it works:**
1. Incoming message → `get_or_create_user(user_id)`
2. Session messages (last 20) injected into LLM context
3. Agent calls `get_user_memory` tool → returns DB-stored summary
4. After 40+ messages: older turns are summarized by Groq → stored in `users.memory_summary`

**Why SQLite:** Zero-dependency, no external service, fits Railway free tier. The `BaseMemory` ABC means swapping to Postgres or Mem0 requires changing one line in `services/chat.py`.

**At scale:** Replace `SQLiteMemory` with `PostgresMemory` + pgvector for semantic recall, or `Mem0Memory` for managed long-term memory. No other files change.

---

## Eval Design

Every response is scored by `llama-3.1-8b-instant` (fast, cheap) on three dimensions:

| Score | What it measures |
|-------|-----------------|
| `groundedness` | Is the answer sourced from catalog data, not hallucinated? |
| `relevance` | Does it answer what was actually asked? |
| `confidence` | Overall reliability of the response |

**Auto-flagging:** If any score < 0.70, `flagged: true` is set and logged to `flagged_logs` table. The `flag_for_human` tool also lets the agent self-escalate.

**Limitations:** LLM self-scoring has confirmation bias — the model that generated the answer tends to rate it higher. At production scale, replace with a dedicated eval pipeline (RAGAS, TruLens, or human labels) using the already-logged `eval_records` table as ground truth.

---

## Project Structure

```
app/
  api/         → route handlers only
  agents/      → agent loop, tool schemas, eval logic
  memory/      → BaseMemory ABC + SQLiteMemory implementation
  tools/       → search_catalog, get_user_memory, flag_for_human
  services/    → chat orchestration, eval aggregation
  models/      → Pydantic schemas
  db/          → SQLAlchemy models, session
catalog.json   → mock product catalog
main.py        → FastAPI entry point
Dockerfile
railway.toml
```

---

## Deployment (Railway)

1. Push to GitHub
2. New Railway project → connect repo
3. Add Volume → mount at `/data`
4. Add env var `GROQ_API_KEY`
5. Deploy — Railway auto-detects Dockerfile

The SQLite DB lives on the Railway Volume at `/data/sales_agent.db` and survives redeploys.
