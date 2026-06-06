"""
Tool schemas passed to Groq API. Defined here so agent loop stays clean.
"""

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "search_catalog",
            "description": (
                "Semantic search over the SaaS product catalog. "
                "Use this to find plans, pricing, features, or add-ons. "
                "Always call this instead of guessing product details."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query (e.g. 'enterprise SSO', 'plan for 10 users', 'storage addon').",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (default 3).",
                        "default": 3,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_user_memory",
            "description": (
                "Retrieves stored memory and past facts about this user from the database. "
                "Call this at the start of a conversation to recall what the user has discussed before."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "The user's unique ID.",
                    },
                },
                "required": ["user_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "flag_for_human",
            "description": (
                "Escalate the conversation for human review. "
                "Use when: confidence is very low, user is frustrated, legal/billing questions arise, "
                "or the query is outside your knowledge."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Short explanation of why human review is needed.",
                    },
                },
                "required": ["reason"],
            },
        },
    },
]
