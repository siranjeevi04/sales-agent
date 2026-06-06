"""
Self-evaluation module. Uses a fast Groq model (llama-3.1-8b-instant) to score
every assistant response on groundedness, relevance, and confidence.
"""
import json
import re
from pathlib import Path
from groq import Groq
from app.config import settings

client = Groq(api_key=settings.groq_api_key_2)

_PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"
_EVAL_SYSTEM = (_PROMPTS_DIR / "eval_system.txt").read_text(encoding="utf-8").strip()

_CATALOG_PATH = Path(__file__).parent.parent.parent / "catalog.json"
_CATALOG_TEXT = _CATALOG_PATH.read_text(encoding="utf-8")


def evaluate_response(
    user_message: str,
    assistant_response: str,
    tools_called: list[str],
) -> dict:
    if not settings.self_eval_enabled:
        return _fallback_eval()

    try:
        resp = client.chat.completions.create(
            model=settings.eval_model,
            max_tokens=512,
            reasoning_effort=settings.reasoning_effort,
            messages=[
                {"role": "system", "content": _EVAL_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"PRODUCT CATALOG (ground truth):\n{_CATALOG_TEXT}\n\n"
                        f"User asked: {user_message}\n\n"
                        f"Tools used: {', '.join(tools_called) if tools_called else 'none'}\n\n"
                        f"Assistant replied: {assistant_response}"
                    ),
                },
            ],
        )

        text = resp.choices[0].message.content.strip()

        # Strip markdown code fences if present
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)

        scores = json.loads(text)

        # Clamp to [0, 1]
        for key in ("groundedness", "relevance", "confidence"):
            scores[key] = max(0.0, min(1.0, float(scores.get(key, 0.5))))

        # Auto-flag if any score below threshold
        if any(scores[k] < settings.confidence_flag_threshold for k in ("groundedness", "relevance", "confidence")):
            scores["flagged"] = True

        scores.setdefault("reasoning", "")
        scores.setdefault("flagged", False)

        return scores

    except Exception as e:
        print(f"[eval] error: {e}")
        return _fallback_eval()


def _fallback_eval() -> dict:
    return {
        "groundedness": 0.5,
        "relevance": 0.5,
        "confidence": 0.5,
        "flagged": False,
        "reasoning": "Eval unavailable.",
    }
