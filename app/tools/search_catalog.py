"""
Semantic catalog search using sentence-transformers/all-MiniLM-L12-v2.
Model is loaded once at module import and cached in memory.
"""
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Load catalog
_CATALOG_PATH = Path(__file__).parent.parent.parent / "catalog.json"
with open(_CATALOG_PATH, encoding="utf-8") as f:
    CATALOG: dict = json.load(f)

# Flatten all items (plans + addons) into a searchable list
_ALL_ITEMS: list[dict] = CATALOG.get("plans", []) + CATALOG.get("addons", [])

# Build text representations for embedding
def _item_to_text(item: dict) -> str:
    features = ", ".join(item.get("features", []))
    return f"{item['name']} plan - {item['price']} - {features}"

_ITEM_TEXTS: list[str] = [_item_to_text(i) for i in _ALL_ITEMS]

# Load model from HuggingFace local cache (already downloaded)
_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")

# Pre-encode all catalog items (at startup, once)
_CATALOG_EMBEDDINGS: np.ndarray = _MODEL.encode(_ITEM_TEXTS, normalize_embeddings=True)


def search_catalog(query: str, top_k: int = 3) -> str:
    """
    Semantic search over the product catalog.
    Returns top_k results ranked by cosine similarity to the query.
    """
    query_emb = _MODEL.encode([query], normalize_embeddings=True)
    scores = (_CATALOG_EMBEDDINGS @ query_emb.T).flatten()

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        item = _ALL_ITEMS[idx]
        results.append({
            "id": item["id"],
            "name": item["name"],
            "price": item["price"],
            "features": item["features"],
            "relevance_score": round(float(scores[idx]), 4),
        })

    return json.dumps({"results": results, "query": query}, indent=2)
