from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    groq_api_key_1: str  # agent
    groq_api_key_2: str  # eval + summarization
    agent_model: str
    eval_model: str
    db_path: str = "/data/sales_agent.db"
    max_history_messages: int = 20    # messages per session injected into context
    summarize_after_messages: int = 40  # trigger summarization after N total messages
    max_tokens: int = 2048
    self_eval_enabled: bool = True
    confidence_flag_threshold: float = 0.70  # flag if confidence below this

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8-sig"  # strips BOM if VSCode saves with BOM
        extra = "ignore"


settings = Settings()

# Local dev fallback when Railway /data volume not mounted
if not Path("/data").exists():
    settings.db_path = "./sales_agent.db"
