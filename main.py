from fastapi import FastAPI
from app.db.session import init_db
from app.api.routes import router

app = FastAPI(
    title="Persistent Sales Assistant Agent",
    description="Conversational sales API with cross-session memory, tool use, and self-evaluation.",
    version="1.0.0",
)


@app.on_event("startup")
def on_startup():
    init_db()


app.include_router(router)
