"""FastAPI application entrypoint (placeholder)."""

from fastapi import FastAPI

from app.config import settings

app = FastAPI(title=settings.APP_NAME)


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}
