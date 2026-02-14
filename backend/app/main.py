"""FastAPI application entrypoint (placeholder)."""

from fastapi import Depends, FastAPI, Query
from sqlalchemy.orm import Session

from app.api.schemas import OHLCCreate, OHLCRead
from app.config import settings
from app.database import Base, engine, get_db
from app.models.ohlc import OHLC1m
from app.services.ohlc_service import get_ohlc_data

app = FastAPI(title=settings.APP_NAME)


@app.on_event("startup")
def on_startup() -> None:
    Base.metadata.create_all(bind=engine)


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}


@app.post("/ohlc")
def create_ohlc(data: OHLCCreate, db: Session = Depends(get_db)):
    candle = OHLC1m(**data.model_dump())
    db.add(candle)
    db.commit()
    db.refresh(candle)
    return {"message": "OHLC inserted"}


@app.get("/ohlc", response_model=list[OHLCRead])
def read_ohlc(
    symbol: str | None = None,
    limit: int = Query(default=100, ge=1),
    db: Session = Depends(get_db),
):
    return get_ohlc_data(db=db, symbol=symbol, limit=limit)
