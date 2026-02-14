from sqlalchemy.orm import Session

from app.models.ohlc import OHLC1m


def get_ohlc_data(db: Session, symbol: str | None = None, limit: int = 100) -> list[OHLC1m]:
    query = db.query(OHLC1m)

    if symbol:
        query = query.filter(OHLC1m.symbol == symbol)

    return query.order_by(OHLC1m.timestamp.desc()).limit(limit).all()