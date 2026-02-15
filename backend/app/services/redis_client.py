import redis

from app.config import settings

redis_client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)


def set_value(key: str, value: str):
    return redis_client.set(key, value)


def get_value(key: str):
    return redis_client.get(key)
