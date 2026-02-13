"""Application configuration."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
	APP_NAME: str = "Quant Research Engine"
	DEBUG: bool = True
	POSTGRES_URL: str
	REDIS_URL: str

	model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
