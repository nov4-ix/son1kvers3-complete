from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    SECRET_KEY: str = "test-key"
    DATABASE_URL: str = "sqlite:///./test.db"

settings = Settings()

def get_settings():
    return settings
