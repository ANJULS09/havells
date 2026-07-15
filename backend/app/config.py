import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    # App Settings
    APP_NAME: str = "Havells Customer Voice Intelligence Agent"
    DEBUG: bool = True
    API_V1_STR: str = "/api/v1"

    # Database Settings
    # Default to sqlite relative to this file
    DATABASE_URL: str = "sqlite:///./voice_cve.db"
    
    # Vector DB Settings
    # Default to "local" file-based storage. Can be "qdrant"
    VECTOR_DB_TYPE: str = "local"
    VECTOR_DB_URL: Optional[str] = None
    VECTOR_DB_API_KEY: Optional[str] = None
    VECTOR_DB_COLLECTION: str = "havells_reviews"

    # LLM Settings
    # Supported: "mock", "gemini", "openai", "ollama"
    LLM_PROVIDER: str = "mock"
    GEMINI_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    OLLAMA_HOST: str = "http://localhost:11434"
    
    # Selected Models
    EMBEDDING_MODEL: str = "text-embedding-3-small"  # or BGE / gemini-embeddings
    LLM_MODEL: str = "gemini-1.5-flash"  # or gpt-4o, llama3
    
    # Security / CORS
    CORS_ORIGINS: list[str] = ["*"]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
