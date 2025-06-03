"""
Application settings and configuration management.
"""

import os
from typing import Optional, List
from dataclasses import dataclass, field
from decouple import config


@dataclass
class Settings:
    """Application settings configuration."""
    
    # Database settings
    database_url: str = config("DATABASE_URL", default="sqlite:///./llm_eval.db")
    
    # API Keys for LLM services
    openai_api_key: Optional[str] = config("OPENAI_API_KEY", default=None)
    anthropic_api_key: Optional[str] = config("ANTHROPIC_API_KEY", default=None)
    google_api_key: Optional[str] = config("GOOGLE_API_KEY", default=None)
    
    # LangSmith configuration
    langchain_api_key: Optional[str] = config("LANGCHAIN_API_KEY", default=None)
    langchain_project: str = config("LANGCHAIN_PROJECT", default="llm-eval-platform")
    
    # Application settings
    debug: bool = config("DEBUG", default=False, cast=bool)
    secret_key: str = config("SECRET_KEY", default="dev-secret-key-change-in-production")
    
    # CORS settings
    ALLOWED_ORIGINS: List[str] = field(default_factory=lambda: config(
        "ALLOWED_ORIGINS", 
        default="http://localhost:3000,http://127.0.0.1:3000,http://localhost:3001", 
        cast=lambda v: [origin.strip() for origin in v.split(",")]
    ))
    
    # Taxonomy builder settings
    taxonomy_cache_hours: int = config("TAXONOMY_CACHE_HOURS", default=24, cast=int)
    max_traces_for_analysis: int = config("MAX_TRACES_FOR_ANALYSIS", default=1000, cast=int)


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings 