"""
Database connection configuration for the LLM Evaluation Platform.
"""

import os
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from decouple import config

from .models import Base

# Database configuration
DATABASE_URL = config(
    "DATABASE_URL",
    default="postgresql+asyncpg://user:password@localhost:5432/llm_eval_platform"
)

# Create async engine
engine = create_async_engine(
    DATABASE_URL,
    echo=config("DATABASE_ECHO", default=False, cast=bool),
    pool_size=config("DATABASE_POOL_SIZE", default=10, cast=int),
    max_overflow=config("DATABASE_MAX_OVERFLOW", default=20, cast=int),
)

# Create async session factory
AsyncSessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency to get async database session.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def create_tables():
    """
    Create all database tables.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def drop_tables():
    """
    Drop all database tables.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all) 