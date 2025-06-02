"""
Database connection configuration for the LLM Evaluation Platform.
Optimized for performance with connection pooling and monitoring.
"""

import os
import logging
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
from decouple import config

from .models import Base
from config.performance import DATABASE_POOL_SETTINGS

logger = logging.getLogger(__name__)

# Database configuration
DATABASE_URL = config(
    "DATABASE_URL",
    default="postgresql+asyncpg://user:password@localhost:5432/llm_eval_platform"
)

# Create async engine with optimized settings
engine = create_async_engine(
    DATABASE_URL,
    echo=config("DATABASE_ECHO", default=False, cast=bool),
    
    # Connection pool optimization
    pool_size=DATABASE_POOL_SETTINGS["pool_size"],
    max_overflow=DATABASE_POOL_SETTINGS["max_overflow"],
    pool_timeout=DATABASE_POOL_SETTINGS["pool_timeout"],
    pool_recycle=DATABASE_POOL_SETTINGS["pool_recycle"],
    pool_pre_ping=DATABASE_POOL_SETTINGS["pool_pre_ping"],
    
    # Performance optimizations
    poolclass=QueuePool,
    
    # Connection parameters for PostgreSQL optimization
    connect_args={
        "command_timeout": 30,
        "server_settings": {
            "application_name": "llm_eval_platform",
            "jit": "off",  # Disable JIT for faster simple queries
        },
    },
    
    # Query optimization
    future=True,  # Use SQLAlchemy 2.0 style
)

# Create async session factory
AsyncSessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    
    # Performance optimizations
    autoflush=False,  # Manual control over when to flush
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency to get async database session.
    Optimized for performance with proper error handling.
    """
    session = AsyncSessionLocal()
    try:
        yield session
        await session.commit()
    except Exception as e:
        await session.rollback()
        logger.error(f"Database session error: {e}")
        raise
    finally:
        await session.close()


async def create_tables():
    """
    Create all database tables.
    """
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to create database tables: {e}")
        raise


async def drop_tables():
    """
    Drop all database tables.
    """
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        logger.info("Database tables dropped successfully")
    except Exception as e:
        logger.error(f"Failed to drop database tables: {e}")
        raise


async def get_db_stats():
    """
    Get database connection pool statistics.
    """
    try:
        pool = engine.pool
        return {
            "pool_size": pool.size(),
            "checked_in": pool.checkedin(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "invalid": pool.invalid(),
        }
    except Exception as e:
        logger.error(f"Failed to get database stats: {e}")
        return {"error": str(e)}


# Health check function
async def check_database_health():
    """
    Check if database connection is healthy.
    """
    try:
        async with engine.begin() as conn:
            result = await conn.execute("SELECT 1")
            await result.fetchone()
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)} 