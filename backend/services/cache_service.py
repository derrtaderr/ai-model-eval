"""
Enhanced Cache Service with Redis Backend
Provides backward-compatible caching interface using Redis for persistence and performance.
"""

import logging
from typing import Any, Optional, Dict, List
from datetime import datetime, timedelta
import json
import hashlib

from .redis_service import redis_service, cache_result
from config.performance import CACHE_TTL_SETTINGS, CACHE_KEY_PREFIXES

logger = logging.getLogger(__name__)


class CacheService:
    """
    Enhanced cache service using Redis backend with fallback to in-memory cache.
    Maintains backward compatibility with existing cache_service interface.
    """
    
    def __init__(self):
        self._memory_cache = {}  # Fallback for when Redis is unavailable
        self._memory_expiry = {}
        self._initialized = False
    
    async def initialize(self):
        """Initialize Redis backend."""
        if not self._initialized:
            await redis_service.initialize()
            self._initialized = True
            
            if redis_service.is_available():
                logger.info("✅ Cache service initialized with Redis backend")
            else:
                logger.warning("⚠️ Cache service falling back to in-memory storage")
    
    def is_available(self) -> bool:
        """Check if caching is available (Redis or memory fallback)."""
        return redis_service.is_available() if self._initialized else True
    
    def _generate_key(self, key: str, prefix: str = "api") -> str:
        """Generate standardized cache key with prefix."""
        prefix_str = CACHE_KEY_PREFIXES.get(prefix, prefix)
        return f"{prefix_str}:{key}"
    
    async def get(self, key: str, default: Any = None, prefix: str = "api") -> Any:
        """
        Get value from cache with Redis backend and memory fallback.
        
        Args:
            key: Cache key
            default: Default value if key not found
            prefix: Key prefix for organization
        """
        full_key = self._generate_key(key, prefix)
        
        try:
            # Try Redis first if available
            if redis_service.is_available():
                result = await redis_service.get(full_key)
                if result is not None:
                    return result
                return default
            
            # Fallback to memory cache
            if full_key in self._memory_cache:
                # Check expiry
                if full_key in self._memory_expiry:
                    if datetime.utcnow() > self._memory_expiry[full_key]:
                        del self._memory_cache[full_key]
                        del self._memory_expiry[full_key]
                        return default
                
                return self._memory_cache[full_key]
            
            return default
            
        except Exception as e:
            logger.error(f"Cache get error for key '{full_key}': {e}")
            return default
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None, 
        prefix: str = "api"
    ) -> bool:
        """
        Set value in cache with TTL.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            prefix: Key prefix for organization
        """
        full_key = self._generate_key(key, prefix)
        cache_ttl = ttl or CACHE_TTL_SETTINGS.get("api_responses", 300)
        
        try:
            # Try Redis first if available
            if redis_service.is_available():
                return await redis_service.set(full_key, value, cache_ttl)
            
            # Fallback to memory cache
            self._memory_cache[full_key] = value
            if cache_ttl:
                self._memory_expiry[full_key] = datetime.utcnow() + timedelta(seconds=cache_ttl)
            
            return True
            
        except Exception as e:
            logger.error(f"Cache set error for key '{full_key}': {e}")
            return False
    
    async def delete(self, key: str, prefix: str = "api") -> bool:
        """
        Delete key from cache.
        
        Args:
            key: Cache key to delete
            prefix: Key prefix for organization
        """
        full_key = self._generate_key(key, prefix)
        
        try:
            # Try Redis first if available
            if redis_service.is_available():
                return await redis_service.delete(full_key)
            
            # Fallback to memory cache
            if full_key in self._memory_cache:
                del self._memory_cache[full_key]
            if full_key in self._memory_expiry:
                del self._memory_expiry[full_key]
            
            return True
            
        except Exception as e:
            logger.error(f"Cache delete error for key '{full_key}': {e}")
            return False
    
    async def delete_pattern(self, pattern: str, prefix: str = "api") -> int:
        """
        Delete all keys matching a pattern.
        
        Args:
            pattern: Pattern to match (e.g., "user:*")
            prefix: Key prefix for organization
        """
        full_pattern = self._generate_key(pattern, prefix)
        
        try:
            # Try Redis first if available
            if redis_service.is_available():
                return await redis_service.delete_pattern(full_pattern)
            
            # Fallback to memory cache (basic pattern matching)
            deleted_count = 0
            keys_to_delete = []
            
            for key in self._memory_cache.keys():
                if key.startswith(full_pattern.replace("*", "")):
                    keys_to_delete.append(key)
            
            for key in keys_to_delete:
                del self._memory_cache[key]
                if key in self._memory_expiry:
                    del self._memory_expiry[key]
                deleted_count += 1
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"Cache pattern delete error for pattern '{full_pattern}': {e}")
            return 0
    
    async def exists(self, key: str, prefix: str = "api") -> bool:
        """
        Check if key exists in cache.
        
        Args:
            key: Cache key to check
            prefix: Key prefix for organization
        """
        full_key = self._generate_key(key, prefix)
        
        try:
            # Try Redis first if available
            if redis_service.is_available():
                return await redis_service.exists(full_key)
            
            # Fallback to memory cache
            if full_key in self._memory_cache:
                # Check expiry
                if full_key in self._memory_expiry:
                    if datetime.utcnow() > self._memory_expiry[full_key]:
                        del self._memory_cache[full_key]
                        del self._memory_expiry[full_key]
                        return False
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Cache exists error for key '{full_key}': {e}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        try:
            if redis_service.is_available():
                redis_stats = redis_service.get_metrics()
                redis_memory = await redis_service.get_memory_usage()
                
                return {
                    "backend": "redis",
                    "redis_stats": redis_stats,
                    "redis_memory": redis_memory,
                    "memory_fallback_keys": len(self._memory_cache)
                }
            else:
                return {
                    "backend": "memory",
                    "memory_keys": len(self._memory_cache),
                    "memory_expired_keys": len(self._memory_expiry)
                }
                
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"error": str(e)}
    
    async def clear_all(self) -> bool:
        """Clear all cache data (use with caution!)."""
        try:
            success = True
            
            # Clear Redis if available
            if redis_service.is_available():
                success = await redis_service.clear_all()
            
            # Clear memory cache
            self._memory_cache.clear()
            self._memory_expiry.clear()
            
            logger.warning("⚠️ All cache data cleared!")
            return success
            
        except Exception as e:
            logger.error(f"Cache clear all error: {e}")
            return False
    
    # Advanced Redis-specific methods
    async def increment(self, key: str, amount: int = 1, prefix: str = "api") -> int:
        """Increment a counter (Redis only)."""
        full_key = self._generate_key(key, prefix)
        
        if redis_service.is_available():
            return await redis_service.increment(full_key, amount)
        
        # Memory fallback
        current = self._memory_cache.get(full_key, 0)
        new_value = current + amount
        self._memory_cache[full_key] = new_value
        return new_value
    
    async def get_ttl(self, key: str, prefix: str = "api") -> int:
        """Get remaining TTL for a key (Redis only)."""
        full_key = self._generate_key(key, prefix)
        
        if redis_service.is_available():
            return await redis_service.get_ttl(full_key)
        
        # Memory fallback
        if full_key in self._memory_expiry:
            remaining = (self._memory_expiry[full_key] - datetime.utcnow()).total_seconds()
            return max(0, int(remaining))
        
        return -1
    
    async def expire(self, key: str, ttl: int, prefix: str = "api") -> bool:
        """Set expiration time for existing key."""
        full_key = self._generate_key(key, prefix)
        
        if redis_service.is_available():
            return await redis_service.expire(full_key, ttl)
        
        # Memory fallback
        if full_key in self._memory_cache:
            self._memory_expiry[full_key] = datetime.utcnow() + timedelta(seconds=ttl)
            return True
        
        return False


# Enhanced caching decorators using Redis backend
def cache_function_result(
    ttl: Optional[int] = None,
    prefix: str = "func",
    key_builder: Optional[callable] = None
):
    """
    Enhanced function result caching decorator with Redis backend.
    
    Args:
        ttl: Cache TTL in seconds
        prefix: Cache key prefix
        key_builder: Custom function to build cache key
    """
    return cache_result(ttl=ttl, key_prefix=prefix)


# Specialized cache managers for different data types
class TraceCacheManager:
    """Specialized cache manager for trace data."""
    
    def __init__(self, cache_service: CacheService):
        self.cache = cache_service
    
    async def get_trace_stats(self, team_id: str, date_range_key: str) -> Optional[Dict]:
        """Get cached trace statistics."""
        key = f"stats:{team_id}:{date_range_key}"
        return await self.cache.get(key, prefix="trace")
    
    async def set_trace_stats(self, team_id: str, date_range_key: str, stats: Dict):
        """Cache trace statistics."""
        key = f"stats:{team_id}:{date_range_key}"
        ttl = CACHE_TTL_SETTINGS["trace_stats"]
        await self.cache.set(key, stats, ttl, prefix="trace")
    
    async def invalidate_team_cache(self, team_id: str):
        """Invalidate all trace cache for a team."""
        pattern = f"trace:{team_id}:*"
        await self.cache.delete_pattern(pattern)


class EvaluationCacheManager:
    """Specialized cache manager for evaluation data."""
    
    def __init__(self, cache_service: CacheService):
        self.cache = cache_service
    
    async def get_evaluation_summary(self, team_id: str, filters_key: str) -> Optional[Dict]:
        """Get cached evaluation summary."""
        key = f"summary:{team_id}:{filters_key}"
        return await self.cache.get(key, prefix="evaluation")
    
    async def set_evaluation_summary(self, team_id: str, filters_key: str, summary: Dict):
        """Cache evaluation summary."""
        key = f"summary:{team_id}:{filters_key}"
        ttl = CACHE_TTL_SETTINGS["evaluation_summaries"]
        await self.cache.set(key, summary, ttl, prefix="evaluation")


# Global cache service instance
cache_service = CacheService()

# Specialized managers
trace_cache = TraceCacheManager(cache_service)
evaluation_cache = EvaluationCacheManager(cache_service)

# Export everything
__all__ = [
    "cache_service", 
    "trace_cache", 
    "evaluation_cache",
    "cache_function_result"
] 