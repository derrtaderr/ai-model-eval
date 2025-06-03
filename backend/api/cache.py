"""
Cache Management API
Provides endpoints for Redis cache monitoring, management, and performance analytics.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional, List
from datetime import datetime
from pydantic import BaseModel, Field

from services.cache_service import cache_service, trace_cache, evaluation_cache
from services.redis_service import redis_service, warm_up_cache, cleanup_expired_cache
from auth.security import require_role, get_current_user
from auth.models import UserRole

router = APIRouter(prefix="/cache", tags=["Cache"])


class CacheStatsResponse(BaseModel):
    """Response model for cache statistics."""
    backend: str
    redis_available: bool
    cache_stats: Dict[str, Any]
    memory_usage: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    timestamp: str


class CacheKeyResponse(BaseModel):
    """Response model for cache key operations."""
    key: str
    exists: bool
    ttl: int
    size_bytes: Optional[int]
    value_type: str


class CacheBulkOperationResponse(BaseModel):
    """Response model for bulk cache operations."""
    operation: str
    keys_affected: int
    success: bool
    errors: List[str]
    execution_time_ms: float


@router.get("/stats", response_model=CacheStatsResponse)
async def get_cache_statistics(
    include_memory: bool = Query(True, description="Include Redis memory usage"),
    current_user = Depends(get_current_user)
):
    """
    Get comprehensive cache performance statistics.
    Requires admin or team admin role.
    """
    # Check permissions
    if current_user.role not in [UserRole.SUPER_ADMIN, UserRole.TEAM_ADMIN]:
        raise HTTPException(
            status_code=403,
            detail="Insufficient permissions to view cache statistics"
        )
    
    try:
        # Get cache service stats
        cache_stats = await cache_service.get_stats()
        
        # Get Redis memory usage if available and requested
        memory_usage = {}
        if include_memory and redis_service.is_available():
            memory_usage = await redis_service.get_memory_usage()
        
        # Get Redis-specific metrics
        redis_metrics = {}
        if redis_service.is_available():
            redis_metrics = redis_service.get_metrics()
        
        return CacheStatsResponse(
            backend=cache_stats.get("backend", "memory"),
            redis_available=redis_service.is_available(),
            cache_stats=cache_stats,
            memory_usage=memory_usage,
            performance_metrics=redis_metrics,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving cache statistics: {str(e)}"
        )


@router.get("/keys/{key_name}")
async def get_cache_key_info(
    key_name: str,
    prefix: str = Query("api", description="Cache key prefix"),
    current_user = Depends(get_current_user)
):
    """
    Get information about a specific cache key.
    Requires admin role.
    """
    if current_user.role not in [UserRole.SUPER_ADMIN, UserRole.TEAM_ADMIN]:
        raise HTTPException(
            status_code=403,
            detail="Insufficient permissions to view cache keys"
        )
    
    try:
        # Check if key exists
        exists = await cache_service.exists(key_name, prefix)
        
        # Get TTL
        ttl = await cache_service.get_ttl(key_name, prefix) if exists else -1
        
        # Get value to determine type and size
        value = None
        size_bytes = None
        value_type = "unknown"
        
        if exists:
            value = await cache_service.get(key_name, prefix=prefix)
            if value is not None:
                import sys
                size_bytes = sys.getsizeof(value)
                value_type = type(value).__name__
        
        return CacheKeyResponse(
            key=f"{prefix}:{key_name}",
            exists=exists,
            ttl=ttl,
            size_bytes=size_bytes,
            value_type=value_type
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving cache key info: {str(e)}"
        )


@router.get("/keys/{key_name}/value")
async def get_cache_key_value(
    key_name: str,
    prefix: str = Query("api", description="Cache key prefix"),
    current_user = Depends(get_current_user)
):
    """
    Get the actual value of a cache key.
    Requires super admin role for security.
    """
    if current_user.role != UserRole.SUPER_ADMIN:
        raise HTTPException(
            status_code=403,
            detail="Insufficient permissions to view cache values"
        )
    
    try:
        value = await cache_service.get(key_name, prefix=prefix)
        
        if value is None:
            raise HTTPException(
                status_code=404,
                detail=f"Cache key not found: {prefix}:{key_name}"
            )
        
        return {
            "key": f"{prefix}:{key_name}",
            "value": value,
            "type": type(value).__name__,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving cache value: {str(e)}"
        )


@router.delete("/keys/{key_name}")
async def delete_cache_key(
    key_name: str,
    prefix: str = Query("api", description="Cache key prefix"),
    current_user = Depends(get_current_user)
):
    """
    Delete a specific cache key.
    Requires admin role.
    """
    if current_user.role not in [UserRole.SUPER_ADMIN, UserRole.TEAM_ADMIN]:
        raise HTTPException(
            status_code=403,
            detail="Insufficient permissions to delete cache keys"
        )
    
    try:
        success = await cache_service.delete(key_name, prefix)
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Cache key not found or deletion failed: {prefix}:{key_name}"
            )
        
        return {
            "message": f"Cache key deleted successfully: {prefix}:{key_name}",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting cache key: {str(e)}"
        )


@router.delete("/patterns/{pattern}")
async def delete_cache_pattern(
    pattern: str,
    prefix: str = Query("api", description="Cache key prefix"),
    current_user = Depends(get_current_user)
):
    """
    Delete all keys matching a pattern.
    Requires super admin role.
    """
    if current_user.role != UserRole.SUPER_ADMIN:
        raise HTTPException(
            status_code=403,
            detail="Insufficient permissions to delete cache patterns"
        )
    
    try:
        start_time = datetime.utcnow()
        deleted_count = await cache_service.delete_pattern(pattern, prefix)
        end_time = datetime.utcnow()
        
        execution_time = (end_time - start_time).total_seconds() * 1000
        
        return CacheBulkOperationResponse(
            operation="delete_pattern",
            keys_affected=deleted_count,
            success=True,
            errors=[],
            execution_time_ms=execution_time
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting cache pattern: {str(e)}"
        )


@router.post("/warm-up")
async def warm_up_cache_endpoint(
    current_user = Depends(get_current_user)
):
    """
    Warm up cache with frequently accessed data.
    Requires admin role.
    """
    if current_user.role not in [UserRole.SUPER_ADMIN, UserRole.TEAM_ADMIN]:
        raise HTTPException(
            status_code=403,
            detail="Insufficient permissions to warm up cache"
        )
    
    try:
        start_time = datetime.utcnow()
        await warm_up_cache()
        end_time = datetime.utcnow()
        
        execution_time = (end_time - start_time).total_seconds() * 1000
        
        return {
            "message": "Cache warm-up completed successfully",
            "execution_time_ms": execution_time,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error warming up cache: {str(e)}"
        )


@router.post("/cleanup")
async def cleanup_cache_endpoint(
    current_user = Depends(get_current_user)
):
    """
    Clean up expired cache entries.
    Requires admin role.
    """
    if current_user.role not in [UserRole.SUPER_ADMIN, UserRole.TEAM_ADMIN]:
        raise HTTPException(
            status_code=403,
            detail="Insufficient permissions to clean up cache"
        )
    
    try:
        start_time = datetime.utcnow()
        await cleanup_expired_cache()
        end_time = datetime.utcnow()
        
        execution_time = (end_time - start_time).total_seconds() * 1000
        
        return {
            "message": "Cache cleanup completed successfully",
            "execution_time_ms": execution_time,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error cleaning up cache: {str(e)}"
        )


@router.delete("/clear-all")
async def clear_all_cache(
    confirm: bool = Query(False, description="Confirmation flag to prevent accidental clearing"),
    current_user = Depends(get_current_user)
):
    """
    Clear ALL cache data (use with extreme caution!).
    Requires super admin role and explicit confirmation.
    """
    if current_user.role != UserRole.SUPER_ADMIN:
        raise HTTPException(
            status_code=403,
            detail="Insufficient permissions to clear all cache"
        )
    
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Confirmation required. Set confirm=true to proceed."
        )
    
    try:
        start_time = datetime.utcnow()
        success = await cache_service.clear_all()
        end_time = datetime.utcnow()
        
        execution_time = (end_time - start_time).total_seconds() * 1000
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to clear cache"
            )
        
        return {
            "message": "⚠️ ALL cache data cleared successfully",
            "execution_time_ms": execution_time,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error clearing all cache: {str(e)}"
        )


@router.post("/invalidate/team/{team_id}")
async def invalidate_team_cache(
    team_id: str,
    cache_type: str = Query("all", description="Cache type to invalidate (all, trace, evaluation)"),
    current_user = Depends(get_current_user)
):
    """
    Invalidate all cache for a specific team.
    Requires admin role or team membership.
    """
    # Check permissions (team admin or member of the team)
    if (current_user.role not in [UserRole.SUPER_ADMIN, UserRole.TEAM_ADMIN] and 
        getattr(current_user, 'default_team_id', None) != team_id):
        raise HTTPException(
            status_code=403,
            detail="Insufficient permissions to invalidate team cache"
        )
    
    try:
        start_time = datetime.utcnow()
        keys_affected = 0
        
        if cache_type in ["all", "trace"]:
            await trace_cache.invalidate_team_cache(team_id)
            keys_affected += 1
        
        if cache_type in ["all", "evaluation"]:
            # Invalidate evaluation cache for team
            pattern = f"evaluation:*{team_id}*"
            deleted = await cache_service.delete_pattern(pattern)
            keys_affected += deleted
        
        if cache_type == "all":
            # Invalidate general team cache
            pattern = f"*{team_id}*"
            deleted = await cache_service.delete_pattern(pattern)
            keys_affected += deleted
        
        end_time = datetime.utcnow()
        execution_time = (end_time - start_time).total_seconds() * 1000
        
        return CacheBulkOperationResponse(
            operation=f"invalidate_team_{cache_type}",
            keys_affected=keys_affected,
            success=True,
            errors=[],
            execution_time_ms=execution_time
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error invalidating team cache: {str(e)}"
        )


@router.post("/test-connection")
async def test_redis_connection(
    current_user = Depends(get_current_user)
):
    """
    Test Redis connection and performance.
    Requires admin role.
    """
    if current_user.role not in [UserRole.SUPER_ADMIN, UserRole.TEAM_ADMIN]:
        raise HTTPException(
            status_code=403,
            detail="Insufficient permissions to test cache connection"
        )
    
    try:
        start_time = datetime.utcnow()
        
        # Test basic operations
        test_key = f"test:connection:{int(start_time.timestamp())}"
        test_value = {"test": True, "timestamp": start_time.isoformat()}
        
        # Test set operation
        set_success = await cache_service.set(test_key, test_value, ttl=60, prefix="system")
        
        # Test get operation
        retrieved_value = await cache_service.get(test_key, prefix="system")
        
        # Test delete operation
        delete_success = await cache_service.delete(test_key, prefix="system")
        
        end_time = datetime.utcnow()
        execution_time = (end_time - start_time).total_seconds() * 1000
        
        # Verify operations
        operations_successful = (
            set_success and 
            retrieved_value == test_value and 
            delete_success
        )
        
        return {
            "redis_available": redis_service.is_available(),
            "operations_successful": operations_successful,
            "test_results": {
                "set_operation": set_success,
                "get_operation": retrieved_value is not None,
                "delete_operation": delete_success,
                "value_integrity": retrieved_value == test_value
            },
            "performance": {
                "total_time_ms": execution_time,
                "operations_per_second": round(3 / (execution_time / 1000), 2) if execution_time > 0 else 0
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error testing cache connection: {str(e)}"
        )


@router.get("/health")
async def cache_health_check():
    """
    Public cache health check endpoint.
    """
    try:
        redis_available = redis_service.is_available()
        cache_available = cache_service.is_available()
        
        # Quick performance test
        start_time = datetime.utcnow()
        test_key = f"health:check:{int(start_time.timestamp())}"
        
        set_success = await cache_service.set(test_key, "health_check", ttl=10, prefix="system")
        get_success = await cache_service.get(test_key, prefix="system") == "health_check"
        
        end_time = datetime.utcnow()
        response_time_ms = (end_time - start_time).total_seconds() * 1000
        
        # Clean up test key
        await cache_service.delete(test_key, prefix="system")
        
        status = "healthy" if (redis_available and cache_available and set_success and get_success) else "degraded"
        
        return {
            "status": status,
            "redis_available": redis_available,
            "cache_available": cache_available,
            "response_time_ms": response_time_ms,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "redis_available": False,
            "cache_available": False,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        } 