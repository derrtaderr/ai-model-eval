"""
Database Performance Optimization Module
Provides intelligent query optimization, caching, and performance monitoring.
"""

import time
import logging
import asyncio
from typing import Dict, Any, List, Optional, Union, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
from functools import wraps
import json
import hashlib

from sqlalchemy import text, event, create_engine
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Query
from sqlalchemy.pool import Pool
from sqlalchemy.engine import Engine

from services.cache_service import cache_service
from config.performance import (
    MONITORING_SETTINGS, 
    CACHE_TTL_SETTINGS,
    OPTIMIZATION_FLAGS
)

logger = logging.getLogger(__name__)


class QueryPerformanceMonitor:
    """Monitor and analyze database query performance."""
    
    def __init__(self):
        self.query_stats = defaultdict(lambda: {
            "count": 0,
            "total_time": 0.0,
            "avg_time": 0.0,
            "max_time": 0.0,
            "min_time": float('inf'),
            "slow_queries": deque(maxlen=100),
            "last_seen": None
        })
        self.connection_stats = {
            "active_connections": 0,
            "peak_connections": 0,
            "total_queries": 0,
            "slow_queries": 0,
            "failed_queries": 0
        }
    
    def record_query(self, query: str, duration: float, success: bool = True):
        """Record query execution statistics."""
        query_hash = self._hash_query(query)
        stats = self.query_stats[query_hash]
        
        stats["count"] += 1
        stats["last_seen"] = datetime.utcnow()
        
        if success:
            stats["total_time"] += duration
            stats["avg_time"] = stats["total_time"] / stats["count"]
            stats["max_time"] = max(stats["max_time"], duration)
            stats["min_time"] = min(stats["min_time"], duration)
            
            # Record slow queries
            if duration > MONITORING_SETTINGS["slow_query_threshold_ms"] / 1000:
                self.connection_stats["slow_queries"] += 1
                stats["slow_queries"].append({
                    "query": query[:500] + "..." if len(query) > 500 else query,
                    "duration": duration,
                    "timestamp": datetime.utcnow().isoformat()
                })
                
                logger.warning(
                    f"Slow query detected: {duration:.3f}s - {query[:200]}..."
                )
        else:
            self.connection_stats["failed_queries"] += 1
            
        self.connection_stats["total_queries"] += 1
    
    def _hash_query(self, query: str) -> str:
        """Create hash for query pattern recognition."""
        # Normalize query for pattern recognition
        normalized = ' '.join(query.split())
        return hashlib.md5(normalized.encode()).hexdigest()[:16]
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        slow_queries = []
        top_queries = []
        
        # Collect slow queries from all patterns
        for query_hash, stats in self.query_stats.items():
            if stats["slow_queries"]:
                slow_queries.extend(stats["slow_queries"])
            
            if stats["count"] > 0:
                top_queries.append({
                    "hash": query_hash,
                    "count": stats["count"],
                    "avg_time": stats["avg_time"],
                    "total_time": stats["total_time"],
                    "max_time": stats["max_time"]
                })
        
        # Sort by impact (count * avg_time)
        top_queries.sort(key=lambda x: x["count"] * x["avg_time"], reverse=True)
        slow_queries.sort(key=lambda x: x["duration"], reverse=True)
        
        return {
            "connection_stats": self.connection_stats,
            "top_queries_by_impact": top_queries[:10],
            "recent_slow_queries": slow_queries[:20],
            "query_patterns": len(self.query_stats),
            "recommendations": self._generate_recommendations(top_queries)
        }
    
    def _generate_recommendations(self, top_queries: List[Dict]) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        for query in top_queries[:5]:
            if query["avg_time"] > 0.5:
                recommendations.append(
                    f"Consider optimizing high-impact query (hash: {query['hash']}) "
                    f"- {query['count']} executions, {query['avg_time']:.3f}s average"
                )
            
            if query["count"] > 1000:
                recommendations.append(
                    f"High-frequency query detected (hash: {query['hash']}) "
                    f"- {query['count']} executions. Consider caching."
                )
        
        if self.connection_stats["slow_queries"] > self.connection_stats["total_queries"] * 0.1:
            recommendations.append(
                "High percentage of slow queries detected. Review query patterns and indexes."
            )
        
        return recommendations


class QueryCache:
    """Intelligent query result caching system."""
    
    def __init__(self):
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_sets = 0
        
    def get_cache_key(self, query: str, params: Dict = None) -> str:
        """Generate cache key for query and parameters."""
        key_data = {
            "query": query,
            "params": params or {}
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return f"query_cache:{hashlib.md5(key_str.encode()).hexdigest()}"
    
    async def get_cached_result(self, query: str, params: Dict = None) -> Optional[Any]:
        """Retrieve cached query result."""
        if not OPTIMIZATION_FLAGS["enable_query_cache"] or not cache_service.is_available():
            return None
        
        cache_key = self.get_cache_key(query, params)
        result = cache_service.get(cache_key)
        
        if result is not None:
            self.cache_hits += 1
            logger.debug(f"Cache hit for query: {query[:100]}...")
            return result
        
        self.cache_misses += 1
        return None
    
    async def cache_result(self, query: str, result: Any, params: Dict = None, ttl: int = None):
        """Cache query result."""
        if not OPTIMIZATION_FLAGS["enable_query_cache"] or not cache_service.is_available():
            return
        
        cache_key = self.get_cache_key(query, params)
        cache_ttl = ttl or CACHE_TTL_SETTINGS.get("trace_stats", 300)
        
        cache_service.set(cache_key, result, cache_ttl)
        self.cache_sets += 1
        logger.debug(f"Cached query result: {query[:100]}...")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_sets": self.cache_sets,
            "hit_rate_percent": round(hit_rate, 2),
            "total_requests": total_requests
        }


class QueryOptimizer:
    """Automatic query optimization and analysis."""
    
    def __init__(self):
        self.optimization_rules = {
            "traces": self._optimize_trace_queries,
            "evaluations": self._optimize_evaluation_queries,
            "experiments": self._optimize_experiment_queries
        }
    
    def optimize_query(self, query: str, table_hint: str = None) -> str:
        """Apply optimization rules to query."""
        optimized_query = query
        
        # Apply table-specific optimizations
        if table_hint and table_hint in self.optimization_rules:
            optimized_query = self.optimization_rules[table_hint](optimized_query)
        
        # Apply general optimizations
        optimized_query = self._apply_general_optimizations(optimized_query)
        
        return optimized_query
    
    def _optimize_trace_queries(self, query: str) -> str:
        """Optimize trace-related queries."""
        # Force index usage for common trace patterns
        if "WHERE" in query.upper() and "team_id" in query.lower():
            if "ORDER BY timestamp" in query:
                # Use composite index for team + timestamp
                query = query.replace(
                    "ORDER BY timestamp",
                    "ORDER BY team_id, timestamp"
                )
        
        return query
    
    def _optimize_evaluation_queries(self, query: str) -> str:
        """Optimize evaluation-related queries."""
        # Add hints for evaluation aggregations
        if "GROUP BY" in query.upper() and "evaluator_type" in query.lower():
            # Ensure proper index utilization
            pass
        
        return query
    
    def _optimize_experiment_queries(self, query: str) -> str:
        """Optimize experiment-related queries."""
        return query
    
    def _apply_general_optimizations(self, query: str) -> str:
        """Apply general query optimizations."""
        # Add LIMIT if missing for potentially large result sets
        if ("SELECT" in query.upper() and 
            "LIMIT" not in query.upper() and 
            "COUNT" not in query.upper()):
            
            # Add reasonable limit for safety
            if not query.rstrip().endswith(";"):
                query += " LIMIT 1000"
            else:
                query = query.rstrip()[:-1] + " LIMIT 1000;"
        
        return query


class ConnectionPoolMonitor:
    """Monitor database connection pool performance."""
    
    def __init__(self):
        self.pool_stats = {
            "connections_created": 0,
            "connections_closed": 0,
            "connections_reused": 0,
            "pool_overflows": 0,
            "connection_errors": 0
        }
    
    def monitor_pool(self, pool: Pool):
        """Setup pool monitoring."""
        @event.listens_for(pool, "connect")
        def on_connect(dbapi_conn, connection_record):
            self.pool_stats["connections_created"] += 1
            logger.debug("Database connection created")
        
        @event.listens_for(pool, "checkout")
        def on_checkout(dbapi_conn, connection_record, connection_proxy):
            self.pool_stats["connections_reused"] += 1
        
        @event.listens_for(pool, "checkin")
        def on_checkin(dbapi_conn, connection_record):
            logger.debug("Database connection returned to pool")
        
        @event.listens_for(pool, "invalidate")
        def on_invalidate(dbapi_conn, connection_record, exception):
            self.pool_stats["connection_errors"] += 1
            logger.warning(f"Database connection invalidated: {exception}")
    
    def get_pool_status(self, pool: Pool) -> Dict[str, Any]:
        """Get current pool status."""
        try:
            return {
                "size": pool.size(),
                "checked_in": pool.checkedin(),
                "checked_out": pool.checkedout(),
                "overflow": pool.overflow(),
                "invalid": pool.invalid(),
                "stats": self.pool_stats
            }
        except Exception as e:
            logger.error(f"Error getting pool status: {e}")
            return {"error": str(e)}


# Global instances
query_monitor = QueryPerformanceMonitor()
query_cache = QueryCache()
query_optimizer = QueryOptimizer()
pool_monitor = ConnectionPoolMonitor()


def query_performance_decorator(cache_ttl: int = None, table_hint: str = None):
    """Decorator for automatic query optimization and caching."""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                # Try to get cached result first
                if hasattr(func, '__name__'):
                    cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
                    cached_result = await query_cache.get_cached_result(
                        cache_key, kwargs
                    )
                    if cached_result is not None:
                        return cached_result
                
                # Execute function
                result = await func(*args, **kwargs)
                
                # Cache successful result
                if result is not None:
                    await query_cache.cache_result(
                        cache_key, result, kwargs, cache_ttl
                    )
                
                # Record performance
                duration = time.time() - start_time
                query_monitor.record_query(
                    str(func.__name__), duration, success=True
                )
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                query_monitor.record_query(
                    str(func.__name__), duration, success=False
                )
                raise
        
        return wrapper
    return decorator


async def get_database_performance_metrics() -> Dict[str, Any]:
    """Get comprehensive database performance metrics."""
    return {
        "query_monitor": query_monitor.get_performance_report(),
        "cache_stats": query_cache.get_cache_stats(),
        "timestamp": datetime.utcnow().isoformat()
    }


async def optimize_database_settings(engine):
    """Apply database-specific optimizations."""
    if engine.dialect.name == "postgresql":
        # PostgreSQL optimizations
        async with engine.begin() as conn:
            await conn.execute(text("SET work_mem = '256MB'"))
            await conn.execute(text("SET shared_buffers = '256MB'"))
            await conn.execute(text("SET effective_cache_size = '1GB'"))
            await conn.execute(text("SET random_page_cost = 1.1"))
            
    elif engine.dialect.name == "sqlite":
        # SQLite optimizations
        async with engine.begin() as conn:
            await conn.execute(text("PRAGMA journal_mode = WAL"))
            await conn.execute(text("PRAGMA synchronous = NORMAL"))
            await conn.execute(text("PRAGMA cache_size = 10000"))
            await conn.execute(text("PRAGMA temp_store = MEMORY"))
    
    logger.info(f"Applied performance optimizations for {engine.dialect.name}")


# Database maintenance utilities
async def analyze_table_statistics(session: AsyncSession, table_name: str):
    """Analyze table statistics for optimization."""
    if session.bind.dialect.name == "postgresql":
        await session.execute(text(f"ANALYZE {table_name}"))
    elif session.bind.dialect.name == "sqlite":
        await session.execute(text(f"ANALYZE {table_name}"))
    
    logger.info(f"Updated statistics for table: {table_name}")


async def vacuum_database(session: AsyncSession):
    """Perform database maintenance."""
    if session.bind.dialect.name == "postgresql":
        await session.execute(text("VACUUM ANALYZE"))
    elif session.bind.dialect.name == "sqlite":
        await session.execute(text("VACUUM"))
    
    logger.info("Database vacuum completed") 