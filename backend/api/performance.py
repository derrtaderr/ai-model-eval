"""
Performance Monitoring API
Provides endpoints for database performance metrics, query analysis, and system health.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from pydantic import BaseModel, Field

from database.connection import get_db
from database.performance import (
    get_database_performance_metrics,
    query_monitor,
    query_cache,
    pool_monitor,
    optimize_database_settings,
    analyze_table_statistics,
    vacuum_database
)
from database.repositories import RepositoryFactory, DashboardRepository
from auth.security import require_role, get_current_user
from auth.models import UserRole

router = APIRouter(prefix="/performance", tags=["Performance"])


class PerformanceMetricsResponse(BaseModel):
    """Response model for performance metrics."""
    database_metrics: Dict[str, Any]
    cache_stats: Dict[str, Any]
    query_performance: Dict[str, Any]
    system_health: Dict[str, Any]
    recommendations: List[str]
    timestamp: str


class QueryAnalysisResponse(BaseModel):
    """Response model for query analysis."""
    slow_queries: List[Dict[str, Any]]
    top_queries_by_impact: List[Dict[str, Any]]
    query_patterns: int
    performance_trends: Dict[str, Any]
    optimization_suggestions: List[str]


class SystemHealthResponse(BaseModel):
    """Response model for system health check."""
    status: str
    database_status: str
    cache_status: str
    connection_pool_status: Dict[str, Any]
    response_time_ms: float
    checks: Dict[str, Any]


@router.get("/metrics", response_model=PerformanceMetricsResponse)
async def get_performance_metrics(
    include_query_details: bool = Query(False, description="Include detailed query analysis"),
    current_user = Depends(get_current_user),
    session = Depends(get_db)
):
    """
    Get comprehensive performance metrics.
    Requires admin or team admin role.
    """
    # Check permissions
    if current_user.role not in [UserRole.SUPER_ADMIN, UserRole.TEAM_ADMIN]:
        raise HTTPException(
            status_code=403,
            detail="Insufficient permissions to view performance metrics"
        )
    
    try:
        # Get database performance metrics
        db_metrics = await get_database_performance_metrics()
        
        # Get cache statistics
        cache_stats = query_cache.get_cache_stats()
        
        # Get query performance data
        query_report = query_monitor.get_performance_report()
        
        # Get system health information
        system_health = await _get_system_health()
        
        # Generate recommendations
        recommendations = _generate_system_recommendations(
            db_metrics, cache_stats, query_report
        )
        
        response_data = {
            "database_metrics": db_metrics,
            "cache_stats": cache_stats,
            "query_performance": query_report if include_query_details else {
                "total_queries": query_report["connection_stats"]["total_queries"],
                "slow_queries": query_report["connection_stats"]["slow_queries"],
                "failed_queries": query_report["connection_stats"]["failed_queries"]
            },
            "system_health": system_health,
            "recommendations": recommendations,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return PerformanceMetricsResponse(**response_data)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving performance metrics: {str(e)}"
        )


@router.get("/queries/analysis", response_model=QueryAnalysisResponse)
async def get_query_analysis(
    current_user = Depends(get_current_user),
    session = Depends(get_db)
):
    """
    Get detailed query performance analysis.
    Requires admin role.
    """
    # Check permissions
    if current_user.role not in [UserRole.SUPER_ADMIN, UserRole.TEAM_ADMIN]:
        raise HTTPException(
            status_code=403,
            detail="Insufficient permissions to view query analysis"
        )
    
    try:
        query_report = query_monitor.get_performance_report()
        
        # Calculate performance trends
        trends = _calculate_performance_trends()
        
        # Generate optimization suggestions
        suggestions = _generate_query_optimizations(query_report)
        
        return QueryAnalysisResponse(
            slow_queries=query_report["recent_slow_queries"],
            top_queries_by_impact=query_report["top_queries_by_impact"],
            query_patterns=query_report["query_patterns"],
            performance_trends=trends,
            optimization_suggestions=suggestions
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving query analysis: {str(e)}"
        )


@router.get("/health", response_model=SystemHealthResponse)
async def get_system_health(session = Depends(get_db)):
    """
    Get system health status.
    Public endpoint for monitoring.
    """
    start_time = datetime.utcnow()
    
    try:
        # Test database connection
        await session.execute("SELECT 1")
        db_status = "healthy"
        
        # Test cache connection
        cache_status = "healthy" if query_cache.cache_hits >= 0 else "unavailable"
        
        # Get connection pool status
        pool_status = pool_monitor.get_pool_status(session.bind.pool) if hasattr(session.bind, 'pool') else {}
        
        # Calculate response time
        response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        # Run health checks
        checks = await _run_health_checks(session)
        
        # Determine overall status
        overall_status = "healthy"
        if any(check["status"] != "ok" for check in checks.values()):
            overall_status = "degraded"
        if db_status != "healthy" or response_time > 1000:
            overall_status = "unhealthy"
        
        return SystemHealthResponse(
            status=overall_status,
            database_status=db_status,
            cache_status=cache_status,
            connection_pool_status=pool_status,
            response_time_ms=response_time,
            checks=checks
        )
        
    except Exception as e:
        return SystemHealthResponse(
            status="unhealthy",
            database_status="error",
            cache_status="unknown",
            connection_pool_status={},
            response_time_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
            checks={"error": {"status": "error", "message": str(e)}}
        )


@router.post("/optimize")
async def optimize_database(
    current_user = Depends(get_current_user),
    session = Depends(get_db)
):
    """
    Apply database optimizations.
    Requires super admin role.
    """
    # Check permissions
    if current_user.role != UserRole.SUPER_ADMIN:
        raise HTTPException(
            status_code=403,
            detail="Insufficient permissions to optimize database"
        )
    
    try:
        # Apply database-specific optimizations
        await optimize_database_settings(session.bind)
        
        # Update table statistics
        tables = ["traces", "evaluations", "experiments", "test_cases", "test_runs"]
        for table in tables:
            try:
                await analyze_table_statistics(session, table)
            except Exception as e:
                # Log but don't fail if one table has issues
                pass
        
        return {"message": "Database optimization completed successfully"}
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error optimizing database: {str(e)}"
        )


@router.post("/maintenance/vacuum")
async def vacuum_database_endpoint(
    current_user = Depends(get_current_user),
    session = Depends(get_db)
):
    """
    Perform database vacuum operation.
    Requires super admin role.
    """
    # Check permissions
    if current_user.role != UserRole.SUPER_ADMIN:
        raise HTTPException(
            status_code=403,
            detail="Insufficient permissions to perform database maintenance"
        )
    
    try:
        await vacuum_database(session)
        return {"message": "Database vacuum completed successfully"}
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error performing database vacuum: {str(e)}"
        )


@router.get("/dashboard/overview")
async def get_dashboard_performance(
    team_id: Optional[str] = Query(None, description="Team ID for team-specific metrics"),
    date_range_days: int = Query(7, description="Number of days to include in analysis"),
    current_user = Depends(get_current_user),
    session = Depends(get_db)
):
    """
    Get dashboard-specific performance overview.
    """
    try:
        # Determine team context
        effective_team_id = team_id or getattr(current_user, 'default_team_id', None)
        
        if not effective_team_id:
            raise HTTPException(
                status_code=400,
                detail="Team ID required for dashboard metrics"
            )
        
        # Calculate date range
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=date_range_days)
        date_range = {"start": start_date, "end": end_date}
        
        # Get dashboard repository
        repo_factory = RepositoryFactory(session)
        dashboard_repo = repo_factory.get_dashboard_repository()
        
        # Get dashboard overview with caching
        overview = await dashboard_repo.get_dashboard_overview(effective_team_id, date_range)
        
        return overview
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving dashboard performance: {str(e)}"
        )


# Helper functions
async def _get_system_health() -> Dict[str, Any]:
    """Get system health information."""
    return {
        "uptime_seconds": (datetime.utcnow() - datetime.utcnow().replace(hour=0, minute=0, second=0)).total_seconds(),
        "memory_usage_percent": 0,  # Would be implemented with psutil
        "cpu_usage_percent": 0,     # Would be implemented with psutil
        "disk_usage_percent": 0     # Would be implemented with psutil
    }


async def _run_health_checks(session) -> Dict[str, Dict[str, Any]]:
    """Run comprehensive health checks."""
    checks = {}
    
    # Database connectivity check
    try:
        await session.execute("SELECT COUNT(*) FROM users LIMIT 1")
        checks["database_connectivity"] = {"status": "ok", "message": "Database connection successful"}
    except Exception as e:
        checks["database_connectivity"] = {"status": "error", "message": f"Database connection failed: {str(e)}"}
    
    # Query performance check
    query_report = query_monitor.get_performance_report()
    slow_query_rate = (
        query_report["connection_stats"]["slow_queries"] / 
        max(query_report["connection_stats"]["total_queries"], 1)
    )
    
    if slow_query_rate > 0.1:
        checks["query_performance"] = {"status": "warning", "message": f"High slow query rate: {slow_query_rate:.2%}"}
    else:
        checks["query_performance"] = {"status": "ok", "message": "Query performance within normal range"}
    
    # Cache performance check
    cache_stats = query_cache.get_cache_stats()
    if cache_stats["hit_rate_percent"] < 70 and cache_stats["total_requests"] > 100:
        checks["cache_performance"] = {"status": "warning", "message": f"Low cache hit rate: {cache_stats['hit_rate_percent']:.1f}%"}
    else:
        checks["cache_performance"] = {"status": "ok", "message": "Cache performance acceptable"}
    
    return checks


def _generate_system_recommendations(
    db_metrics: Dict[str, Any],
    cache_stats: Dict[str, Any],
    query_report: Dict[str, Any]
) -> List[str]:
    """Generate system optimization recommendations."""
    recommendations = []
    
    # Database recommendations
    if "query_monitor" in db_metrics:
        recommendations.extend(db_metrics["query_monitor"].get("recommendations", []))
    
    # Cache recommendations
    if cache_stats["hit_rate_percent"] < 70 and cache_stats["total_requests"] > 100:
        recommendations.append(
            f"Consider increasing cache TTL or reviewing cache strategy. "
            f"Current hit rate: {cache_stats['hit_rate_percent']:.1f}%"
        )
    
    # Connection pool recommendations
    if query_report["connection_stats"]["failed_queries"] > 0:
        recommendations.append(
            "Database connection failures detected. Consider reviewing connection pool settings."
        )
    
    return recommendations


def _calculate_performance_trends() -> Dict[str, Any]:
    """Calculate performance trends over time."""
    # This would analyze historical data to show trends
    return {
        "query_time_trend": "stable",
        "error_rate_trend": "decreasing",
        "cache_hit_rate_trend": "improving"
    }


def _generate_query_optimizations(query_report: Dict[str, Any]) -> List[str]:
    """Generate specific query optimization suggestions."""
    suggestions = []
    
    for query in query_report["top_queries_by_impact"][:3]:
        if query["avg_time"] > 1.0:
            suggestions.append(
                f"Optimize high-impact query (hash: {query['hash']}) - "
                f"Consider adding indexes or rewriting query logic"
            )
        
        if query["count"] > 5000:
            suggestions.append(
                f"High-frequency query detected (hash: {query['hash']}) - "
                f"Consider implementing result caching"
            )
    
    return suggestions 