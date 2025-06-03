"""
Analytics API
Provides endpoints for real-time metrics, trend analysis, alerts, and dashboard analytics.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from pydantic import BaseModel, Field

from database.connection import get_async_session
from sqlalchemy.ext.asyncio import AsyncSession
from analytics.service import AnalyticsService
from analytics.models import (
    MetricDefinitionCreate, MetricDefinitionResponse, MetricValueResponse,
    TrendAnalysisResponse, AlertCreate, AlertResponse, DashboardMetricsResponse,
    MetricDefinition, MetricValue, TrendAnalysis, Alert
)
from analytics.jobs import calculate_team_metrics_task, analyze_team_trends_task
from auth.security import require_role, get_current_user
from auth.models import UserRole
from database.models import User
from services.cache_service import cache_service

router = APIRouter(prefix="/analytics", tags=["Analytics"])


class DateRangeQuery(BaseModel):
    """Query model for date range filtering."""
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None


class MetricsCalculationResponse(BaseModel):
    """Response model for metrics calculation."""
    job_id: str
    message: str
    estimated_completion_seconds: int


# Dashboard Analytics Endpoints

@router.get("/dashboard", response_model=DashboardMetricsResponse)
async def get_dashboard_analytics(
    start_date: Optional[datetime] = Query(None, description="Start date for analytics"),
    end_date: Optional[datetime] = Query(None, description="End date for analytics"),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Get comprehensive dashboard analytics for the user's team.
    Includes metrics, trends, and alerts with intelligent caching.
    """
    try:
        analytics_service = AnalyticsService(session)
        
        # Set default time range if not provided
        time_range = None
        if start_date or end_date:
            time_range = {
                'start': start_date or (datetime.utcnow() - timedelta(days=7)),
                'end': end_date or datetime.utcnow()
            }
        
        # Get dashboard overview with caching
        dashboard_data = await analytics_service.get_dashboard_overview(
            team_id=str(current_user.default_team_id), 
            time_range=time_range
        )
        
        return dashboard_data
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving dashboard analytics: {str(e)}"
        )


@router.get("/dashboard/realtime")
async def get_realtime_dashboard(
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Get real-time dashboard analytics (last 1 hour).
    Optimized for frequent polling with aggressive caching.
    """
    try:
        # Check cache first for real-time data
        cache_key = f"realtime_dashboard:{current_user.default_team_id}"
        cached_data = await cache_service.get(cache_key, prefix="analytics")
        
        if cached_data:
            return cached_data
        
        analytics_service = AnalyticsService(session)
        
        # Get last hour data
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=1)
        time_range = {'start': start_time, 'end': end_time}
        
        dashboard_data = await analytics_service.get_dashboard_overview(
            team_id=str(current_user.default_team_id),
            time_range=time_range
        )
        
        # Cache for 1 minute (aggressive caching for real-time data)
        await cache_service.set(cache_key, dashboard_data, ttl=60, prefix="analytics")
        
        return dashboard_data
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving real-time dashboard: {str(e)}"
        )


# Metrics Definition Management

@router.get("/metrics/definitions", response_model=List[MetricDefinitionResponse])
async def get_metric_definitions(
    metric_type: Optional[str] = Query(None, description="Filter by metric type"),
    is_active: bool = Query(True, description="Filter by active status"),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Get all metric definitions for the user's team."""
    try:
        from sqlalchemy import select, and_
        
        query = select(MetricDefinition).where(
            MetricDefinition.team_id == current_user.default_team_id
        )
        
        if metric_type:
            query = query.where(MetricDefinition.metric_type == metric_type)
        
        if is_active is not None:
            query = query.where(MetricDefinition.is_active == is_active)
        
        query = query.order_by(MetricDefinition.display_order, MetricDefinition.name)
        
        result = await session.execute(query)
        metrics = result.scalars().all()
        
        return [MetricDefinitionResponse.model_validate(metric) for metric in metrics]
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving metric definitions: {str(e)}"
        )


@router.post("/metrics/definitions", response_model=MetricDefinitionResponse)
async def create_metric_definition(
    metric_data: MetricDefinitionCreate,
    current_user: User = Depends(require_role([UserRole.TEAM_ADMIN, UserRole.SUPER_ADMIN])),
    session: AsyncSession = Depends(get_async_session)
):
    """Create a new metric definition."""
    try:
        metric = MetricDefinition(
            team_id=current_user.default_team_id,
            created_by=current_user.id,
            **metric_data.model_dump()
        )
        
        session.add(metric)
        await session.commit()
        await session.refresh(metric)
        
        # Invalidate related caches
        await cache_service.delete_pattern(f"*{current_user.default_team_id}*", prefix="analytics")
        
        return MetricDefinitionResponse.model_validate(metric)
        
    except Exception as e:
        await session.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Error creating metric definition: {str(e)}"
        )


@router.get("/metrics/values/{metric_id}", response_model=List[MetricValueResponse])
async def get_metric_values(
    metric_id: str,
    start_date: Optional[datetime] = Query(None, description="Start date for values"),
    end_date: Optional[datetime] = Query(None, description="End date for values"),
    limit: int = Query(1000, le=5000, description="Maximum number of values to return"),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Get metric values for a specific metric over time."""
    try:
        from sqlalchemy import select, and_, desc
        
        # Verify metric belongs to user's team
        metric_query = select(MetricDefinition).where(
            and_(
                MetricDefinition.id == metric_id,
                MetricDefinition.team_id == current_user.default_team_id
            )
        )
        
        metric_result = await session.execute(metric_query)
        metric = metric_result.scalar_one_or_none()
        
        if not metric:
            raise HTTPException(status_code=404, detail="Metric definition not found")
        
        # Build query for metric values
        values_query = select(MetricValue).where(
            MetricValue.metric_definition_id == metric_id
        )
        
        if start_date:
            values_query = values_query.where(MetricValue.timestamp >= start_date)
        if end_date:
            values_query = values_query.where(MetricValue.timestamp <= end_date)
        
        values_query = values_query.order_by(desc(MetricValue.timestamp)).limit(limit)
        
        result = await session.execute(values_query)
        values = result.scalars().all()
        
        return [MetricValueResponse.model_validate(value) for value in values]
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving metric values: {str(e)}"
        )


# Trend Analysis Endpoints

@router.get("/trends", response_model=List[TrendAnalysisResponse])
async def get_trend_analyses(
    metric_id: Optional[str] = Query(None, description="Filter by metric ID"),
    days_back: int = Query(30, ge=1, le=365, description="Number of days to analyze"),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Get trend analyses for metrics."""
    try:
        from sqlalchemy import select, and_, desc
        
        query = select(TrendAnalysis).where(
            TrendAnalysis.team_id == current_user.default_team_id
        )
        
        if metric_id:
            query = query.where(TrendAnalysis.metric_definition_id == metric_id)
        
        # Filter by analysis recency
        cutoff_date = datetime.utcnow() - timedelta(days=days_back)
        query = query.where(TrendAnalysis.analysis_timestamp >= cutoff_date)
        
        query = query.order_by(desc(TrendAnalysis.analysis_timestamp))
        
        result = await session.execute(query)
        trends = result.scalars().all()
        
        return [TrendAnalysisResponse.model_validate(trend) for trend in trends]
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving trend analyses: {str(e)}"
        )


@router.post("/trends/analyze")
async def trigger_trend_analysis(
    background_tasks: BackgroundTasks,
    metric_id: Optional[str] = Query(None, description="Specific metric to analyze"),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Trigger trend analysis for team metrics."""
    try:
        if metric_id:
            # Verify metric belongs to user's team
            from sqlalchemy import select, and_
            
            metric_query = select(MetricDefinition).where(
                and_(
                    MetricDefinition.id == metric_id,
                    MetricDefinition.team_id == current_user.default_team_id
                )
            )
            
            metric_result = await session.execute(metric_query)
            metric = metric_result.scalar_one_or_none()
            
            if not metric:
                raise HTTPException(status_code=404, detail="Metric definition not found")
        
        # Schedule trend analysis job
        job = analyze_team_trends_task.delay(str(current_user.default_team_id))
        
        return {
            "job_id": job.id,
            "message": "Trend analysis scheduled",
            "estimated_completion_seconds": 60
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error scheduling trend analysis: {str(e)}"
        )


# Alerts Management

@router.get("/alerts", response_model=List[AlertResponse])
async def get_alerts(
    status: Optional[str] = Query(None, description="Filter by alert status"),
    severity: Optional[str] = Query(None, description="Filter by alert severity"),
    limit: int = Query(100, le=500, description="Maximum number of alerts to return"),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Get alerts for the user's team."""
    try:
        from sqlalchemy import select, and_, desc
        
        query = select(Alert).where(Alert.team_id == current_user.default_team_id)
        
        if status:
            query = query.where(Alert.status == status)
        if severity:
            query = query.where(Alert.severity == severity)
        
        query = query.order_by(desc(Alert.triggered_at)).limit(limit)
        
        result = await session.execute(query)
        alerts = result.scalars().all()
        
        return [AlertResponse.model_validate(alert) for alert in alerts]
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving alerts: {str(e)}"
        )


@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: str,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Acknowledge an active alert."""
    try:
        from sqlalchemy import select, and_
        from analytics.models import AlertStatus
        
        query = select(Alert).where(
            and_(
                Alert.id == alert_id,
                Alert.team_id == current_user.default_team_id
            )
        )
        
        result = await session.execute(query)
        alert = result.scalar_one_or_none()
        
        if not alert:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        if alert.status != AlertStatus.ACTIVE:
            raise HTTPException(status_code=400, detail="Alert is not active")
        
        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_at = datetime.utcnow()
        alert.acknowledged_by = current_user.id
        
        await session.commit()
        
        return {"message": "Alert acknowledged successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Error acknowledging alert: {str(e)}"
        )


@router.post("/alerts/{alert_id}/resolve")
async def resolve_alert(
    alert_id: str,
    resolution_notes: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Resolve an alert."""
    try:
        analytics_service = AnalyticsService(session)
        
        success = await analytics_service.alert_manager.resolve_alert(
            alert_id, str(current_user.id), resolution_notes
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Alert not found or cannot be resolved")
        
        return {"message": "Alert resolved successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error resolving alert: {str(e)}"
        )


# Metrics Calculation Endpoints

@router.post("/metrics/calculate", response_model=MetricsCalculationResponse)
async def trigger_metrics_calculation(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """Trigger metrics calculation for the user's team."""
    try:
        # Schedule metrics calculation job
        job = calculate_team_metrics_task.delay(str(current_user.default_team_id))
        
        return MetricsCalculationResponse(
            job_id=job.id,
            message="Metrics calculation scheduled",
            estimated_completion_seconds=30
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error scheduling metrics calculation: {str(e)}"
        )


@router.get("/metrics/job/{job_id}")
async def get_metrics_job_status(
    job_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get the status of a metrics calculation job."""
    try:
        from analytics.jobs import celery_app
        
        result = celery_app.AsyncResult(job_id)
        
        return {
            "job_id": job_id,
            "status": result.status,
            "result": result.result if result.ready() else None,
            "error": str(result.info) if result.failed() else None
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving job status: {str(e)}"
        )


# Advanced Analytics Endpoints

@router.get("/metrics/summary")
async def get_metrics_summary(
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Get a summary of all metrics for the team."""
    try:
        analytics_service = AnalyticsService(session)
        
        # Get cached summary if available
        cache_key = f"metrics_summary:{current_user.default_team_id}"
        cached_summary = await cache_service.get(cache_key, prefix="analytics")
        
        if cached_summary:
            return cached_summary
        
        # Calculate summary
        summary = await analytics_service.get_dashboard_overview(
            team_id=str(current_user.default_team_id)
        )
        
        # Extract key metrics for summary
        metrics_summary = {
            "total_requests": summary.get('throughput_metrics', {}).get('total_requests', 0),
            "avg_latency_ms": summary.get('latency_metrics', {}).get('avg_latency_ms', 0),
            "error_rate": summary.get('error_metrics', {}).get('error_rate', 0),
            "total_cost_usd": summary.get('cost_metrics', {}).get('total_cost_usd', 0),
            "avg_quality_score": summary.get('quality_metrics', {}).get('avg_evaluation_score', 0),
            "active_alerts": len(summary.get('active_alerts', [])),
            "last_updated": summary.get('generated_at')
        }
        
        # Cache for 5 minutes
        await cache_service.set(cache_key, metrics_summary, ttl=300, prefix="analytics")
        
        return metrics_summary
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving metrics summary: {str(e)}"
        )


@router.get("/health")
async def analytics_health_check():
    """Health check for analytics service."""
    try:
        # Check Redis connection
        redis_healthy = await cache_service.exists("health_check", prefix="system")
        
        # Check if background jobs are working
        from analytics.jobs import celery_app
        job_queue_healthy = celery_app.control.active() is not None
        
        status = "healthy" if (redis_healthy and job_queue_healthy) else "degraded"
        
        return {
            "status": status,
            "redis_connection": redis_healthy,
            "job_queue": job_queue_healthy,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        } 