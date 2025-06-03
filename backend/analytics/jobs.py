"""
Analytics Background Jobs
Handles background processing of analytics calculations, trend analysis, and alert generation.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List
from celery import Celery
import asyncio

from database.connection import get_async_session
from .service import AnalyticsService
from .models import MetricDefinition, Alert, AlertStatus
from services.cache_service import cache_service
from config.performance import BACKGROUND_JOB_SETTINGS

logger = logging.getLogger(__name__)

# Initialize Celery
celery_app = Celery(
    'analytics',
    broker='redis://localhost:6379/1',  # Using Redis as broker
    backend='redis://localhost:6379/1'
)

# Configure Celery
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    worker_prefetch_multiplier=BACKGROUND_JOB_SETTINGS['worker_prefetch_multiplier'],
    task_time_limit=BACKGROUND_JOB_SETTINGS['task_time_limit'],
    task_soft_time_limit=BACKGROUND_JOB_SETTINGS['task_soft_time_limit'],
    task_routes=BACKGROUND_JOB_SETTINGS['task_routes']
)


@celery_app.task(bind=True, name='analytics.calculate_team_metrics')
def calculate_team_metrics_task(self, team_id: str):
    """Background task to calculate metrics for a team."""
    try:
        logger.info(f"Starting metrics calculation for team {team_id}")
        
        # Run async function in sync context
        result = asyncio.run(calculate_team_metrics_async(team_id))
        
        logger.info(f"Completed metrics calculation for team {team_id}: {result['metrics_calculated']} metrics")
        return result
        
    except Exception as e:
        logger.error(f"Error calculating metrics for team {team_id}: {e}")
        self.retry(countdown=60, max_retries=3)


@celery_app.task(bind=True, name='analytics.analyze_team_trends')
def analyze_team_trends_task(self, team_id: str):
    """Background task to analyze trends for a team."""
    try:
        logger.info(f"Starting trend analysis for team {team_id}")
        
        result = asyncio.run(analyze_team_trends_async(team_id))
        
        logger.info(f"Completed trend analysis for team {team_id}: {len(result)} trends analyzed")
        return result
        
    except Exception as e:
        logger.error(f"Error analyzing trends for team {team_id}: {e}")
        self.retry(countdown=120, max_retries=3)


@celery_app.task(bind=True, name='analytics.process_alerts')
def process_alerts_task(self, team_id: str = None):
    """Background task to process and send alerts."""
    try:
        logger.info(f"Starting alert processing{' for team ' + team_id if team_id else ''}")
        
        result = asyncio.run(process_alerts_async(team_id))
        
        logger.info(f"Completed alert processing: {result['alerts_processed']} alerts processed")
        return result
        
    except Exception as e:
        logger.error(f"Error processing alerts: {e}")
        self.retry(countdown=30, max_retries=5)


@celery_app.task(bind=True, name='analytics.cleanup_old_data')
def cleanup_old_data_task(self, days_to_keep: int = 90):
    """Background task to clean up old analytics data."""
    try:
        logger.info(f"Starting cleanup of analytics data older than {days_to_keep} days")
        
        result = asyncio.run(cleanup_old_data_async(days_to_keep))
        
        logger.info(f"Completed data cleanup: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Error cleaning up old data: {e}")
        self.retry(countdown=300, max_retries=2)


@celery_app.task(bind=True, name='analytics.generate_daily_reports')
def generate_daily_reports_task(self, team_id: str = None):
    """Background task to generate daily analytics reports."""
    try:
        logger.info(f"Starting daily report generation{' for team ' + team_id if team_id else ''}")
        
        result = asyncio.run(generate_daily_reports_async(team_id))
        
        logger.info(f"Completed daily report generation: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Error generating daily reports: {e}")
        self.retry(countdown=600, max_retries=2)


# Async helper functions

async def calculate_team_metrics_async(team_id: str) -> Dict[str, Any]:
    """Calculate metrics for a team asynchronously."""
    async with get_async_session() as session:
        analytics_service = AnalyticsService(session)
        result = await analytics_service.calculate_and_store_metrics(team_id)
        
        # Invalidate relevant caches
        await cache_service.delete_pattern(f"*{team_id}*", prefix="analytics")
        await cache_service.delete_pattern(f"*{team_id}*", prefix="dashboard")
        
        return result


async def analyze_team_trends_async(team_id: str) -> List[Dict[str, Any]]:
    """Analyze trends for a team asynchronously."""
    async with get_async_session() as session:
        analytics_service = AnalyticsService(session)
        trends = await analytics_service.analyze_trends_for_team(team_id)
        
        # Cache trend results
        cache_key = f"trends:{team_id}"
        await cache_service.set(cache_key, trends, ttl=3600, prefix="analytics")  # Cache for 1 hour
        
        return trends


async def process_alerts_async(team_id: str = None) -> Dict[str, Any]:
    """Process alerts asynchronously."""
    from sqlalchemy import select, and_
    
    alerts_processed = 0
    notifications_sent = 0
    
    async with get_async_session() as session:
        # Get active alerts
        query = select(Alert).where(Alert.status == AlertStatus.ACTIVE)
        if team_id:
            query = query.where(Alert.team_id == team_id)
        
        result = await session.execute(query)
        alerts = result.scalars().all()
        
        for alert in alerts:
            try:
                # Process alert (send notifications, escalate, etc.)
                await process_single_alert(alert, session)
                alerts_processed += 1
                
                # Track notifications sent
                if alert.notifications_sent:
                    notifications_sent += len(alert.notifications_sent.get('channels', []))
                
            except Exception as e:
                logger.error(f"Error processing alert {alert.id}: {e}")
        
        await session.commit()
    
    return {
        'alerts_processed': alerts_processed,
        'notifications_sent': notifications_sent,
        'processed_at': datetime.utcnow().isoformat()
    }


async def process_single_alert(alert: Alert, session):
    """Process a single alert (notifications, escalation, etc.)."""
    # This is where you would integrate with notification services
    # For now, we'll just log and mark as processed
    
    logger.info(f"Processing alert: {alert.title} (Severity: {alert.severity})")
    
    # Simulate notification sending
    notifications_sent = {
        'channels': ['email', 'slack'],  # Example channels
        'sent_at': datetime.utcnow().isoformat(),
        'recipients': ['team@example.com']  # Example recipients
    }
    
    alert.notifications_sent = notifications_sent
    
    # Auto-resolve alerts that are older than 24 hours for non-critical issues
    if (alert.severity != 'critical' and 
        alert.triggered_at < datetime.utcnow() - timedelta(hours=24)):
        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = datetime.utcnow()
        alert.resolution_notes = "Auto-resolved after 24 hours"


async def cleanup_old_data_async(days_to_keep: int) -> Dict[str, Any]:
    """Clean up old analytics data."""
    from sqlalchemy import delete
    from .models import MetricValue, TrendAnalysis
    
    cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
    
    async with get_async_session() as session:
        # Clean up old metric values
        metric_values_deleted = await session.execute(
            delete(MetricValue).where(MetricValue.timestamp < cutoff_date)
        )
        
        # Clean up old trend analyses
        trend_analyses_deleted = await session.execute(
            delete(TrendAnalysis).where(TrendAnalysis.analysis_timestamp < cutoff_date)
        )
        
        # Clean up resolved alerts older than retention period
        old_alerts_deleted = await session.execute(
            delete(Alert).where(
                and_(
                    Alert.resolved_at < cutoff_date,
                    Alert.status == AlertStatus.RESOLVED
                )
            )
        )
        
        await session.commit()
        
        return {
            'metric_values_deleted': metric_values_deleted.rowcount,
            'trend_analyses_deleted': trend_analyses_deleted.rowcount,
            'old_alerts_deleted': old_alerts_deleted.rowcount,
            'cutoff_date': cutoff_date.isoformat()
        }


async def generate_daily_reports_async(team_id: str = None) -> Dict[str, Any]:
    """Generate daily analytics reports."""
    reports_generated = 0
    
    if team_id:
        # Generate report for specific team
        report = await generate_team_daily_report(team_id)
        if report:
            reports_generated = 1
    else:
        # Generate reports for all teams
        from sqlalchemy import select
        from database.models import Team
        
        async with get_async_session() as session:
            teams_query = select(Team.id).where(Team.is_active == True)
            result = await session.execute(teams_query)
            team_ids = [row[0] for row in result.all()]
            
            for tid in team_ids:
                try:
                    report = await generate_team_daily_report(tid)
                    if report:
                        reports_generated += 1
                except Exception as e:
                    logger.error(f"Error generating daily report for team {tid}: {e}")
    
    return {
        'reports_generated': reports_generated,
        'generated_at': datetime.utcnow().isoformat()
    }


async def generate_team_daily_report(team_id: str) -> Dict[str, Any]:
    """Generate daily report for a specific team."""
    async with get_async_session() as session:
        analytics_service = AnalyticsService(session)
        
        # Get yesterday's data
        end_time = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        start_time = end_time - timedelta(days=1)
        time_range = {'start': start_time, 'end': end_time}
        
        # Generate comprehensive report
        report = await analytics_service.get_dashboard_overview(team_id, time_range)
        
        # Cache the daily report
        cache_key = f"daily_report:{team_id}:{start_time.strftime('%Y-%m-%d')}"
        await cache_service.set(cache_key, report, ttl=86400, prefix="analytics")  # Cache for 24 hours
        
        logger.info(f"Generated daily report for team {team_id}")
        return report


# Periodic task scheduler
@celery_app.task(bind=True)
def setup_periodic_tasks(sender, **kwargs):
    """Set up periodic analytics tasks."""
    from celery.schedules import crontab
    
    # Schedule metrics calculation every 5 minutes
    sender.add_periodic_task(
        300.0,  # 5 minutes
        calculate_all_teams_metrics.s(),
        name='calculate metrics every 5 minutes'
    )
    
    # Schedule trend analysis every hour
    sender.add_periodic_task(
        crontab(minute=0),  # Every hour
        analyze_all_teams_trends.s(),
        name='analyze trends hourly'
    )
    
    # Schedule alert processing every 2 minutes
    sender.add_periodic_task(
        120.0,  # 2 minutes
        process_alerts_task.s(),
        name='process alerts every 2 minutes'
    )
    
    # Schedule daily reports at 1 AM
    sender.add_periodic_task(
        crontab(hour=1, minute=0),
        generate_daily_reports_task.s(),
        name='generate daily reports at 1 AM'
    )
    
    # Schedule weekly cleanup on Sundays at 2 AM
    sender.add_periodic_task(
        crontab(hour=2, minute=0, day_of_week=0),
        cleanup_old_data_task.s(),
        name='cleanup old data weekly'
    )


@celery_app.task(name='analytics.calculate_all_teams_metrics')
def calculate_all_teams_metrics():
    """Calculate metrics for all active teams."""
    return asyncio.run(calculate_all_teams_metrics_async())


@celery_app.task(name='analytics.analyze_all_teams_trends')
def analyze_all_teams_trends():
    """Analyze trends for all active teams."""
    return asyncio.run(analyze_all_teams_trends_async())


async def calculate_all_teams_metrics_async():
    """Calculate metrics for all teams asynchronously."""
    from sqlalchemy import select
    from database.models import Team
    
    teams_processed = 0
    
    async with get_async_session() as session:
        teams_query = select(Team.id).where(Team.is_active == True)
        result = await session.execute(teams_query)
        team_ids = [row[0] for row in result.all()]
        
        for team_id in team_ids:
            try:
                # Schedule individual team metric calculation
                calculate_team_metrics_task.delay(str(team_id))
                teams_processed += 1
            except Exception as e:
                logger.error(f"Error scheduling metrics calculation for team {team_id}: {e}")
    
    return {'teams_processed': teams_processed}


async def analyze_all_teams_trends_async():
    """Analyze trends for all teams asynchronously."""
    from sqlalchemy import select
    from database.models import Team
    
    teams_processed = 0
    
    async with get_async_session() as session:
        teams_query = select(Team.id).where(Team.is_active == True)
        result = await session.execute(teams_query)
        team_ids = [row[0] for row in result.all()]
        
        for team_id in team_ids:
            try:
                # Schedule individual team trend analysis
                analyze_team_trends_task.delay(str(team_id))
                teams_processed += 1
            except Exception as e:
                logger.error(f"Error scheduling trend analysis for team {team_id}: {e}")
    
    return {'teams_processed': teams_processed} 