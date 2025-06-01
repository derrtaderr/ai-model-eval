"""
Performance Monitoring and Analytics Service
Comprehensive dashboard and reporting for model-based evaluation engine.
Part of Task 6.5 - Develop Reporting and Analytics Dashboard.
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, Counter
import statistics

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_, desc, asc
from sqlalchemy.orm import selectinload

from database.models import Trace, Evaluation, User
from database.connection import get_db
from config.settings import get_settings

# Import batch processing for monitoring
try:
    from services.batch_evaluation import batch_processor, BatchStatus
    BATCH_MONITORING_AVAILABLE = True
except ImportError:
    batch_processor = None
    BATCH_MONITORING_AVAILABLE = False

# Import calibration system for performance metrics
try:
    from services.scoring_calibration import calibration_system
    CALIBRATION_MONITORING_AVAILABLE = True
except ImportError:
    calibration_system = None
    CALIBRATION_MONITORING_AVAILABLE = False

logger = logging.getLogger(__name__)

class MetricType(str, Enum):
    """Types of performance metrics."""
    THROUGHPUT = "throughput"
    ACCURACY = "accuracy"
    COST = "cost"
    LATENCY = "latency"
    SUCCESS_RATE = "success_rate"
    CALIBRATION_PERFORMANCE = "calibration_performance"

class TimeRange(str, Enum):
    """Time range options for analytics."""
    HOUR = "1h"
    DAY = "24h"
    WEEK = "7d"
    MONTH = "30d"
    QUARTER = "90d"
    YEAR = "365d"

class AlertLevel(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class PerformanceMetric:
    """Individual performance metric."""
    metric_type: MetricType
    value: float
    unit: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TrendData:
    """Time series data for trend analysis."""
    timestamps: List[str]
    values: List[float]
    labels: List[str] = field(default_factory=list)
    
@dataclass
class AnalyticsReport:
    """Comprehensive analytics report."""
    report_id: str
    title: str
    time_range: TimeRange
    generated_at: str
    metrics: Dict[str, PerformanceMetric]
    trends: Dict[str, TrendData]
    insights: List[str]
    recommendations: List[str]
    raw_data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ModelPerformanceComparison:
    """Performance comparison between evaluator models."""
    model_name: str
    total_evaluations: int
    success_rate: float
    average_latency_ms: float
    total_cost_usd: float
    average_score: float
    score_std_dev: float
    cost_per_evaluation: float
    throughput_per_hour: float
    error_rate: float
    calibration_accuracy: Optional[float] = None

@dataclass
class SystemAlert:
    """System performance alert."""
    id: str
    level: AlertLevel
    title: str
    message: str
    metric_type: MetricType
    threshold_value: float
    current_value: float
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    resolved: bool = False

class PerformanceAnalytics:
    """Comprehensive performance monitoring and analytics service."""
    
    def __init__(self):
        self.settings = get_settings()
        
        # Alert thresholds
        self.alert_thresholds = {
            MetricType.SUCCESS_RATE: {"warning": 0.95, "error": 0.90, "critical": 0.80},
            MetricType.THROUGHPUT: {"warning": 10.0, "error": 5.0, "critical": 1.0},  # per hour
            MetricType.COST: {"warning": 100.0, "error": 500.0, "critical": 1000.0},  # per day
            MetricType.LATENCY: {"warning": 5000, "error": 10000, "critical": 30000},  # ms
        }
        
        # Cache for performance data
        self._metrics_cache = {}
        self._cache_expiry = {}
        self._cache_duration_minutes = 5
        
        logger.info("PerformanceAnalytics service initialized")
    
    async def get_system_overview(self, time_range: TimeRange = TimeRange.DAY) -> Dict[str, Any]:
        """Get comprehensive system performance overview."""
        end_time = datetime.utcnow()
        start_time = self._get_start_time(end_time, time_range)
        
        async with get_db() as session:
            # Get basic metrics
            overview = {
                "time_range": time_range.value,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "evaluation_metrics": await self._get_evaluation_metrics(session, start_time, end_time),
                "batch_metrics": await self._get_batch_metrics(start_time, end_time),
                "cost_metrics": await self._get_cost_metrics(session, start_time, end_time),
                "performance_trends": await self._get_performance_trends(session, start_time, end_time),
                "model_comparison": await self._get_model_comparison(session, start_time, end_time),
                "system_health": await self._get_system_health(),
                "alerts": await self._get_active_alerts(),
                "calibration_metrics": await self._get_calibration_metrics(start_time, end_time)
            }
            
            return overview
    
    async def _get_evaluation_metrics(self, session: AsyncSession, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Get evaluation performance metrics."""
        # Total evaluations
        total_query = select(func.count(Evaluation.id)).where(
            and_(
                Evaluation.evaluated_at >= start_time,
                Evaluation.evaluated_at <= end_time,
                Evaluation.evaluator_type == "ai_model"
            )
        )
        total_result = await session.execute(total_query)
        total_evaluations = total_result.scalar() or 0
        
        # Success rate (non-null scores)
        success_query = select(func.count(Evaluation.id)).where(
            and_(
                Evaluation.evaluated_at >= start_time,
                Evaluation.evaluated_at <= end_time,
                Evaluation.evaluator_type == "ai_model",
                Evaluation.score.isnot(None)
            )
        )
        success_result = await session.execute(success_query)
        successful_evaluations = success_result.scalar() or 0
        
        success_rate = (successful_evaluations / total_evaluations) if total_evaluations > 0 else 0
        
        # Average score
        avg_score_query = select(func.avg(Evaluation.score)).where(
            and_(
                Evaluation.evaluated_at >= start_time,
                Evaluation.evaluated_at <= end_time,
                Evaluation.evaluator_type == "ai_model",
                Evaluation.score.isnot(None)
            )
        )
        avg_score_result = await session.execute(avg_score_query)
        average_score = float(avg_score_result.scalar() or 0)
        
        # Score distribution
        score_distribution = await self._get_score_distribution(session, start_time, end_time)
        
        # Throughput (evaluations per hour)
        duration_hours = (end_time - start_time).total_seconds() / 3600
        throughput = total_evaluations / duration_hours if duration_hours > 0 else 0
        
        return {
            "total_evaluations": total_evaluations,
            "successful_evaluations": successful_evaluations,
            "failed_evaluations": total_evaluations - successful_evaluations,
            "success_rate": success_rate,
            "average_score": average_score,
            "throughput_per_hour": throughput,
            "score_distribution": score_distribution
        }
    
    async def _get_score_distribution(self, session: AsyncSession, start_time: datetime, end_time: datetime) -> Dict[str, int]:
        """Get distribution of evaluation scores."""
        scores_query = select(Evaluation.score).where(
            and_(
                Evaluation.evaluated_at >= start_time,
                Evaluation.evaluated_at <= end_time,
                Evaluation.evaluator_type == "ai_model",
                Evaluation.score.isnot(None)
            )
        )
        scores_result = await session.execute(scores_query)
        scores = [float(score[0]) for score in scores_result.fetchall()]
        
        # Create score buckets
        distribution = {
            "0.0-0.2": 0,
            "0.2-0.4": 0,
            "0.4-0.6": 0,
            "0.6-0.8": 0,
            "0.8-1.0": 0
        }
        
        for score in scores:
            if score < 0.2:
                distribution["0.0-0.2"] += 1
            elif score < 0.4:
                distribution["0.2-0.4"] += 1
            elif score < 0.6:
                distribution["0.4-0.6"] += 1
            elif score < 0.8:
                distribution["0.6-0.8"] += 1
            else:
                distribution["0.8-1.0"] += 1
        
        return distribution
    
    async def _get_batch_metrics(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Get batch processing performance metrics."""
        if not BATCH_MONITORING_AVAILABLE or not batch_processor:
            return {
                "total_jobs": 0,
                "active_jobs": 0,
                "completed_jobs": 0,
                "failed_jobs": 0,
                "average_completion_time_minutes": 0,
                "total_tasks_processed": 0,
                "system_utilization": 0
            }
        
        # Get batch processor statistics
        system_stats = await batch_processor.get_system_stats()
        
        # Filter jobs by time range
        jobs = await batch_processor.list_batch_jobs()
        time_filtered_jobs = [
            job for job in jobs 
            if start_time <= datetime.fromisoformat(job.created_at.replace('Z', '+00:00').replace('+00:00', '')) <= end_time
        ]
        
        completed_jobs = [job for job in time_filtered_jobs if job.status == BatchStatus.COMPLETED]
        failed_jobs = [job for job in time_filtered_jobs if job.status == BatchStatus.FAILED]
        
        # Calculate average completion time
        completion_times = []
        for job in completed_jobs:
            if job.started_at and job.completed_at:
                start = datetime.fromisoformat(job.started_at.replace('Z', '+00:00').replace('+00:00', ''))
                end = datetime.fromisoformat(job.completed_at.replace('Z', '+00:00').replace('+00:00', ''))
                completion_times.append((end - start).total_seconds() / 60)  # minutes
        
        avg_completion_time = statistics.mean(completion_times) if completion_times else 0
        
        return {
            "total_jobs": len(time_filtered_jobs),
            "active_jobs": system_stats["running_jobs"],
            "completed_jobs": len(completed_jobs),
            "failed_jobs": len(failed_jobs),
            "average_completion_time_minutes": avg_completion_time,
            "total_tasks_processed": system_stats["total_processed"],
            "system_utilization": system_stats["total_workers"] / system_stats["max_workers"] if system_stats["max_workers"] > 0 else 0,
            "success_rate": system_stats["success_rate"],
            "throughput_per_hour": system_stats["throughput_per_hour"]
        }
    
    async def _get_cost_metrics(self, session: AsyncSession, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Get cost analysis metrics."""
        # Get evaluation costs from metadata
        cost_query = select(Evaluation.eval_metadata).where(
            and_(
                Evaluation.evaluated_at >= start_time,
                Evaluation.evaluated_at <= end_time,
                Evaluation.evaluator_type == "ai_model",
                Evaluation.eval_metadata.isnot(None)
            )
        )
        cost_result = await session.execute(cost_query)
        
        costs = []
        for metadata_row in cost_result.fetchall():
            metadata = metadata_row[0]
            if isinstance(metadata, dict) and 'cost_usd' in metadata:
                cost = metadata.get('cost_usd')
                if cost is not None:
                    costs.append(float(cost))
        
        total_cost = sum(costs)
        average_cost = statistics.mean(costs) if costs else 0
        
        # Cost by model
        model_costs = await self._get_cost_by_model(session, start_time, end_time)
        
        # Project monthly cost
        duration_days = (end_time - start_time).days or 1
        monthly_projection = (total_cost / duration_days) * 30
        
        return {
            "total_cost_usd": total_cost,
            "average_cost_per_evaluation": average_cost,
            "total_evaluations": len(costs),
            "cost_by_model": model_costs,
            "monthly_projection_usd": monthly_projection,
            "cost_trend": await self._get_cost_trend(session, start_time, end_time)
        }
    
    async def _get_cost_by_model(self, session: AsyncSession, start_time: datetime, end_time: datetime) -> Dict[str, float]:
        """Get cost breakdown by evaluator model."""
        costs_query = select(Evaluation.eval_metadata).where(
            and_(
                Evaluation.evaluated_at >= start_time,
                Evaluation.evaluated_at <= end_time,
                Evaluation.evaluator_type == "ai_model",
                Evaluation.eval_metadata.isnot(None)
            )
        )
        costs_result = await session.execute(costs_query)
        
        model_costs = defaultdict(float)
        for metadata_row in costs_result.fetchall():
            metadata = metadata_row[0]
            if isinstance(metadata, dict):
                model = metadata.get('evaluator_model', 'unknown')
                cost = metadata.get('cost_usd', 0)
                if cost is not None:
                    model_costs[model] += float(cost)
        
        return dict(model_costs)
    
    async def _get_cost_trend(self, session: AsyncSession, start_time: datetime, end_time: datetime) -> TrendData:
        """Get cost trend over time."""
        # Get hourly cost data
        costs_query = select(
            func.date_trunc('hour', Evaluation.evaluated_at).label('hour'),
            Evaluation.eval_metadata
        ).where(
            and_(
                Evaluation.evaluated_at >= start_time,
                Evaluation.evaluated_at <= end_time,
                Evaluation.evaluator_type == "ai_model",
                Evaluation.eval_metadata.isnot(None)
            )
        ).order_by('hour')
        
        costs_result = await session.execute(costs_query)
        
        hourly_costs = defaultdict(float)
        for hour, metadata in costs_result.fetchall():
            if isinstance(metadata, dict) and 'cost_usd' in metadata:
                cost = metadata.get('cost_usd')
                if cost is not None:
                    hourly_costs[hour.isoformat()] += float(cost)
        
        timestamps = list(hourly_costs.keys())
        values = list(hourly_costs.values())
        
        return TrendData(
            timestamps=timestamps,
            values=values,
            labels=[f"${v:.4f}" for v in values]
        )
    
    async def _get_performance_trends(self, session: AsyncSession, start_time: datetime, end_time: datetime) -> Dict[str, TrendData]:
        """Get performance trends over time."""
        # Hourly evaluation counts
        counts_query = select(
            func.date_trunc('hour', Evaluation.evaluated_at).label('hour'),
            func.count(Evaluation.id).label('count')
        ).where(
            and_(
                Evaluation.evaluated_at >= start_time,
                Evaluation.evaluated_at <= end_time,
                Evaluation.evaluator_type == "ai_model"
            )
        ).group_by('hour').order_by('hour')
        
        counts_result = await session.execute(counts_query)
        
        timestamps = []
        values = []
        for hour, count in counts_result.fetchall():
            timestamps.append(hour.isoformat())
            values.append(int(count))
        
        throughput_trend = TrendData(
            timestamps=timestamps,
            values=values,
            labels=[f"{v} evals" for v in values]
        )
        
        # Success rate trend
        success_trend = await self._get_success_rate_trend(session, start_time, end_time)
        
        return {
            "throughput": throughput_trend,
            "success_rate": success_trend
        }
    
    async def _get_success_rate_trend(self, session: AsyncSession, start_time: datetime, end_time: datetime) -> TrendData:
        """Get success rate trend over time."""
        # Hourly success rates
        success_query = select(
            func.date_trunc('hour', Evaluation.evaluated_at).label('hour'),
            func.count(Evaluation.id).label('total'),
            func.count(Evaluation.score).label('successful')
        ).where(
            and_(
                Evaluation.evaluated_at >= start_time,
                Evaluation.evaluated_at <= end_time,
                Evaluation.evaluator_type == "ai_model"
            )
        ).group_by('hour').order_by('hour')
        
        success_result = await session.execute(success_query)
        
        timestamps = []
        values = []
        for hour, total, successful in success_result.fetchall():
            timestamps.append(hour.isoformat())
            success_rate = (successful / total) if total > 0 else 0
            values.append(success_rate)
        
        return TrendData(
            timestamps=timestamps,
            values=values,
            labels=[f"{v:.1%}" for v in values]
        )
    
    async def _get_model_comparison(self, session: AsyncSession, start_time: datetime, end_time: datetime) -> List[ModelPerformanceComparison]:
        """Get performance comparison between different evaluator models."""
        models_query = select(Evaluation.eval_metadata).where(
            and_(
                Evaluation.evaluated_at >= start_time,
                Evaluation.evaluated_at <= end_time,
                Evaluation.evaluator_type == "ai_model",
                Evaluation.eval_metadata.isnot(None)
            )
        )
        models_result = await session.execute(models_query)
        
        model_stats = defaultdict(lambda: {
            "evaluations": 0,
            "successful": 0,
            "total_latency": 0,
            "total_cost": 0,
            "scores": []
        })
        
        for metadata_row in models_result.fetchall():
            metadata = metadata_row[0]
            if isinstance(metadata, dict):
                model = metadata.get('evaluator_model', 'unknown')
                stats = model_stats[model]
                
                stats["evaluations"] += 1
                
                if 'score' in metadata and metadata['score'] is not None:
                    stats["successful"] += 1
                    stats["scores"].append(float(metadata['score']))
                
                if 'evaluation_time_ms' in metadata:
                    latency = metadata.get('evaluation_time_ms', 0)
                    if latency:
                        stats["total_latency"] += float(latency)
                
                if 'cost_usd' in metadata:
                    cost = metadata.get('cost_usd', 0)
                    if cost:
                        stats["total_cost"] += float(cost)
        
        # Calculate comparison metrics
        comparisons = []
        duration_hours = (end_time - start_time).total_seconds() / 3600
        
        for model, stats in model_stats.items():
            if stats["evaluations"] > 0:
                success_rate = stats["successful"] / stats["evaluations"]
                avg_latency = stats["total_latency"] / stats["evaluations"] if stats["evaluations"] > 0 else 0
                cost_per_eval = stats["total_cost"] / stats["evaluations"] if stats["evaluations"] > 0 else 0
                throughput = stats["evaluations"] / duration_hours if duration_hours > 0 else 0
                error_rate = 1 - success_rate
                
                # Score statistics
                scores = stats["scores"]
                avg_score = statistics.mean(scores) if scores else 0
                score_std_dev = statistics.stdev(scores) if len(scores) > 1 else 0
                
                comparisons.append(ModelPerformanceComparison(
                    model_name=model,
                    total_evaluations=stats["evaluations"],
                    success_rate=success_rate,
                    average_latency_ms=avg_latency,
                    total_cost_usd=stats["total_cost"],
                    average_score=avg_score,
                    score_std_dev=score_std_dev,
                    cost_per_evaluation=cost_per_eval,
                    throughput_per_hour=throughput,
                    error_rate=error_rate
                ))
        
        # Sort by total evaluations (most used first)
        comparisons.sort(key=lambda x: x.total_evaluations, reverse=True)
        
        return comparisons
    
    async def _get_system_health(self) -> Dict[str, Any]:
        """Get overall system health metrics."""
        health = {
            "status": "healthy",
            "uptime_seconds": 0,
            "memory_usage": "unknown",
            "cpu_usage": "unknown",
            "database_status": "healthy",
            "batch_processor_status": "unknown",
            "calibration_system_status": "unknown"
        }
        
        # Check batch processor health
        if BATCH_MONITORING_AVAILABLE and batch_processor:
            try:
                stats = await batch_processor.get_system_stats()
                health["batch_processor_status"] = "healthy"
                health["uptime_seconds"] = stats["uptime_seconds"]
                
                # Check if success rate is concerning
                if stats["success_rate"] < 0.9:
                    health["status"] = "warning"
            except Exception as e:
                health["batch_processor_status"] = f"error: {str(e)}"
                health["status"] = "warning"
        
        # Check calibration system health
        if CALIBRATION_MONITORING_AVAILABLE and calibration_system:
            try:
                cal_stats = await calibration_system.get_calibration_stats()
                health["calibration_system_status"] = "healthy"
            except Exception as e:
                health["calibration_system_status"] = f"error: {str(e)}"
                health["status"] = "warning"
        
        return health
    
    async def _get_active_alerts(self) -> List[SystemAlert]:
        """Get active system alerts."""
        alerts = []
        
        # Check batch processor metrics for alerts
        if BATCH_MONITORING_AVAILABLE and batch_processor:
            try:
                stats = await batch_processor.get_system_stats()
                
                # Success rate alert
                success_rate = stats["success_rate"]
                thresholds = self.alert_thresholds[MetricType.SUCCESS_RATE]
                
                if success_rate < thresholds["critical"]:
                    alerts.append(SystemAlert(
                        id=f"success_rate_critical_{int(time.time())}",
                        level=AlertLevel.CRITICAL,
                        title="Critical Success Rate",
                        message=f"Evaluation success rate is critically low: {success_rate:.1%}",
                        metric_type=MetricType.SUCCESS_RATE,
                        threshold_value=thresholds["critical"],
                        current_value=success_rate
                    ))
                elif success_rate < thresholds["error"]:
                    alerts.append(SystemAlert(
                        id=f"success_rate_error_{int(time.time())}",
                        level=AlertLevel.ERROR,
                        title="Low Success Rate",
                        message=f"Evaluation success rate is low: {success_rate:.1%}",
                        metric_type=MetricType.SUCCESS_RATE,
                        threshold_value=thresholds["error"],
                        current_value=success_rate
                    ))
                elif success_rate < thresholds["warning"]:
                    alerts.append(SystemAlert(
                        id=f"success_rate_warning_{int(time.time())}",
                        level=AlertLevel.WARNING,
                        title="Success Rate Warning",
                        message=f"Evaluation success rate is below optimal: {success_rate:.1%}",
                        metric_type=MetricType.SUCCESS_RATE,
                        threshold_value=thresholds["warning"],
                        current_value=success_rate
                    ))
                
                # Throughput alert
                throughput = stats["throughput_per_hour"]
                throughput_thresholds = self.alert_thresholds[MetricType.THROUGHPUT]
                
                if throughput < throughput_thresholds["critical"]:
                    alerts.append(SystemAlert(
                        id=f"throughput_critical_{int(time.time())}",
                        level=AlertLevel.CRITICAL,
                        title="Critical Low Throughput",
                        message=f"System throughput is critically low: {throughput:.1f} evals/hour",
                        metric_type=MetricType.THROUGHPUT,
                        threshold_value=throughput_thresholds["critical"],
                        current_value=throughput
                    ))
                
            except Exception as e:
                logger.error(f"Error checking batch processor alerts: {e}")
        
        return alerts
    
    async def _get_calibration_metrics(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Get calibration system performance metrics."""
        if not CALIBRATION_MONITORING_AVAILABLE or not calibration_system:
            return {
                "available": False,
                "total_calibration_points": 0,
                "models_trained": 0,
                "average_accuracy": 0
            }
        
        try:
            cal_stats = await calibration_system.get_calibration_stats()
            return {
                "available": True,
                "total_calibration_points": cal_stats.get("total_data_points", 0),
                "models_trained": cal_stats.get("trained_models", 0),
                "average_accuracy": cal_stats.get("average_performance", {}).get("r2_score", 0),
                "calibration_coverage": cal_stats.get("calibration_coverage", {}),
                "recent_training": cal_stats.get("recent_training_activity", [])
            }
        except Exception as e:
            logger.error(f"Error getting calibration metrics: {e}")
            return {
                "available": False,
                "error": str(e)
            }
    
    async def generate_analytics_report(
        self, 
        time_range: TimeRange = TimeRange.DAY,
        include_recommendations: bool = True
    ) -> AnalyticsReport:
        """Generate comprehensive analytics report."""
        report_id = f"report_{int(time.time())}"
        
        # Get system overview
        overview = await self.get_system_overview(time_range)
        
        # Extract key metrics
        metrics = {}
        eval_metrics = overview["evaluation_metrics"]
        batch_metrics = overview["batch_metrics"]
        cost_metrics = overview["cost_metrics"]
        
        # Create performance metrics
        metrics["throughput"] = PerformanceMetric(
            metric_type=MetricType.THROUGHPUT,
            value=eval_metrics["throughput_per_hour"],
            unit="evaluations/hour"
        )
        
        metrics["success_rate"] = PerformanceMetric(
            metric_type=MetricType.SUCCESS_RATE,
            value=eval_metrics["success_rate"],
            unit="percentage"
        )
        
        metrics["total_cost"] = PerformanceMetric(
            metric_type=MetricType.COST,
            value=cost_metrics["total_cost_usd"],
            unit="USD"
        )
        
        # Generate insights
        insights = await self._generate_insights(overview)
        
        # Generate recommendations
        recommendations = []
        if include_recommendations:
            recommendations = await self._generate_recommendations(overview)
        
        return AnalyticsReport(
            report_id=report_id,
            title=f"Model Evaluation Performance Report - {time_range.value}",
            time_range=time_range,
            generated_at=datetime.utcnow().isoformat(),
            metrics=metrics,
            trends=overview["performance_trends"],
            insights=insights,
            recommendations=recommendations,
            raw_data=overview
        )
    
    async def _generate_insights(self, overview: Dict[str, Any]) -> List[str]:
        """Generate automated insights from performance data."""
        insights = []
        
        eval_metrics = overview["evaluation_metrics"]
        batch_metrics = overview["batch_metrics"]
        cost_metrics = overview["cost_metrics"]
        model_comparison = overview["model_comparison"]
        
        # Success rate insights
        success_rate = eval_metrics["success_rate"]
        if success_rate >= 0.98:
            insights.append(f"Excellent evaluation success rate of {success_rate:.1%} indicates robust system performance.")
        elif success_rate >= 0.95:
            insights.append(f"Good evaluation success rate of {success_rate:.1%} with room for improvement.")
        else:
            insights.append(f"Evaluation success rate of {success_rate:.1%} requires attention to improve reliability.")
        
        # Throughput insights
        throughput = eval_metrics["throughput_per_hour"]
        if throughput > 100:
            insights.append(f"High throughput of {throughput:.1f} evaluations/hour indicates efficient processing.")
        elif throughput > 10:
            insights.append(f"Moderate throughput of {throughput:.1f} evaluations/hour is acceptable for current workload.")
        else:
            insights.append(f"Low throughput of {throughput:.1f} evaluations/hour may indicate performance bottlenecks.")
        
        # Cost insights
        avg_cost = cost_metrics["average_cost_per_evaluation"]
        if avg_cost < 0.01:
            insights.append(f"Low average cost of ${avg_cost:.4f} per evaluation indicates cost-efficient operations.")
        elif avg_cost < 0.05:
            insights.append(f"Moderate average cost of ${avg_cost:.4f} per evaluation is within expected range.")
        else:
            insights.append(f"High average cost of ${avg_cost:.4f} per evaluation suggests potential for cost optimization.")
        
        # Model comparison insights
        if model_comparison:
            best_model = max(model_comparison, key=lambda x: x.success_rate)
            most_used = max(model_comparison, key=lambda x: x.total_evaluations)
            
            if best_model.model_name != most_used.model_name:
                insights.append(f"Most used model ({most_used.model_name}) differs from best performing model ({best_model.model_name}).")
            
            if len(model_comparison) > 1:
                cost_efficient = min(model_comparison, key=lambda x: x.cost_per_evaluation)
                insights.append(f"Most cost-efficient model is {cost_efficient.model_name} at ${cost_efficient.cost_per_evaluation:.4f} per evaluation.")
        
        return insights
    
    async def _generate_recommendations(self, overview: Dict[str, Any]) -> List[str]:
        """Generate automated recommendations for system optimization."""
        recommendations = []
        
        eval_metrics = overview["evaluation_metrics"]
        batch_metrics = overview["batch_metrics"]
        cost_metrics = overview["cost_metrics"]
        model_comparison = overview["model_comparison"]
        
        # Success rate recommendations
        success_rate = eval_metrics["success_rate"]
        if success_rate < 0.95:
            recommendations.append("Consider investigating evaluation failures and implementing retry mechanisms.")
            recommendations.append("Review error logs to identify common failure patterns and address root causes.")
        
        # Cost optimization recommendations
        if model_comparison and len(model_comparison) > 1:
            most_expensive = max(model_comparison, key=lambda x: x.cost_per_evaluation)
            least_expensive = min(model_comparison, key=lambda x: x.cost_per_evaluation)
            
            if most_expensive.cost_per_evaluation > least_expensive.cost_per_evaluation * 2:
                recommendations.append(f"Consider using {least_expensive.model_name} instead of {most_expensive.model_name} for cost savings.")
        
        # Throughput optimization recommendations
        if batch_metrics["system_utilization"] < 0.5:
            recommendations.append("System utilization is low. Consider increasing parallel workers for batch processing.")
        elif batch_metrics["system_utilization"] > 0.9:
            recommendations.append("System utilization is high. Consider adding more capacity or implementing load balancing.")
        
        # Batch processing recommendations
        if batch_metrics["failed_jobs"] > 0:
            recommendations.append("Monitor and investigate failed batch jobs to improve reliability.")
        
        # Calibration recommendations
        cal_metrics = overview["calibration_metrics"]
        if cal_metrics["available"] and cal_metrics["total_calibration_points"] < 100:
            recommendations.append("Collect more human evaluation data to improve calibration model accuracy.")
        
        return recommendations
    
    def _get_start_time(self, end_time: datetime, time_range: TimeRange) -> datetime:
        """Calculate start time based on time range."""
        if time_range == TimeRange.HOUR:
            return end_time - timedelta(hours=1)
        elif time_range == TimeRange.DAY:
            return end_time - timedelta(days=1)
        elif time_range == TimeRange.WEEK:
            return end_time - timedelta(days=7)
        elif time_range == TimeRange.MONTH:
            return end_time - timedelta(days=30)
        elif time_range == TimeRange.QUARTER:
            return end_time - timedelta(days=90)
        elif time_range == TimeRange.YEAR:
            return end_time - timedelta(days=365)
        else:
            return end_time - timedelta(days=1)  # Default to 1 day
    
    async def export_report_data(
        self, 
        report: AnalyticsReport, 
        format: str = "json"
    ) -> Union[str, Dict[str, Any]]:
        """Export analytics report in specified format."""
        if format.lower() == "json":
            return {
                "report_id": report.report_id,
                "title": report.title,
                "time_range": report.time_range.value,
                "generated_at": report.generated_at,
                "metrics": {k: {
                    "type": v.metric_type.value,
                    "value": v.value,
                    "unit": v.unit,
                    "timestamp": v.timestamp
                } for k, v in report.metrics.items()},
                "trends": {k: {
                    "timestamps": v.timestamps,
                    "values": v.values,
                    "labels": v.labels
                } for k, v in report.trends.items()},
                "insights": report.insights,
                "recommendations": report.recommendations,
                "raw_data": report.raw_data
            }
        elif format.lower() == "csv":
            # Generate CSV data for key metrics
            csv_lines = ["Metric,Value,Unit,Timestamp"]
            for name, metric in report.metrics.items():
                csv_lines.append(f"{name},{metric.value},{metric.unit},{metric.timestamp}")
            return "\n".join(csv_lines)
        else:
            raise ValueError(f"Unsupported export format: {format}")

# Global performance analytics instance
performance_analytics = PerformanceAnalytics() 