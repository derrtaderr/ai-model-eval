"""
Analytics Service
Handles real-time metrics calculation, trend analysis, and alerting with intelligent caching.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from sqlalchemy import select, func, and_, desc, asc
from sqlalchemy.ext.asyncio import AsyncSession
import numpy as np
from scipy import stats
import json

from .models import (
    MetricDefinition, MetricValue, TrendAnalysis, Alert,
    MetricType, AlertSeverity, AlertStatus, TrendDirection
)
from database.models import Trace, Evaluation, Experiment
from services.cache_service import cache_service
from services.redis_service import cache_result
from config.performance import CACHE_TTL_SETTINGS

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Handles calculation of various metrics from trace data."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def calculate_latency_metrics(self, team_id: str, time_range: Dict[str, datetime]) -> Dict[str, float]:
        """Calculate latency-based metrics."""
        query = (
            select(
                func.avg(Trace.latency_ms).label('avg_latency'),
                func.percentile_cont(0.5).within_group(Trace.latency_ms.asc()).label('p50_latency'),
                func.percentile_cont(0.95).within_group(Trace.latency_ms.asc()).label('p95_latency'),
                func.percentile_cont(0.99).within_group(Trace.latency_ms.asc()).label('p99_latency'),
                func.max(Trace.latency_ms).label('max_latency'),
                func.count(Trace.id).label('total_requests')
            )
            .where(
                and_(
                    Trace.team_id == team_id,
                    Trace.timestamp >= time_range['start'],
                    Trace.timestamp <= time_range['end']
                )
            )
        )
        
        result = await self.session.execute(query)
        row = result.first()
        
        return {
            'avg_latency_ms': float(row.avg_latency) if row.avg_latency else 0.0,
            'p50_latency_ms': float(row.p50_latency) if row.p50_latency else 0.0,
            'p95_latency_ms': float(row.p95_latency) if row.p95_latency else 0.0,
            'p99_latency_ms': float(row.p99_latency) if row.p99_latency else 0.0,
            'max_latency_ms': float(row.max_latency) if row.max_latency else 0.0,
            'total_requests': int(row.total_requests) if row.total_requests else 0
        }
    
    async def calculate_cost_metrics(self, team_id: str, time_range: Dict[str, datetime]) -> Dict[str, float]:
        """Calculate cost-based metrics."""
        query = (
            select(
                func.sum(Trace.cost_usd).label('total_cost'),
                func.avg(Trace.cost_usd).label('avg_cost'),
                func.sum(Trace.input_tokens + Trace.output_tokens).label('total_tokens'),
                func.avg(Trace.input_tokens + Trace.output_tokens).label('avg_tokens'),
                func.count(Trace.id).label('total_requests')
            )
            .where(
                and_(
                    Trace.team_id == team_id,
                    Trace.timestamp >= time_range['start'],
                    Trace.timestamp <= time_range['end']
                )
            )
        )
        
        result = await self.session.execute(query)
        row = result.first()
        
        total_cost = float(row.total_cost) if row.total_cost else 0.0
        total_requests = int(row.total_requests) if row.total_requests else 0
        
        return {
            'total_cost_usd': total_cost,
            'avg_cost_per_request_usd': float(row.avg_cost) if row.avg_cost else 0.0,
            'cost_per_1k_requests_usd': (total_cost / total_requests * 1000) if total_requests > 0 else 0.0,
            'total_tokens': int(row.total_tokens) if row.total_tokens else 0,
            'avg_tokens_per_request': float(row.avg_tokens) if row.avg_tokens else 0.0,
            'total_requests': total_requests
        }
    
    async def calculate_quality_metrics(self, team_id: str, time_range: Dict[str, datetime]) -> Dict[str, float]:
        """Calculate quality-based metrics from evaluations."""
        query = (
            select(
                func.avg(Evaluation.score).label('avg_score'),
                func.count(Evaluation.id).label('total_evaluations'),
                func.sum(func.case((Evaluation.label == 'good', 1), else_=0)).label('good_evaluations'),
                func.sum(func.case((Evaluation.label == 'bad', 1), else_=0)).label('bad_evaluations')
            )
            .join(Trace, Evaluation.trace_id == Trace.id)
            .where(
                and_(
                    Trace.team_id == team_id,
                    Evaluation.evaluated_at >= time_range['start'],
                    Evaluation.evaluated_at <= time_range['end']
                )
            )
        )
        
        result = await self.session.execute(query)
        row = result.first()
        
        total_evaluations = int(row.total_evaluations) if row.total_evaluations else 0
        good_evaluations = int(row.good_evaluations) if row.good_evaluations else 0
        bad_evaluations = int(row.bad_evaluations) if row.bad_evaluations else 0
        
        return {
            'avg_evaluation_score': float(row.avg_score) if row.avg_score else 0.0,
            'total_evaluations': total_evaluations,
            'good_evaluation_rate': (good_evaluations / total_evaluations) if total_evaluations > 0 else 0.0,
            'bad_evaluation_rate': (bad_evaluations / total_evaluations) if total_evaluations > 0 else 0.0,
            'evaluation_coverage': total_evaluations  # Can be enhanced with total traces
        }
    
    async def calculate_error_metrics(self, team_id: str, time_range: Dict[str, datetime]) -> Dict[str, float]:
        """Calculate error rate and failure metrics."""
        query = (
            select(
                func.count(Trace.id).label('total_requests'),
                func.sum(func.case((Trace.status == 'error', 1), else_=0)).label('error_requests'),
                func.sum(func.case((Trace.status == 'timeout', 1), else_=0)).label('timeout_requests'),
                func.sum(func.case((Trace.status == 'success', 1), else_=0)).label('success_requests')
            )
            .where(
                and_(
                    Trace.team_id == team_id,
                    Trace.timestamp >= time_range['start'],
                    Trace.timestamp <= time_range['end']
                )
            )
        )
        
        result = await self.session.execute(query)
        row = result.first()
        
        total_requests = int(row.total_requests) if row.total_requests else 0
        error_requests = int(row.error_requests) if row.error_requests else 0
        timeout_requests = int(row.timeout_requests) if row.timeout_requests else 0
        success_requests = int(row.success_requests) if row.success_requests else 0
        
        return {
            'total_requests': total_requests,
            'error_rate': (error_requests / total_requests) if total_requests > 0 else 0.0,
            'timeout_rate': (timeout_requests / total_requests) if total_requests > 0 else 0.0,
            'success_rate': (success_requests / total_requests) if total_requests > 0 else 0.0,
            'availability': 1.0 - ((error_requests + timeout_requests) / total_requests) if total_requests > 0 else 1.0
        }
    
    async def calculate_throughput_metrics(self, team_id: str, time_range: Dict[str, datetime]) -> Dict[str, float]:
        """Calculate throughput and usage metrics."""
        time_diff = time_range['end'] - time_range['start']
        hours = time_diff.total_seconds() / 3600
        
        query = (
            select(
                func.count(Trace.id).label('total_requests'),
                func.count(func.distinct(Trace.user_id)).label('unique_users'),
                func.count(func.distinct(Trace.model_name)).label('unique_models')
            )
            .where(
                and_(
                    Trace.team_id == team_id,
                    Trace.timestamp >= time_range['start'],
                    Trace.timestamp <= time_range['end']
                )
            )
        )
        
        result = await self.session.execute(query)
        row = result.first()
        
        total_requests = int(row.total_requests) if row.total_requests else 0
        
        return {
            'total_requests': total_requests,
            'requests_per_hour': total_requests / hours if hours > 0 else 0.0,
            'requests_per_minute': total_requests / (hours * 60) if hours > 0 else 0.0,
            'unique_users': int(row.unique_users) if row.unique_users else 0,
            'unique_models': int(row.unique_models) if row.unique_models else 0
        }


class TrendAnalyzer:
    """Analyzes trends in metric data using statistical methods."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def analyze_metric_trend(
        self, 
        metric_definition_id: str, 
        days_back: int = 30
    ) -> Optional[Dict[str, Any]]:
        """Analyze trend for a specific metric over time."""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days_back)
        
        # Get metric values for the period
        query = (
            select(MetricValue.timestamp, MetricValue.value)
            .where(
                and_(
                    MetricValue.metric_definition_id == metric_definition_id,
                    MetricValue.timestamp >= start_date,
                    MetricValue.timestamp <= end_date
                )
            )
            .order_by(MetricValue.timestamp)
        )
        
        result = await self.session.execute(query)
        data_points = result.all()
        
        if len(data_points) < 3:
            return None
        
        # Prepare data for analysis
        timestamps = [point.timestamp for point in data_points]
        values = [point.value for point in data_points]
        
        # Convert timestamps to numerical values (days since start)
        time_numeric = [(ts - start_date).total_seconds() / 86400 for ts in timestamps]
        
        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(time_numeric, values)
        
        # Calculate trend metrics
        start_value = values[0]
        end_value = values[-1]
        absolute_change = end_value - start_value
        percentage_change = (absolute_change / start_value * 100) if start_value != 0 else 0
        
        # Calculate volatility (standard deviation)
        volatility = float(np.std(values))
        
        # Determine trend direction and strength
        direction = self._determine_trend_direction(slope, r_value, volatility)
        strength = min(abs(r_value), 1.0)  # R-squared gives us trend strength
        confidence = 1.0 - p_value if p_value < 1.0 else 0.0
        
        return {
            'direction': direction.value,
            'strength': strength,
            'confidence': confidence,
            'slope': slope,
            'r_squared': r_value ** 2,
            'absolute_change': absolute_change,
            'percentage_change': percentage_change,
            'volatility': volatility,
            'data_points': len(data_points),
            'start_date': start_date,
            'end_date': end_date,
            'algorithm_used': 'linear_regression'
        }
    
    def _determine_trend_direction(self, slope: float, r_value: float, volatility: float) -> TrendDirection:
        """Determine trend direction based on statistical analysis."""
        # Thresholds for trend classification
        SLOPE_THRESHOLD = 0.01
        R_VALUE_THRESHOLD = 0.3
        VOLATILITY_THRESHOLD = 0.5
        
        if abs(r_value) < R_VALUE_THRESHOLD or volatility > VOLATILITY_THRESHOLD:
            return TrendDirection.VOLATILE
        elif abs(slope) < SLOPE_THRESHOLD:
            return TrendDirection.STABLE
        elif slope > 0:
            return TrendDirection.UP
        else:
            return TrendDirection.DOWN


class AlertManager:
    """Manages alert generation and notification."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def check_metric_thresholds(self, metric_definition: MetricDefinition, current_value: float) -> Optional[Alert]:
        """Check if a metric value triggers any alerts."""
        if not metric_definition.is_alertable:
            return None
        
        alert_type = None
        threshold_value = None
        
        # Check critical threshold first
        if (metric_definition.critical_threshold is not None and 
            self._check_threshold(current_value, metric_definition.critical_threshold, metric_definition.threshold_direction)):
            alert_type = "critical"
            threshold_value = metric_definition.critical_threshold
        # Then check warning threshold
        elif (metric_definition.warning_threshold is not None and 
              self._check_threshold(current_value, metric_definition.warning_threshold, metric_definition.threshold_direction)):
            alert_type = "warning"
            threshold_value = metric_definition.warning_threshold
        
        if alert_type:
            # Check if there's already an active alert for this metric
            existing_alert_query = (
                select(Alert)
                .where(
                    and_(
                        Alert.metric_definition_id == metric_definition.id,
                        Alert.status == AlertStatus.ACTIVE
                    )
                )
            )
            
            result = await self.session.execute(existing_alert_query)
            existing_alert = result.scalar_one_or_none()
            
            if existing_alert is None:
                # Create new alert
                severity = AlertSeverity.CRITICAL if alert_type == "critical" else AlertSeverity.MEDIUM
                
                alert = Alert(
                    metric_definition_id=metric_definition.id,
                    team_id=metric_definition.team_id,
                    title=f"{metric_definition.display_name} {alert_type.upper()} threshold exceeded",
                    description=f"Metric value {current_value} {metric_definition.unit or ''} exceeded {alert_type} threshold of {threshold_value} {metric_definition.unit or ''}",
                    severity=severity,
                    trigger_value=current_value,
                    threshold_value=threshold_value,
                    threshold_type=alert_type
                )
                
                self.session.add(alert)
                await self.session.flush()
                
                return alert
        
        return None
    
    def _check_threshold(self, value: float, threshold: float, direction: str) -> bool:
        """Check if value crosses threshold in the specified direction."""
        if direction == "above":
            return value > threshold
        elif direction == "below":
            return value < threshold
        return False
    
    async def resolve_alert(self, alert_id: str, user_id: str, resolution_notes: str = None) -> bool:
        """Resolve an active alert."""
        query = select(Alert).where(Alert.id == alert_id)
        result = await self.session.execute(query)
        alert = result.scalar_one_or_none()
        
        if alert and alert.status == AlertStatus.ACTIVE:
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.utcnow()
            alert.resolved_by = user_id
            alert.resolution_notes = resolution_notes
            
            await self.session.commit()
            return True
        
        return False


class AnalyticsService:
    """Main analytics service that coordinates all analytics operations."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.calculator = MetricsCalculator(session)
        self.trend_analyzer = TrendAnalyzer(session)
        self.alert_manager = AlertManager(session)
    
    @cache_result(ttl=CACHE_TTL_SETTINGS["dashboard_analytics"], key_prefix="analytics")
    async def get_dashboard_overview(self, team_id: str, time_range: Dict[str, datetime] = None) -> Dict[str, Any]:
        """Get comprehensive dashboard analytics with intelligent caching."""
        if not time_range:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=24)
            time_range = {'start': start_time, 'end': end_time}
        
        # Calculate all metrics in parallel
        metrics_tasks = [
            self.calculator.calculate_latency_metrics(team_id, time_range),
            self.calculator.calculate_cost_metrics(team_id, time_range),
            self.calculator.calculate_quality_metrics(team_id, time_range),
            self.calculator.calculate_error_metrics(team_id, time_range),
            self.calculator.calculate_throughput_metrics(team_id, time_range)
        ]
        
        results = await asyncio.gather(*metrics_tasks, return_exceptions=True)
        
        # Combine results
        dashboard_data = {
            'latency_metrics': results[0] if not isinstance(results[0], Exception) else {},
            'cost_metrics': results[1] if not isinstance(results[1], Exception) else {},
            'quality_metrics': results[2] if not isinstance(results[2], Exception) else {},
            'error_metrics': results[3] if not isinstance(results[3], Exception) else {},
            'throughput_metrics': results[4] if not isinstance(results[4], Exception) else {},
            'time_range': {
                'start': time_range['start'].isoformat(),
                'end': time_range['end'].isoformat()
            },
            'generated_at': datetime.utcnow().isoformat()
        }
        
        # Get active alerts
        alerts_query = (
            select(Alert)
            .where(
                and_(
                    Alert.team_id == team_id,
                    Alert.status == AlertStatus.ACTIVE
                )
            )
            .order_by(desc(Alert.triggered_at))
            .limit(10)
        )
        
        alerts_result = await self.session.execute(alerts_query)
        active_alerts = alerts_result.scalars().all()
        
        dashboard_data['active_alerts'] = [
            {
                'id': str(alert.id),
                'title': alert.title,
                'severity': alert.severity,
                'triggered_at': alert.triggered_at.isoformat()
            }
            for alert in active_alerts
        ]
        
        return dashboard_data
    
    async def calculate_and_store_metrics(self, team_id: str) -> Dict[str, Any]:
        """Calculate all metrics for a team and store the results."""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(minutes=5)  # 5-minute window
        time_range = {'start': start_time, 'end': end_time}
        
        # Get all active metric definitions for the team
        metric_defs_query = (
            select(MetricDefinition)
            .where(
                and_(
                    MetricDefinition.team_id == team_id,
                    MetricDefinition.is_active == True
                )
            )
        )
        
        result = await self.session.execute(metric_defs_query)
        metric_definitions = result.scalars().all()
        
        stored_metrics = []
        
        for metric_def in metric_definitions:
            try:
                # Calculate metric based on type
                value = await self._calculate_metric_value(metric_def, time_range)
                
                if value is not None:
                    # Store metric value
                    metric_value = MetricValue(
                        metric_definition_id=metric_def.id,
                        team_id=team_id,
                        value=value,
                        timestamp=end_time,
                        metadata={'calculation_window_minutes': 5}
                    )
                    
                    self.session.add(metric_value)
                    stored_metrics.append({
                        'metric_name': metric_def.name,
                        'value': value,
                        'timestamp': end_time.isoformat()
                    })
                    
                    # Check for alerts
                    alert = await self.alert_manager.check_metric_thresholds(metric_def, value)
                    if alert:
                        logger.warning(f"Alert triggered for metric {metric_def.name}: {alert.title}")
                
            except Exception as e:
                logger.error(f"Error calculating metric {metric_def.name}: {e}")
        
        await self.session.commit()
        
        return {
            'team_id': team_id,
            'calculated_at': end_time.isoformat(),
            'metrics_calculated': len(stored_metrics),
            'metrics': stored_metrics
        }
    
    async def _calculate_metric_value(self, metric_def: MetricDefinition, time_range: Dict[str, datetime]) -> Optional[float]:
        """Calculate a single metric value based on its type."""
        metric_type = MetricType(metric_def.metric_type)
        
        if metric_type == MetricType.LATENCY:
            metrics = await self.calculator.calculate_latency_metrics(metric_def.team_id, time_range)
            return metrics.get('avg_latency_ms')
        elif metric_type == MetricType.COST:
            metrics = await self.calculator.calculate_cost_metrics(metric_def.team_id, time_range)
            return metrics.get('total_cost_usd')
        elif metric_type == MetricType.ERROR_RATE:
            metrics = await self.calculator.calculate_error_metrics(metric_def.team_id, time_range)
            return metrics.get('error_rate')
        elif metric_type == MetricType.THROUGHPUT:
            metrics = await self.calculator.calculate_throughput_metrics(metric_def.team_id, time_range)
            return metrics.get('requests_per_minute')
        elif metric_type == MetricType.QUALITY:
            metrics = await self.calculator.calculate_quality_metrics(metric_def.team_id, time_range)
            return metrics.get('avg_evaluation_score')
        
        return None
    
    async def analyze_trends_for_team(self, team_id: str) -> List[Dict[str, Any]]:
        """Analyze trends for all metrics of a team."""
        # Get all metric definitions for the team
        metric_defs_query = (
            select(MetricDefinition)
            .where(
                and_(
                    MetricDefinition.team_id == team_id,
                    MetricDefinition.is_active == True
                )
            )
        )
        
        result = await self.session.execute(metric_defs_query)
        metric_definitions = result.scalars().all()
        
        trend_results = []
        
        for metric_def in metric_definitions:
            try:
                trend_data = await self.trend_analyzer.analyze_metric_trend(metric_def.id)
                
                if trend_data:
                    # Store trend analysis
                    trend_analysis = TrendAnalysis(
                        metric_definition_id=metric_def.id,
                        team_id=team_id,
                        **trend_data
                    )
                    
                    self.session.add(trend_analysis)
                    
                    trend_results.append({
                        'metric_name': metric_def.name,
                        'metric_id': str(metric_def.id),
                        **trend_data
                    })
                
            except Exception as e:
                logger.error(f"Error analyzing trend for metric {metric_def.name}: {e}")
        
        await self.session.commit()
        
        return trend_results 