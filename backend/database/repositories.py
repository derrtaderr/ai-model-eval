"""
Optimized Repository Pattern with Performance Monitoring
Provides high-performance data access with intelligent caching and query optimization.
"""

import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from abc import ABC, abstractmethod

from sqlalchemy import select, update, delete, func, and_, or_, desc, asc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload, joinedload
from pydantic import BaseModel

from .models import Trace, Evaluation, Experiment, TestCase, TestRun, User, TraceTag
from .performance import (
    query_performance_decorator,
    query_monitor,
    query_optimizer,
    query_cache
)
from services.cache_service import cache_service, trace_cache, evaluation_cache
from config.performance import CACHE_TTL_SETTINGS, RESPONSE_SIZE_LIMITS

logger = logging.getLogger(__name__)


class BaseRepository(ABC):
    """Base repository with common optimization patterns."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    @abstractmethod
    def get_model_class(self):
        """Return the SQLAlchemy model class for this repository."""
        pass
    
    async def get_by_id(self, id: str, team_id: str = None) -> Optional[Any]:
        """Get entity by ID with team isolation."""
        model = self.get_model_class()
        query = select(model).where(model.id == id)
        
        # Add team isolation if applicable
        if team_id and hasattr(model, 'team_id'):
            query = query.where(model.team_id == team_id)
        
        result = await self.session.execute(query)
        return result.scalar_one_or_none()
    
    async def get_paginated(
        self,
        filters: Dict[str, Any] = None,
        order_by: str = None,
        limit: int = 50,
        offset: int = 0,
        team_id: str = None
    ) -> Dict[str, Any]:
        """Get paginated results with intelligent caching."""
        model = self.get_model_class()
        
        # Build base query
        query = select(model)
        count_query = select(func.count(model.id))
        
        # Apply team isolation
        if team_id and hasattr(model, 'team_id'):
            query = query.where(model.team_id == team_id)
            count_query = count_query.where(model.team_id == team_id)
        
        # Apply filters
        if filters:
            filter_conditions = self._build_filter_conditions(model, filters)
            if filter_conditions is not None:
                query = query.where(filter_conditions)
                count_query = count_query.where(filter_conditions)
        
        # Apply ordering
        if order_by:
            order_column = getattr(model, order_by.lstrip('-'), None)
            if order_column is not None:
                if order_by.startswith('-'):
                    query = query.order_by(desc(order_column))
                else:
                    query = query.order_by(asc(order_column))
        
        # Apply pagination
        query = query.limit(min(limit, RESPONSE_SIZE_LIMITS["pagination_max_limit"]))
        query = query.offset(offset)
        
        # Execute queries
        items_result = await self.session.execute(query)
        count_result = await self.session.execute(count_query)
        
        items = items_result.scalars().all()
        total = count_result.scalar()
        
        return {
            "items": items,
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": offset + len(items) < total
        }
    
    def _build_filter_conditions(self, model, filters: Dict[str, Any]):
        """Build SQLAlchemy filter conditions from filter dictionary."""
        conditions = []
        
        for field, value in filters.items():
            if not hasattr(model, field):
                continue
            
            column = getattr(model, field)
            
            if isinstance(value, dict):
                # Handle range filters
                if 'gte' in value:
                    conditions.append(column >= value['gte'])
                if 'lte' in value:
                    conditions.append(column <= value['lte'])
                if 'gt' in value:
                    conditions.append(column > value['gt'])
                if 'lt' in value:
                    conditions.append(column < value['lt'])
                if 'in' in value:
                    conditions.append(column.in_(value['in']))
                if 'contains' in value:
                    conditions.append(column.contains(value['contains']))
            elif isinstance(value, list):
                conditions.append(column.in_(value))
            else:
                conditions.append(column == value)
        
        return and_(*conditions) if conditions else None


class TraceRepository(BaseRepository):
    """Optimized repository for trace operations."""
    
    def get_model_class(self):
        return Trace
    
    @query_performance_decorator(cache_ttl=CACHE_TTL_SETTINGS["trace_stats"], table_hint="traces")
    async def get_traces_with_evaluations(
        self,
        team_id: str,
        filters: Dict[str, Any] = None,
        limit: int = 50,
        offset: int = 0
    ) -> Dict[str, Any]:
        """Get traces with their evaluations using optimized joins."""
        
        # Optimized query with eager loading
        query = (
            select(Trace)
            .options(
                selectinload(Trace.evaluations),
                selectinload(Trace.trace_tags)
            )
            .where(Trace.team_id == team_id)
            .order_by(desc(Trace.timestamp))
        )
        
        # Apply filters
        if filters:
            filter_conditions = self._build_filter_conditions(Trace, filters)
            if filter_conditions is not None:
                query = query.where(filter_conditions)
        
        # Apply pagination
        query = query.limit(limit).offset(offset)
        
        # Get count for pagination
        count_query = (
            select(func.count(Trace.id))
            .where(Trace.team_id == team_id)
        )
        if filters:
            filter_conditions = self._build_filter_conditions(Trace, filters)
            if filter_conditions is not None:
                count_query = count_query.where(filter_conditions)
        
        # Execute queries
        traces_result = await self.session.execute(query)
        count_result = await self.session.execute(count_query)
        
        traces = traces_result.scalars().all()
        total = count_result.scalar()
        
        return {
            "traces": traces,
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": offset + len(traces) < total
        }
    
    @query_performance_decorator(cache_ttl=CACHE_TTL_SETTINGS["trace_stats"])
    async def get_trace_statistics(self, team_id: str, date_range: Dict[str, datetime] = None) -> Dict[str, Any]:
        """Get comprehensive trace statistics with caching."""
        
        base_query = select(Trace).where(Trace.team_id == team_id)
        
        # Apply date range filter
        if date_range:
            if 'start' in date_range:
                base_query = base_query.where(Trace.timestamp >= date_range['start'])
            if 'end' in date_range:
                base_query = base_query.where(Trace.timestamp <= date_range['end'])
        
        # Get various statistics
        stats_queries = {
            "total_traces": select(func.count(Trace.id)).select_from(base_query.subquery()),
            "avg_latency": select(func.avg(Trace.latency_ms)).select_from(base_query.subquery()),
            "total_cost": select(func.sum(Trace.cost_usd)).select_from(base_query.subquery()),
            "model_distribution": (
                select(Trace.model_name, func.count(Trace.id).label('count'))
                .select_from(base_query.subquery())
                .group_by(Trace.model_name)
            ),
            "daily_counts": (
                select(
                    func.date(Trace.timestamp).label('date'),
                    func.count(Trace.id).label('count')
                )
                .select_from(base_query.subquery())
                .group_by(func.date(Trace.timestamp))
                .order_by(func.date(Trace.timestamp))
            )
        }
        
        results = {}
        for stat_name, query in stats_queries.items():
            result = await self.session.execute(query)
            if stat_name in ["model_distribution", "daily_counts"]:
                results[stat_name] = [{"name": row[0], "value": row[1]} for row in result.all()]
            else:
                results[stat_name] = result.scalar()
        
        return results
    
    async def create_trace_batch(self, traces_data: List[Dict[str, Any]], team_id: str) -> List[Trace]:
        """Optimized batch trace creation."""
        traces = []
        
        for trace_data in traces_data:
            trace_data['team_id'] = team_id
            trace = Trace(**trace_data)
            traces.append(trace)
        
        self.session.add_all(traces)
        await self.session.flush()  # Get IDs without committing
        
        # Invalidate related caches
        await self._invalidate_trace_caches(team_id)
        
        return traces
    
    async def _invalidate_trace_caches(self, team_id: str):
        """Invalidate trace-related cache entries."""
        # Use the enhanced cache service with Redis backend
        await trace_cache.invalidate_team_cache(team_id)


class EvaluationRepository(BaseRepository):
    """Optimized repository for evaluation operations."""
    
    def get_model_class(self):
        return Evaluation
    
    @query_performance_decorator(cache_ttl=CACHE_TTL_SETTINGS["evaluation_summaries"])
    async def get_evaluation_summary(self, team_id: str, date_range: Dict[str, datetime] = None) -> Dict[str, Any]:
        """Get evaluation performance summary."""
        
        base_query = (
            select(Evaluation)
            .join(Trace, Evaluation.trace_id == Trace.id)
            .where(Trace.team_id == team_id)
        )
        
        # Apply date range
        if date_range:
            if 'start' in date_range:
                base_query = base_query.where(Evaluation.evaluated_at >= date_range['start'])
            if 'end' in date_range:
                base_query = base_query.where(Evaluation.evaluated_at <= date_range['end'])
        
        # Get evaluation statistics
        stats_queries = {
            "total_evaluations": select(func.count(Evaluation.id)).select_from(base_query.subquery()),
            "avg_score": select(func.avg(Evaluation.score)).select_from(base_query.subquery()),
            "evaluation_by_type": (
                select(Evaluation.evaluator_type, func.count(Evaluation.id).label('count'))
                .select_from(base_query.subquery())
                .group_by(Evaluation.evaluator_type)
            ),
            "label_distribution": (
                select(Evaluation.label, func.count(Evaluation.id).label('count'))
                .select_from(base_query.subquery())
                .group_by(Evaluation.label)
            )
        }
        
        results = {}
        for stat_name, query in stats_queries.items():
            result = await self.session.execute(query)
            if stat_name in ["evaluation_by_type", "label_distribution"]:
                results[stat_name] = [{"name": row[0], "value": row[1]} for row in result.all()]
            else:
                results[stat_name] = result.scalar()
        
        return results
    
    async def create_evaluation(self, evaluation_data: Dict[str, Any], team_id: str) -> Evaluation:
        """Create evaluation with cache invalidation."""
        evaluation_data['team_id'] = team_id
        evaluation = Evaluation(**evaluation_data)
        
        self.session.add(evaluation)
        await self.session.flush()
        
        # Invalidate related caches
        await self._invalidate_evaluation_caches(team_id)
        
        return evaluation
    
    async def _invalidate_evaluation_caches(self, team_id: str):
        """Invalidate evaluation-related cache entries."""
        # Use the enhanced cache service with Redis backend
        cache_patterns = [
            f"*{team_id}*"
        ]
        
        for pattern in cache_patterns:
            await cache_service.delete_pattern(pattern, prefix="evaluation")


class ExperimentRepository(BaseRepository):
    """Optimized repository for experiment operations."""
    
    def get_model_class(self):
        return Experiment
    
    @query_performance_decorator(cache_ttl=CACHE_TTL_SETTINGS["experiment_results"])
    async def get_experiment_results(self, experiment_id: str, team_id: str) -> Dict[str, Any]:
        """Get detailed experiment results with caching."""
        
        # Get experiment with related data
        experiment_query = (
            select(Experiment)
            .where(and_(Experiment.id == experiment_id, Experiment.team_id == team_id))
        )
        
        experiment_result = await self.session.execute(experiment_query)
        experiment = experiment_result.scalar_one_or_none()
        
        if not experiment:
            return None
        
        # Get related traces and evaluations
        # This would be expanded based on experiment design
        
        return {
            "experiment": experiment,
            "status": experiment.status,
            "metrics": experiment.metrics,
            "config": experiment.config
        }


class DashboardRepository:
    """Specialized repository for dashboard analytics."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    @query_performance_decorator(cache_ttl=CACHE_TTL_SETTINGS["dashboard_analytics"])
    async def get_dashboard_overview(self, team_id: str, date_range: Dict[str, datetime] = None) -> Dict[str, Any]:
        """Get comprehensive dashboard overview with aggressive caching."""
        
        # Initialize repositories
        trace_repo = TraceRepository(self.session)
        eval_repo = EvaluationRepository(self.session)
        
        # Get all dashboard data in parallel
        import asyncio
        
        dashboard_data = await asyncio.gather(
            trace_repo.get_trace_statistics(team_id, date_range),
            eval_repo.get_evaluation_summary(team_id, date_range),
            self._get_recent_activity(team_id),
            return_exceptions=True
        )
        
        trace_stats, eval_summary, recent_activity = dashboard_data
        
        return {
            "trace_statistics": trace_stats if not isinstance(trace_stats, Exception) else {},
            "evaluation_summary": eval_summary if not isinstance(eval_summary, Exception) else {},
            "recent_activity": recent_activity if not isinstance(recent_activity, Exception) else [],
            "performance_metrics": await self._get_performance_metrics(),
            "generated_at": datetime.utcnow().isoformat()
        }
    
    async def _get_recent_activity(self, team_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent activity across all entities."""
        
        # Get recent traces
        recent_traces = await self.session.execute(
            select(Trace.id, Trace.timestamp, Trace.model_name, Trace.status)
            .where(Trace.team_id == team_id)
            .order_by(desc(Trace.timestamp))
            .limit(limit)
        )
        
        # Get recent evaluations
        recent_evaluations = await self.session.execute(
            select(Evaluation.id, Evaluation.evaluated_at, Evaluation.evaluator_type, Evaluation.label)
            .join(Trace, Evaluation.trace_id == Trace.id)
            .where(Trace.team_id == team_id)
            .order_by(desc(Evaluation.evaluated_at))
            .limit(limit)
        )
        
        activity = []
        
        for trace in recent_traces.all():
            activity.append({
                "type": "trace",
                "timestamp": trace.timestamp,
                "details": {
                    "id": str(trace.id),
                    "model_name": trace.model_name,
                    "status": trace.status
                }
            })
        
        for evaluation in recent_evaluations.all():
            activity.append({
                "type": "evaluation",
                "timestamp": evaluation.evaluated_at,
                "details": {
                    "id": str(evaluation.id),
                    "evaluator_type": evaluation.evaluator_type,
                    "label": evaluation.label
                }
            })
        
        # Sort by timestamp and return most recent
        activity.sort(key=lambda x: x["timestamp"], reverse=True)
        return activity[:limit]
    
    async def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get current system performance metrics."""
        from .performance import get_database_performance_metrics
        return await get_database_performance_metrics()


# Repository factory
class RepositoryFactory:
    """Factory for creating repository instances."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    def get_trace_repository(self) -> TraceRepository:
        return TraceRepository(self.session)
    
    def get_evaluation_repository(self) -> EvaluationRepository:
        return EvaluationRepository(self.session)
    
    def get_experiment_repository(self) -> ExperimentRepository:
        return ExperimentRepository(self.session)
    
    def get_dashboard_repository(self) -> DashboardRepository:
        return DashboardRepository(self.session) 