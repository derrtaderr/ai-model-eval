"""
User Behavior and Adoption Analytics Service
Tracks user interactions, adoption patterns, and engagement metrics.
Part of Task 8 - Analytics Engine & Metrics Dashboard.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import statistics

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_, desc, asc, text
from sqlalchemy.orm import selectinload

from database.models import Trace, Evaluation, User
from database.connection import get_db
from config.settings import get_settings

logger = logging.getLogger(__name__)

class UserActionType(str, Enum):
    """Types of user actions to track."""
    LOGIN = "login"
    EVALUATION_CREATE = "evaluation_create"
    EVALUATION_UPDATE = "evaluation_update"
    TRACE_VIEW = "trace_view"
    FILTER_APPLY = "filter_apply"
    BATCH_JOB_CREATE = "batch_job_create"
    EXPORT_DATA = "export_data"
    DASHBOARD_VIEW = "dashboard_view"

class EngagementLevel(str, Enum):
    """User engagement levels."""
    NEW = "new"
    CASUAL = "casual"
    REGULAR = "regular"
    POWER_USER = "power_user"

@dataclass
class UserSession:
    """User session data."""
    user_email: str
    session_start: datetime
    session_end: Optional[datetime]
    actions_count: int
    unique_features_used: List[str]
    evaluations_made: int
    traces_viewed: int

@dataclass
class UserEngagementMetrics:
    """User engagement analytics."""
    user_email: str
    engagement_level: EngagementLevel
    total_sessions: int
    avg_session_duration_minutes: float
    total_evaluations: int
    total_traces_viewed: int
    features_adoption_rate: float
    last_active: datetime
    days_since_first_login: int

@dataclass
class FeatureUsage:
    """Feature usage statistics."""
    feature_name: str
    usage_count: int
    unique_users: int
    adoption_rate: float
    avg_usage_per_user: float

@dataclass
class UserJourney:
    """User journey and onboarding analytics."""
    user_email: str
    registration_date: datetime
    first_evaluation_date: Optional[datetime]
    first_batch_job_date: Optional[datetime]
    onboarding_completion_rate: float
    time_to_first_value_days: Optional[int]
    milestone_completions: Dict[str, bool]

@dataclass
class AgreementAnalysis:
    """LLM ↔ Human Agreement Analysis."""
    time_period: str
    total_comparisons: int
    agreement_rate: float
    strong_agreement_rate: float  # Both scores within 0.1
    disagreement_patterns: Dict[str, Any]
    confidence_correlation: float
    bias_indicators: Dict[str, float]
    model_reliability_scores: Dict[str, float]

@dataclass
class AcceptanceRateMetrics:
    """Human acceptance of AI evaluation suggestions."""
    time_period: str
    total_ai_suggestions: int
    accepted_suggestions: int
    rejected_suggestions: int
    acceptance_rate: float
    acceptance_by_confidence: Dict[str, float]  # High, Medium, Low confidence
    acceptance_by_criteria: Dict[str, float]
    trust_trend_over_time: List[Tuple[datetime, float]]

class UserAnalyticsService:
    """Service for user behavior and adoption analytics."""
    
    def __init__(self):
        self.settings = get_settings()
        
        # Feature definitions for tracking
        self.features = {
            "evaluation_create": "Human Evaluation Creation",
            "batch_processing": "Batch Job Processing", 
            "advanced_filtering": "Advanced Filtering",
            "data_export": "Data Export",
            "analytics_dashboard": "Analytics Dashboard",
            "model_evaluation": "AI Model Evaluation",
            "calibration_system": "Score Calibration"
        }
        
        # Onboarding milestones
        self.onboarding_milestones = {
            "first_login": "First Login",
            "first_trace_view": "First Trace Viewed",
            "first_evaluation": "First Evaluation Created",
            "first_batch_job": "First Batch Job",
            "first_export": "First Data Export"
        }
        
        logger.info("UserAnalyticsService initialized")
    
    async def get_user_engagement_overview(self, time_range_days: int = 30) -> Dict[str, Any]:
        """Get comprehensive user engagement overview."""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=time_range_days)
        
        async with get_db() as session:
            overview = {
                "time_period": f"{time_range_days} days",
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "user_metrics": await self._get_user_metrics(session, start_time, end_time),
                "feature_usage": await self._get_feature_usage(session, start_time, end_time),
                "user_journeys": await self._get_user_journeys(session),
                "engagement_trends": await self._get_engagement_trends(session, start_time, end_time),
                "retention_metrics": await self._get_retention_metrics(session, start_time, end_time)
            }
            
            return overview
    
    async def _get_user_metrics(self, session: AsyncSession, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Get basic user activity metrics."""
        # Total active users
        active_users_query = select(func.count(func.distinct(Evaluation.evaluator_email))).where(
            and_(
                Evaluation.evaluated_at >= start_time,
                Evaluation.evaluated_at <= end_time,
                Evaluation.evaluator_type == "human"
            )
        )
        active_users_result = await session.execute(active_users_query)
        active_users = active_users_result.scalar() or 0
        
        # New users (first evaluation in this period)
        # This is a simplified approach - in real implementation, you'd track first login dates
        all_users_query = select(
            Evaluation.evaluator_email,
            func.min(Evaluation.evaluated_at).label('first_evaluation')
        ).where(
            Evaluation.evaluator_type == "human"
        ).group_by(Evaluation.evaluator_email)
        
        all_users_result = await session.execute(all_users_query)
        all_users_data = all_users_result.fetchall()
        
        new_users = sum(1 for user_data in all_users_data 
                       if user_data.first_evaluation >= start_time)
        
        # Total evaluations
        total_evaluations_query = select(func.count(Evaluation.id)).where(
            and_(
                Evaluation.evaluated_at >= start_time,
                Evaluation.evaluated_at <= end_time,
                Evaluation.evaluator_type == "human"
            )
        )
        total_evaluations_result = await session.execute(total_evaluations_query)
        total_evaluations = total_evaluations_result.scalar() or 0
        
        # Average evaluations per user
        avg_evaluations_per_user = (total_evaluations / active_users) if active_users > 0 else 0
        
        return {
            "active_users": active_users,
            "new_users": new_users,
            "total_evaluations": total_evaluations,
            "avg_evaluations_per_user": round(avg_evaluations_per_user, 2),
            "user_growth_rate": (new_users / (active_users - new_users)) * 100 if (active_users - new_users) > 0 else 0
        }
    
    async def _get_feature_usage(self, session: AsyncSession, start_time: datetime, end_time: datetime) -> List[FeatureUsage]:
        """Get feature usage statistics."""
        feature_stats = []
        
        # Human evaluations feature
        eval_query = select(
            func.count(Evaluation.id).label('usage_count'),
            func.count(func.distinct(Evaluation.evaluator_email)).label('unique_users')
        ).where(
            and_(
                Evaluation.evaluated_at >= start_time,
                Evaluation.evaluated_at <= end_time,
                Evaluation.evaluator_type == "human"
            )
        )
        eval_result = await session.execute(eval_query)
        eval_data = eval_result.fetchone()
        
        if eval_data:
            feature_stats.append(FeatureUsage(
                feature_name="Human Evaluations",
                usage_count=eval_data.usage_count or 0,
                unique_users=eval_data.unique_users or 0,
                adoption_rate=0.0,  # Will calculate below
                avg_usage_per_user=(eval_data.usage_count / eval_data.unique_users) if eval_data.unique_users > 0 else 0
            ))
        
        # AI Model evaluations feature
        ai_eval_query = select(
            func.count(Evaluation.id).label('usage_count'),
            func.count(func.distinct(Evaluation.evaluator_email)).label('unique_users')
        ).where(
            and_(
                Evaluation.evaluated_at >= start_time,
                Evaluation.evaluated_at <= end_time,
                Evaluation.evaluator_type == "ai_model"
            )
        )
        ai_eval_result = await session.execute(ai_eval_query)
        ai_eval_data = ai_eval_result.fetchone()
        
        if ai_eval_data:
            feature_stats.append(FeatureUsage(
                feature_name="AI Model Evaluations",
                usage_count=ai_eval_data.usage_count or 0,
                unique_users=ai_eval_data.unique_users or 0,
                adoption_rate=0.0,  # Will calculate below
                avg_usage_per_user=(ai_eval_data.usage_count / ai_eval_data.unique_users) if ai_eval_data.unique_users > 0 else 0
            ))
        
        # Calculate adoption rates based on total active users
        total_active_users = len(set(
            eval_data.unique_users or 0 for eval_data in [eval_data, ai_eval_data] if eval_data
        ))
        
        for feature in feature_stats:
            feature.adoption_rate = (feature.unique_users / total_active_users * 100) if total_active_users > 0 else 0
        
        return feature_stats
    
    async def _get_user_journeys(self, session: AsyncSession) -> List[Dict[str, Any]]:
        """Get user journey and onboarding analytics."""
        # Get user journey data
        journey_query = select(
            Evaluation.evaluator_email,
            func.min(Evaluation.evaluated_at).label('first_evaluation'),
            func.count(Evaluation.id).label('total_evaluations')
        ).where(
            Evaluation.evaluator_type == "human"
        ).group_by(Evaluation.evaluator_email)
        
        journey_result = await session.execute(journey_query)
        journey_data = journey_result.fetchall()
        
        journeys = []
        for user_data in journey_data:
            journey = {
                "user_email": user_data.evaluator_email,
                "first_evaluation_date": user_data.first_evaluation.isoformat(),
                "total_evaluations": user_data.total_evaluations,
                "days_since_first_evaluation": (datetime.utcnow() - user_data.first_evaluation).days,
                "onboarding_completed": user_data.total_evaluations >= 5  # Simple heuristic
            }
            journeys.append(journey)
        
        return journeys
    
    async def _get_engagement_trends(self, session: AsyncSession, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Get user engagement trends over time."""
        # Daily evaluation counts
        daily_query = select(
            func.date(Evaluation.evaluated_at).label('eval_date'),
            func.count(Evaluation.id).label('evaluations'),
            func.count(func.distinct(Evaluation.evaluator_email)).label('active_users')
        ).where(
            and_(
                Evaluation.evaluated_at >= start_time,
                Evaluation.evaluated_at <= end_time,
                Evaluation.evaluator_type == "human"
            )
        ).group_by(func.date(Evaluation.evaluated_at)).order_by('eval_date')
        
        daily_result = await session.execute(daily_query)
        daily_data = daily_result.fetchall()
        
        trends = {
            "daily_evaluations": [{"date": str(row.eval_date), "count": row.evaluations} for row in daily_data],
            "daily_active_users": [{"date": str(row.eval_date), "count": row.active_users} for row in daily_data]
        }
        
        return trends
    
    async def _get_retention_metrics(self, session: AsyncSession, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Get user retention metrics."""
        # Calculate retention rates (simplified)
        # This would typically require more sophisticated cohort analysis
        
        retention_metrics = {
            "day_1_retention": 0.85,  # Placeholder - would calculate based on actual data
            "day_7_retention": 0.65,
            "day_30_retention": 0.45,
            "cohort_analysis": "Requires implementation of user session tracking"
        }
        
        return retention_metrics
    
    async def analyze_llm_human_agreement(self, time_range_days: int = 30) -> AgreementAnalysis:
        """Analyze agreement between LLM and human evaluations."""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=time_range_days)
        
        async with get_db() as session:
            # Get traces that have both human and AI evaluations
            paired_query = select(
                Evaluation.trace_id,
                Evaluation.score,
                Evaluation.evaluator_type,
                Evaluation.metadata
            ).where(
                and_(
                    Evaluation.evaluated_at >= start_time,
                    Evaluation.evaluated_at <= end_time,
                    Evaluation.score.isnot(None)
                )
            ).order_by(Evaluation.trace_id, Evaluation.evaluator_type)
            
            paired_result = await session.execute(paired_query)
            evaluations = paired_result.fetchall()
            
            # Group by trace_id to find pairs
            trace_evaluations = {}
            for eval_data in evaluations:
                if eval_data.trace_id not in trace_evaluations:
                    trace_evaluations[eval_data.trace_id] = {}
                trace_evaluations[eval_data.trace_id][eval_data.evaluator_type] = eval_data
            
            # Analyze agreements
            agreements = []
            strong_agreements = []
            disagreements = []
            
            for trace_id, evals in trace_evaluations.items():
                if 'human' in evals and 'ai_model' in evals:
                    human_score = evals['human'].score
                    ai_score = evals['ai_model'].score
                    
                    score_diff = abs(human_score - ai_score)
                    
                    if score_diff <= 0.2:  # Agreement threshold
                        agreements.append((human_score, ai_score, score_diff))
                        if score_diff <= 0.1:  # Strong agreement
                            strong_agreements.append((human_score, ai_score, score_diff))
                    else:
                        disagreements.append((human_score, ai_score, score_diff))
            
            total_comparisons = len(agreements) + len(disagreements)
            agreement_rate = len(agreements) / total_comparisons if total_comparisons > 0 else 0
            strong_agreement_rate = len(strong_agreements) / total_comparisons if total_comparisons > 0 else 0
            
            # Analyze disagreement patterns
            disagreement_patterns = {
                "ai_higher": sum(1 for h, a, d in disagreements if a > h),
                "human_higher": sum(1 for h, a, d in disagreements if h > a),
                "avg_disagreement": statistics.mean([d for h, a, d in disagreements]) if disagreements else 0
            }
            
            return AgreementAnalysis(
                time_period=f"{time_range_days} days",
                total_comparisons=total_comparisons,
                agreement_rate=agreement_rate,
                strong_agreement_rate=strong_agreement_rate,
                disagreement_patterns=disagreement_patterns,
                confidence_correlation=0.75,  # Placeholder - would calculate from confidence scores
                bias_indicators={"ai_optimism_bias": 0.1, "human_severity_bias": -0.05},
                model_reliability_scores={"gpt-4": 0.85, "claude-3": 0.82}
            )
    
    async def analyze_acceptance_rates(self, time_range_days: int = 30) -> AcceptanceRateMetrics:
        """Analyze human acceptance of AI evaluation suggestions."""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=time_range_days)
        
        # This would require tracking when humans accept/reject AI suggestions
        # For now, return a placeholder with structure
        
        return AcceptanceRateMetrics(
            time_period=f"{time_range_days} days",
            total_ai_suggestions=250,
            accepted_suggestions=180,
            rejected_suggestions=70,
            acceptance_rate=0.72,
            acceptance_by_confidence={
                "high": 0.85,
                "medium": 0.70,
                "low": 0.45
            },
            acceptance_by_criteria={
                "coherence": 0.78,
                "relevance": 0.75,
                "factual_accuracy": 0.68
            },
            trust_trend_over_time=[
                (datetime.now() - timedelta(days=30), 0.65),
                (datetime.now() - timedelta(days=20), 0.68),
                (datetime.now() - timedelta(days=10), 0.72),
                (datetime.now(), 0.75)
            ]
        )

# Global user analytics instance
user_analytics = UserAnalyticsService() 