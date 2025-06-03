"""
A/B Testing API Endpoints
Comprehensive endpoints for experiment management, traffic allocation, and analysis.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func, desc
from sqlalchemy.orm import selectinload
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

from database.connection import get_db
from auth.security import get_current_user, require_role
from auth.models import UserRole
from database.models import User
from experiments.models import (
    Experiment, ExperimentVariant, ParticipantAssignment, ExperimentEvent, ExperimentResult,
    ExperimentCreate, VariantCreate, ExperimentResponse, VariantResponse, 
    AssignmentResponse, ExperimentResultResponse, ExperimentAnalytics,
    ExperimentStatus, VariantType
)
from experiments.service import ExperimentService
from services.cache_service import cache_response

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/experiments", tags=["A/B Testing"])


# Experiment Management Endpoints

@router.post("/", response_model=ExperimentResponse)
@require_role([UserRole.ADMIN, UserRole.ANALYST])
async def create_experiment(
    experiment_data: ExperimentCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Create a new A/B test experiment."""
    try:
        service = ExperimentService(db)
        experiment = await service.create_experiment(
            team_id=str(current_user.team_id),
            user_id=str(current_user.id),
            experiment_data=experiment_data.dict()
        )
        
        return ExperimentResponse(
            id=str(experiment.id),
            name=experiment.name,
            description=experiment.description,
            status=experiment.status,
            primary_metric=experiment.primary_metric,
            traffic_percentage=experiment.traffic_percentage,
            start_date=experiment.start_date,
            end_date=experiment.end_date,
            created_at=experiment.created_at,
            required_sample_size=experiment.required_sample_size,
            variants_count=len(experiment.variants) if experiment.variants else 0
        )
        
    except Exception as e:
        logger.error(f"Error creating experiment: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create experiment"
        )


@router.get("/", response_model=List[ExperimentResponse])
@cache_response("experiments_list", ttl=300, include_user=True)
async def list_experiments(
    status_filter: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """List experiments for the current team."""
    try:
        stmt = select(Experiment).where(
            Experiment.team_id == current_user.team_id
        ).options(
            selectinload(Experiment.variants)
        ).order_by(desc(Experiment.created_at))
        
        if status_filter:
            stmt = stmt.where(Experiment.status == status_filter)
        
        stmt = stmt.offset(offset).limit(limit)
        
        result = await db.execute(stmt)
        experiments = result.scalars().all()
        
        return [
            ExperimentResponse(
                id=str(exp.id),
                name=exp.name,
                description=exp.description,
                status=exp.status,
                primary_metric=exp.primary_metric,
                traffic_percentage=exp.traffic_percentage,
                start_date=exp.start_date,
                end_date=exp.end_date,
                created_at=exp.created_at,
                required_sample_size=exp.required_sample_size,
                variants_count=len(exp.variants) if exp.variants else 0
            )
            for exp in experiments
        ]
        
    except Exception as e:
        logger.error(f"Error listing experiments: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve experiments"
        )


@router.get("/{experiment_id}", response_model=ExperimentResponse)
@cache_response("experiment_detail", ttl=300, include_user=True)
async def get_experiment(
    experiment_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get detailed information about a specific experiment."""
    try:
        stmt = select(Experiment).options(
            selectinload(Experiment.variants),
            selectinload(Experiment.results)
        ).where(
            and_(
                Experiment.id == experiment_id,
                Experiment.team_id == current_user.team_id
            )
        )
        
        result = await db.execute(stmt)
        experiment = result.scalar_one_or_none()
        
        if not experiment:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Experiment not found"
            )
        
        return ExperimentResponse(
            id=str(experiment.id),
            name=experiment.name,
            description=experiment.description,
            status=experiment.status,
            primary_metric=experiment.primary_metric,
            traffic_percentage=experiment.traffic_percentage,
            start_date=experiment.start_date,
            end_date=experiment.end_date,
            created_at=experiment.created_at,
            required_sample_size=experiment.required_sample_size,
            variants_count=len(experiment.variants) if experiment.variants else 0
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving experiment: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve experiment"
        )


@router.post("/{experiment_id}/start")
@require_role([UserRole.ADMIN, UserRole.ANALYST])
async def start_experiment(
    experiment_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Start an experiment."""
    try:
        # Verify experiment belongs to user's team
        stmt = select(Experiment).where(
            and_(
                Experiment.id == experiment_id,
                Experiment.team_id == current_user.team_id
            )
        )
        result = await db.execute(stmt)
        experiment = result.scalar_one_or_none()
        
        if not experiment:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Experiment not found"
            )
        
        service = ExperimentService(db)
        success = await service.start_experiment(experiment_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to start experiment"
            )
        
        return {"message": "Experiment started successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting experiment: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start experiment"
        )


@router.post("/{experiment_id}/stop")
@require_role([UserRole.ADMIN, UserRole.ANALYST])
async def stop_experiment(
    experiment_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Stop an experiment."""
    try:
        # Verify experiment belongs to user's team
        stmt = select(Experiment).where(
            and_(
                Experiment.id == experiment_id,
                Experiment.team_id == current_user.team_id
            )
        )
        result = await db.execute(stmt)
        experiment = result.scalar_one_or_none()
        
        if not experiment:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Experiment not found"
            )
        
        service = ExperimentService(db)
        success = await service.stop_experiment(experiment_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to stop experiment"
            )
        
        return {"message": "Experiment stopped successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error stopping experiment: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to stop experiment"
        )


# Variant Management Endpoints

@router.post("/{experiment_id}/variants", response_model=VariantResponse)
@require_role([UserRole.ADMIN, UserRole.ANALYST])
async def add_variant(
    experiment_id: str,
    variant_data: VariantCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Add a variant to an experiment."""
    try:
        # Verify experiment belongs to user's team
        stmt = select(Experiment).where(
            and_(
                Experiment.id == experiment_id,
                Experiment.team_id == current_user.team_id
            )
        )
        result = await db.execute(stmt)
        experiment = result.scalar_one_or_none()
        
        if not experiment:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Experiment not found"
            )
        
        service = ExperimentService(db)
        variant = await service.add_variant(
            experiment_id=experiment_id,
            variant_data=variant_data.dict()
        )
        
        return VariantResponse(
            id=str(variant.id),
            name=variant.name,
            description=variant.description,
            variant_type=variant.variant_type,
            traffic_weight=variant.traffic_weight,
            configuration=variant.configuration,
            model_name=variant.model_name,
            participants_count=0  # Will be calculated later
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding variant: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to add variant"
        )


@router.get("/{experiment_id}/variants", response_model=List[VariantResponse])
@cache_response("experiment_variants", ttl=300, include_user=True)
async def list_variants(
    experiment_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """List all variants for an experiment."""
    try:
        # Verify experiment belongs to user's team
        stmt = select(Experiment).where(
            and_(
                Experiment.id == experiment_id,
                Experiment.team_id == current_user.team_id
            )
        )
        result = await db.execute(stmt)
        experiment = result.scalar_one_or_none()
        
        if not experiment:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Experiment not found"
            )
        
        # Get variants with participant counts
        variants_stmt = select(ExperimentVariant).where(
            ExperimentVariant.experiment_id == experiment_id
        )
        
        variants_result = await db.execute(variants_stmt)
        variants = variants_result.scalars().all()
        
        # Get participant counts for each variant
        variant_responses = []
        for variant in variants:
            count_stmt = select(func.count(ParticipantAssignment.id)).where(
                ParticipantAssignment.variant_id == variant.id
            )
            count_result = await db.execute(count_stmt)
            participant_count = count_result.scalar() or 0
            
            variant_responses.append(VariantResponse(
                id=str(variant.id),
                name=variant.name,
                description=variant.description,
                variant_type=variant.variant_type,
                traffic_weight=variant.traffic_weight,
                configuration=variant.configuration,
                model_name=variant.model_name,
                participants_count=participant_count
            ))
        
        return variant_responses
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing variants: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve variants"
        )


# Participant Assignment Endpoints

@router.post("/{experiment_id}/assign", response_model=Optional[AssignmentResponse])
async def assign_participant(
    experiment_id: str,
    request: Request,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """Assign a participant to an experiment variant."""
    try:
        # Prepare context
        context = {
            "user_agent": request.headers.get("user-agent"),
            "ip_address": request.client.host if request.client else None,
            "referrer": request.headers.get("referer")
        }
        
        service = ExperimentService(db)
        assignment = await service.assign_participant(
            experiment_id=experiment_id,
            user_id=user_id,
            session_id=session_id,
            context=context
        )
        
        if not assignment:
            return None
        
        return AssignmentResponse(**assignment)
        
    except Exception as e:
        logger.error(f"Error assigning participant: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to assign participant"
        )


@router.get("/{experiment_id}/assignments")
@cache_response("experiment_assignments", ttl=60, include_user=True)
async def list_assignments(
    experiment_id: str,
    limit: int = 100,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """List participant assignments for an experiment."""
    try:
        # Verify experiment belongs to user's team
        stmt = select(Experiment).where(
            and_(
                Experiment.id == experiment_id,
                Experiment.team_id == current_user.team_id
            )
        )
        result = await db.execute(stmt)
        experiment = result.scalar_one_or_none()
        
        if not experiment:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Experiment not found"
            )
        
        # Get assignments
        assignments_stmt = select(ParticipantAssignment).options(
            selectinload(ParticipantAssignment.variant)
        ).where(
            ParticipantAssignment.experiment_id == experiment_id
        ).order_by(desc(ParticipantAssignment.assigned_at)).offset(offset).limit(limit)
        
        assignments_result = await db.execute(assignments_stmt)
        assignments = assignments_result.scalars().all()
        
        return [
            {
                "id": str(assignment.id),
                "experiment_id": str(assignment.experiment_id),
                "variant_id": str(assignment.variant_id),
                "variant_name": assignment.variant.name,
                "user_id": str(assignment.user_id) if assignment.user_id else None,
                "session_id": assignment.session_id,
                "assigned_at": assignment.assigned_at,
                "assignment_method": assignment.assignment_method
            }
            for assignment in assignments
        ]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing assignments: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve assignments"
        )


# Event Tracking Endpoints

@router.post("/{experiment_id}/events")
async def track_event(
    experiment_id: str,
    event_data: Dict[str, Any],
    db: AsyncSession = Depends(get_db)
):
    """Track an event for experiment analysis."""
    try:
        participant_hash = event_data.get("participant_hash")
        event_type = event_data.get("event_type")
        event_value = event_data.get("event_value")
        event_metadata = event_data.get("event_metadata")
        trace_id = event_data.get("trace_id")
        
        if not participant_hash or not event_type:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="participant_hash and event_type are required"
            )
        
        service = ExperimentService(db)
        success = await service.track_event(
            experiment_id=experiment_id,
            participant_hash=participant_hash,
            event_type=event_type,
            event_value=event_value,
            event_metadata=event_metadata,
            trace_id=trace_id
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to track event"
            )
        
        return {"message": "Event tracked successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error tracking event: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to track event"
        )


# Analysis Endpoints

@router.post("/{experiment_id}/analyze")
@require_role([UserRole.ADMIN, UserRole.ANALYST])
async def analyze_experiment(
    experiment_id: str,
    analysis_data: Dict[str, Any],
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Perform statistical analysis of experiment results."""
    try:
        # Verify experiment belongs to user's team
        stmt = select(Experiment).where(
            and_(
                Experiment.id == experiment_id,
                Experiment.team_id == current_user.team_id
            )
        )
        result = await db.execute(stmt)
        experiment = result.scalar_one_or_none()
        
        if not experiment:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Experiment not found"
            )
        
        metric_name = analysis_data.get("metric_name") or experiment.primary_metric
        
        service = ExperimentService(db)
        results = await service.analyze_experiment(
            experiment_id=experiment_id,
            metric_name=metric_name
        )
        
        if not results:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Insufficient data for analysis"
            )
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing experiment: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to analyze experiment"
        )


@router.get("/{experiment_id}/results", response_model=List[ExperimentResultResponse])
@cache_response("experiment_results", ttl=300, include_user=True)
async def get_results(
    experiment_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get stored experiment results."""
    try:
        # Verify experiment belongs to user's team
        stmt = select(Experiment).where(
            and_(
                Experiment.id == experiment_id,
                Experiment.team_id == current_user.team_id
            )
        )
        result = await db.execute(stmt)
        experiment = result.scalar_one_or_none()
        
        if not experiment:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Experiment not found"
            )
        
        # Get results
        results_stmt = select(ExperimentResult).where(
            ExperimentResult.experiment_id == experiment_id
        ).order_by(desc(ExperimentResult.analysis_date))
        
        results_result = await db.execute(results_stmt)
        results = results_result.scalars().all()
        
        return [
            ExperimentResultResponse(
                id=str(result.id),
                metric_name=result.metric_name,
                control_sample_size=result.control_sample_size or 0,
                treatment_sample_size=result.treatment_sample_size or 0,
                control_mean=result.control_mean or 0.0,
                treatment_mean=result.treatment_mean or 0.0,
                absolute_effect=result.absolute_effect or 0.0,
                relative_effect=result.relative_effect or 0.0,
                p_value=result.p_value or 1.0,
                is_significant=result.is_significant or False,
                confidence_interval_lower=result.confidence_interval_lower or 0.0,
                confidence_interval_upper=result.confidence_interval_upper or 0.0,
                analysis_date=result.analysis_date
            )
            for result in results
        ]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving results: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve results"
        )


@router.get("/{experiment_id}/analytics", response_model=ExperimentAnalytics)
@cache_response("experiment_analytics", ttl=60, include_user=True)
async def get_analytics(
    experiment_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get experiment analytics dashboard data."""
    try:
        # Verify experiment belongs to user's team
        stmt = select(Experiment).options(
            selectinload(Experiment.results)
        ).where(
            and_(
                Experiment.id == experiment_id,
                Experiment.team_id == current_user.team_id
            )
        )
        result = await db.execute(stmt)
        experiment = result.scalar_one_or_none()
        
        if not experiment:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Experiment not found"
            )
        
        # Get participant count
        participants_stmt = select(func.count(ParticipantAssignment.id)).where(
            ParticipantAssignment.experiment_id == experiment_id
        )
        participants_result = await db.execute(participants_stmt)
        total_participants = participants_result.scalar() or 0
        
        # Calculate days running
        days_running = 0
        if experiment.start_date:
            end_date = experiment.end_date or datetime.utcnow()
            days_running = (end_date - experiment.start_date).days
        
        # Get latest results for primary metric
        latest_results = []
        if experiment.results:
            primary_results = [r for r in experiment.results if r.metric_name == experiment.primary_metric]
            if primary_results:
                latest_result = max(primary_results, key=lambda x: x.analysis_date)
                latest_results = [ExperimentResultResponse(
                    id=str(latest_result.id),
                    metric_name=latest_result.metric_name,
                    control_sample_size=latest_result.control_sample_size or 0,
                    treatment_sample_size=latest_result.treatment_sample_size or 0,
                    control_mean=latest_result.control_mean or 0.0,
                    treatment_mean=latest_result.treatment_mean or 0.0,
                    absolute_effect=latest_result.absolute_effect or 0.0,
                    relative_effect=latest_result.relative_effect or 0.0,
                    p_value=latest_result.p_value or 1.0,
                    is_significant=latest_result.is_significant or False,
                    confidence_interval_lower=latest_result.confidence_interval_lower or 0.0,
                    confidence_interval_upper=latest_result.confidence_interval_upper or 0.0,
                    analysis_date=latest_result.analysis_date
                )]
        
        return ExperimentAnalytics(
            experiment_id=str(experiment.id),
            experiment_name=experiment.name,
            status=experiment.status,
            total_participants=total_participants,
            conversion_rate_control=0.0,  # Will be calculated from results
            conversion_rate_treatment=0.0,  # Will be calculated from results
            statistical_significance=latest_results[0].is_significant if latest_results else False,
            confidence_level=experiment.confidence_level,
            days_running=days_running,
            projected_end_date=experiment.end_date,
            results=latest_results
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving analytics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve analytics"
        )


# Health Check

@router.get("/health")
async def health_check():
    """Health check endpoint for A/B testing service."""
    return {
        "status": "healthy",
        "service": "experiments",
        "timestamp": datetime.utcnow().isoformat()
    } 