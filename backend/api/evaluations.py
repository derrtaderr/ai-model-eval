"""
API endpoints for human evaluation management.
Implements the Human Evaluation Dashboard backend for Task 4.
"""

from datetime import datetime
from typing import Dict, Any, Optional, List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, and_, or_, func
from sqlalchemy.orm import selectinload

from auth.security import get_current_user_email
from database.connection import get_db
from database.models import Evaluation, Trace, User, EvaluationStatus


router = APIRouter()


class EvaluationCreate(BaseModel):
    """Schema for creating a new human evaluation."""
    trace_id: str = Field(..., description="ID of the trace being evaluated")
    label: str = Field(..., description="Evaluation label: accepted, rejected, or in_review")
    score: Optional[float] = Field(None, ge=0, le=1, description="Numeric score between 0 and 1")
    critique: Optional[str] = Field(None, description="Detailed feedback and notes")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional evaluation metadata")


class EvaluationUpdate(BaseModel):
    """Schema for updating an existing evaluation."""
    label: Optional[str] = Field(None, description="Updated evaluation label")
    score: Optional[float] = Field(None, ge=0, le=1, description="Updated numeric score")
    critique: Optional[str] = Field(None, description="Updated feedback and notes")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Updated metadata")


class EvaluationResponse(BaseModel):
    """Schema for evaluation response."""
    id: str
    trace_id: str
    evaluator_type: str
    evaluator_id: Optional[str]
    evaluator_email: Optional[str]
    score: Optional[float]
    label: Optional[str]
    critique: Optional[str]
    metadata: Optional[Dict[str, Any]]
    evaluated_at: str


class TraceWithEvaluations(BaseModel):
    """Schema for trace with its evaluations."""
    id: str
    timestamp: str
    user_input: str
    model_output: str
    model_name: str
    system_prompt: Optional[str]
    session_id: Optional[str]
    trace_metadata: Optional[Dict[str, Any]]
    latency_ms: Optional[int]
    token_count: Optional[Dict[str, int]]
    cost_usd: Optional[float]
    status: str
    evaluations: List[EvaluationResponse]
    human_evaluation_status: str  # "pending", "accepted", "rejected", "in_review"


class EvaluationStats(BaseModel):
    """Schema for evaluation statistics."""
    total_traces: int
    evaluated_traces: int
    pending_traces: int
    accepted_traces: int
    rejected_traces: int
    in_review_traces: int
    evaluation_rate: float
    acceptance_rate: float
    agreement_data: List[Dict[str, Any]]


@router.post("/evaluations", response_model=Dict[str, str], status_code=status.HTTP_201_CREATED)
async def create_evaluation(
    evaluation_data: EvaluationCreate,
    current_user_email: str = Depends(get_current_user_email),
    db: AsyncSession = Depends(get_db)
):
    """
    Submit a human evaluation for a trace.
    
    This endpoint allows human evaluators to accept, reject, or mark traces for review.
    """
    try:
        # Validate trace exists
        trace_uuid = UUID(evaluation_data.trace_id)
        trace_query = select(Trace).where(Trace.id == trace_uuid)
        result = await db.execute(trace_query)
        trace = result.scalar_one_or_none()
        
        if not trace:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Trace not found"
            )
        
        # Validate evaluation label
        valid_labels = ["accepted", "rejected", "in_review"]
        if evaluation_data.label not in valid_labels:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid label. Must be one of: {valid_labels}"
            )
        
        # For now, we'll use a placeholder user_id since user management isn't fully implemented
        # In production, you'd look up the user by email
        evaluator_id = None  # TODO: Implement user lookup by email
        
        # Check if evaluation already exists for this trace by this evaluator
        existing_eval_query = select(Evaluation).where(
            and_(
                Evaluation.trace_id == trace_uuid,
                Evaluation.evaluator_type == "human",
                Evaluation.evaluator_id == evaluator_id
            )
        )
        result = await db.execute(existing_eval_query)
        existing_eval = result.scalar_one_or_none()
        
        if existing_eval:
            # Update existing evaluation
            existing_eval.score = evaluation_data.score
            existing_eval.label = evaluation_data.label
            existing_eval.critique = evaluation_data.critique
            existing_eval.eval_metadata = evaluation_data.metadata
            existing_eval.evaluated_at = datetime.utcnow()
            
            await db.commit()
            await db.refresh(existing_eval)
            
            return {
                "evaluation_id": str(existing_eval.id),
                "message": "Evaluation updated successfully"
            }
        else:
            # Create new evaluation
            new_evaluation = Evaluation(
                trace_id=trace_uuid,
                evaluator_type="human",
                evaluator_id=evaluator_id,
                score=evaluation_data.score,
                label=evaluation_data.label,
                critique=evaluation_data.critique,
                eval_metadata=evaluation_data.metadata,
                evaluated_at=datetime.utcnow()
            )
            
            db.add(new_evaluation)
            await db.commit()
            await db.refresh(new_evaluation)
            
            return {
                "evaluation_id": str(new_evaluation.id),
                "message": "Evaluation created successfully"
            }
    
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid trace ID format"
        )
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create evaluation: {str(e)}"
        )


@router.get("/evaluations/traces", response_model=List[TraceWithEvaluations])
async def get_traces_for_evaluation(
    limit: int = Query(50, ge=1, le=200, description="Number of traces to return"),
    offset: int = Query(0, ge=0, description="Number of traces to skip"),
    status_filter: Optional[str] = Query(None, description="Filter by evaluation status"),
    model_name: Optional[str] = Query(None, description="Filter by model name"),
    current_user_email: str = Depends(get_current_user_email),
    db: AsyncSession = Depends(get_db)
):
    """
    Get traces with their evaluation status for the evaluation dashboard.
    
    Returns traces along with their human evaluations, optimized for the evaluation interface.
    """
    try:
        # Build base query with evaluations loaded
        query = select(Trace).options(selectinload(Trace.evaluations))
        
        # Apply filters
        if model_name:
            query = query.where(Trace.model_name == model_name)
        
        # Order by timestamp (newest first)
        query = query.order_by(Trace.timestamp.desc())
        
        # Apply pagination
        query = query.offset(offset).limit(limit)
        
        result = await db.execute(query)
        traces = result.scalars().all()
        
        # Process traces and determine evaluation status
        traces_with_evaluations = []
        for trace in traces:
            # Determine human evaluation status
            human_evaluations = [e for e in trace.evaluations if e.evaluator_type == "human"]
            
            if not human_evaluations:
                human_eval_status = "pending"
            else:
                # Use the most recent human evaluation
                latest_eval = max(human_evaluations, key=lambda e: e.evaluated_at)
                human_eval_status = latest_eval.label or "pending"
            
            # Filter by evaluation status if requested
            if status_filter and human_eval_status != status_filter.lower():
                continue
            
            # Format evaluations
            formatted_evaluations = []
            for eval in trace.evaluations:
                formatted_evaluations.append(EvaluationResponse(
                    id=str(eval.id),
                    trace_id=str(eval.trace_id),
                    evaluator_type=eval.evaluator_type,
                    evaluator_id=str(eval.evaluator_id) if eval.evaluator_id else None,
                    evaluator_email=None,  # TODO: Load from user relationship
                    score=eval.score,
                    label=eval.label,
                    critique=eval.critique,
                    metadata=eval.eval_metadata,
                    evaluated_at=eval.evaluated_at.isoformat()
                ))
            
            traces_with_evaluations.append(TraceWithEvaluations(
                id=str(trace.id),
                timestamp=trace.timestamp.isoformat(),
                user_input=trace.user_input,
                model_output=trace.model_output,
                model_name=trace.model_name,
                system_prompt=trace.system_prompt,
                session_id=trace.session_id,
                trace_metadata=trace.trace_metadata,
                latency_ms=trace.latency_ms,
                token_count=trace.token_count,
                cost_usd=trace.cost_usd,
                status=trace.status,
                evaluations=formatted_evaluations,
                human_evaluation_status=human_eval_status
            ))
        
        return traces_with_evaluations
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve traces for evaluation: {str(e)}"
        )


@router.get("/evaluations/stats", response_model=EvaluationStats)
async def get_evaluation_statistics(
    current_user_email: str = Depends(get_current_user_email),
    db: AsyncSession = Depends(get_db)
):
    """
    Get evaluation statistics for the analytics dashboard.
    
    Returns metrics like evaluation rates, acceptance rates, and agreement data.
    """
    try:
        # Get total traces count
        total_traces_query = select(func.count(Trace.id))
        result = await db.execute(total_traces_query)
        total_traces = result.scalar() or 0
        
        # Get evaluation counts by status
        evaluation_stats_query = select(
            Evaluation.label,
            func.count(Evaluation.id).label('count')
        ).where(
            Evaluation.evaluator_type == "human"
        ).group_by(Evaluation.label)
        
        result = await db.execute(evaluation_stats_query)
        eval_counts = {row.label: row.count for row in result}
        
        # Calculate metrics
        evaluated_traces = sum(eval_counts.values())
        pending_traces = total_traces - evaluated_traces
        accepted_traces = eval_counts.get("accepted", 0)
        rejected_traces = eval_counts.get("rejected", 0)
        in_review_traces = eval_counts.get("in_review", 0)
        
        evaluation_rate = (evaluated_traces / total_traces * 100) if total_traces > 0 else 0
        acceptance_rate = (accepted_traces / evaluated_traces * 100) if evaluated_traces > 0 else 0
        
        # Generate mock agreement data for now (in production, this would compare model vs human evaluations)
        agreement_data = []
        
        return EvaluationStats(
            total_traces=total_traces,
            evaluated_traces=evaluated_traces,
            pending_traces=pending_traces,
            accepted_traces=accepted_traces,
            rejected_traces=rejected_traces,
            in_review_traces=in_review_traces,
            evaluation_rate=round(evaluation_rate, 2),
            acceptance_rate=round(acceptance_rate, 2),
            agreement_data=agreement_data
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve evaluation statistics: {str(e)}"
        )


@router.get("/evaluations/{evaluation_id}", response_model=EvaluationResponse)
async def get_evaluation(
    evaluation_id: str,
    current_user_email: str = Depends(get_current_user_email),
    db: AsyncSession = Depends(get_db)
):
    """
    Get a specific evaluation by its ID.
    """
    try:
        eval_uuid = UUID(evaluation_id)
        query = select(Evaluation).where(Evaluation.id == eval_uuid)
        result = await db.execute(query)
        evaluation = result.scalar_one_or_none()
        
        if not evaluation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Evaluation not found"
            )
        
        return EvaluationResponse(
            id=str(evaluation.id),
            trace_id=str(evaluation.trace_id),
            evaluator_type=evaluation.evaluator_type,
            evaluator_id=str(evaluation.evaluator_id) if evaluation.evaluator_id else None,
            evaluator_email=None,  # TODO: Load from user relationship
            score=evaluation.score,
            label=evaluation.label,
            critique=evaluation.critique,
            metadata=evaluation.eval_metadata,
            evaluated_at=evaluation.evaluated_at.isoformat()
        )
    
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid evaluation ID format"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve evaluation: {str(e)}"
        )


@router.put("/evaluations/{evaluation_id}", response_model=Dict[str, str])
async def update_evaluation(
    evaluation_id: str,
    evaluation_update: EvaluationUpdate,
    current_user_email: str = Depends(get_current_user_email),
    db: AsyncSession = Depends(get_db)
):
    """
    Update an existing evaluation.
    """
    try:
        eval_uuid = UUID(evaluation_id)
        
        # Validate label if provided
        if evaluation_update.label:
            valid_labels = ["accepted", "rejected", "in_review"]
            if evaluation_update.label not in valid_labels:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid label. Must be one of: {valid_labels}"
                )
        
        # Build update dictionary
        update_data = {}
        if evaluation_update.label is not None:
            update_data["label"] = evaluation_update.label
        if evaluation_update.score is not None:
            update_data["score"] = evaluation_update.score
        if evaluation_update.critique is not None:
            update_data["critique"] = evaluation_update.critique
        if evaluation_update.metadata is not None:
            update_data["eval_metadata"] = evaluation_update.metadata
        
        if update_data:
            update_data["evaluated_at"] = datetime.utcnow()
            
            query = update(Evaluation).where(Evaluation.id == eval_uuid).values(**update_data)
            result = await db.execute(query)
            
            if result.rowcount == 0:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Evaluation not found"
                )
            
            await db.commit()
        
        return {
            "evaluation_id": evaluation_id,
            "message": "Evaluation updated successfully"
        }
    
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid evaluation ID format"
        )
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update evaluation: {str(e)}"
        )


@router.delete("/evaluations/{evaluation_id}", response_model=Dict[str, str])
async def delete_evaluation(
    evaluation_id: str,
    current_user_email: str = Depends(get_current_user_email),
    db: AsyncSession = Depends(get_db)
):
    """
    Delete an evaluation.
    """
    try:
        eval_uuid = UUID(evaluation_id)
        
        query = select(Evaluation).where(Evaluation.id == eval_uuid)
        result = await db.execute(query)
        evaluation = result.scalar_one_or_none()
        
        if not evaluation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Evaluation not found"
            )
        
        await db.delete(evaluation)
        await db.commit()
        
        return {
            "evaluation_id": evaluation_id,
            "message": "Evaluation deleted successfully"
        }
    
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid evaluation ID format"
        )
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete evaluation: {str(e)}"
        ) 