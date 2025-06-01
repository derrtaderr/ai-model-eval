"""
External Tool Integration API for LLM Evaluation Platform.
Provides RESTful endpoints for third-party tool integration with authentication,
rate limiting, and comprehensive documentation.
"""

import hashlib
import hmac
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union
from uuid import uuid4

from fastapi import APIRouter, Request, HTTPException, Depends, Header, Query, Path
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, ConfigDict, validator
from sqlalchemy import select, func, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

# Simple rate limiting implementation (can be replaced with Redis in production)
from functools import wraps
import asyncio

from database.connection import get_db
from database.models import Trace, Evaluation, User
from auth.security import get_current_user_email

logger = logging.getLogger(__name__)
router = APIRouter()

# Security scheme
security = HTTPBearer()

# Simple rate limiting decorator (replace with Redis-based solution in production)
def rate_limit(requests_per_hour: int):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Rate limiting is handled in get_api_key_from_header and check_rate_limit functions
            return await func(*args, **kwargs)
        return wrapper
    return decorator

# ============================================================================
# API KEY MANAGEMENT & AUTHENTICATION
# ============================================================================

class APIKey(BaseModel):
    """API Key model for external authentication."""
    model_config = ConfigDict(protected_namespaces=())
    
    id: str = Field(..., description="Unique API key identifier")
    key_hash: str = Field(..., description="Hashed API key")
    name: str = Field(..., description="Human-readable name for the key")
    user_email: str = Field(..., description="Owner email address")
    tier: str = Field("free", description="API tier (free, premium, enterprise)")
    rate_limit: int = Field(100, description="Requests per hour")
    is_active: bool = Field(True, description="Whether the key is active")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_used_at: Optional[datetime] = Field(None, description="Last usage timestamp")
    usage_count: int = Field(0, description="Total usage count")


class APIKeyCreate(BaseModel):
    """Schema for creating new API keys."""
    model_config = ConfigDict(protected_namespaces=())
    
    name: str = Field(..., description="Human-readable name for the key")
    tier: str = Field("free", description="API tier (free, premium, enterprise)")
    description: Optional[str] = Field(None, description="Key description")
    expires_in_days: Optional[int] = Field(None, ge=1, le=365, description="Expiration period")


class APIKeyResponse(BaseModel):
    """Schema for API key response."""
    model_config = ConfigDict(protected_namespaces=())
    
    id: str
    name: str
    key_preview: str = Field(..., description="First 8 characters of the key")
    tier: str
    rate_limit: int
    is_active: bool
    created_at: str
    last_used_at: Optional[str]
    usage_count: int
    usage_today: int


class APIKeyUsage(BaseModel):
    """Schema for API key usage tracking."""
    model_config = ConfigDict(protected_namespaces=())
    
    api_key_id: str
    endpoint: str
    method: str
    status_code: int
    response_time_ms: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    user_agent: Optional[str] = None
    ip_address: Optional[str] = None


# In-memory storage for demo (in production, use Redis or database)
API_KEYS = {}
API_USAGE = []

def generate_api_key() -> tuple[str, str]:
    """Generate a new API key and its hash."""
    key = f"llm-eval-{uuid4().hex[:24]}"
    key_hash = hashlib.sha256(key.encode()).hexdigest()
    return key, key_hash


def verify_api_key(api_key: str) -> Optional[APIKey]:
    """Verify an API key and return the key info if valid."""
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    
    for key_id, key_info in API_KEYS.items():
        if key_info["key_hash"] == key_hash and key_info["is_active"]:
            return APIKey(**key_info)
    
    return None


async def get_api_key_from_header(
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
) -> APIKey:
    """Extract and verify API key from request headers."""
    
    api_key = None
    
    # Try X-API-Key header first
    if x_api_key:
        api_key = x_api_key
    # Try Authorization header (Bearer token)
    elif authorization and authorization.startswith("Bearer "):
        api_key = authorization.replace("Bearer ", "")
    
    if not api_key:
        raise HTTPException(
            status_code=401, 
            detail="API key required. Use X-API-Key header or Authorization: Bearer <key>"
        )
    
    key_info = verify_api_key(api_key)
    if not key_info:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    # Update last used timestamp
    API_KEYS[key_info.id]["last_used_at"] = datetime.utcnow()
    API_KEYS[key_info.id]["usage_count"] += 1
    
    return key_info


def check_rate_limit(api_key: APIKey, request: Request) -> bool:
    """Check if the API key has exceeded its rate limit."""
    now = datetime.utcnow()
    hour_ago = now - timedelta(hours=1)
    
    # Count recent usage
    recent_usage = [
        usage for usage in API_USAGE
        if usage["api_key_id"] == api_key.id and usage["timestamp"] > hour_ago
    ]
    
    if len(recent_usage) >= api_key.rate_limit:
        return False
    
    return True


def log_api_usage(api_key: APIKey, request: Request, response_time: float, status_code: int):
    """Log API usage for analytics and rate limiting."""
    usage = {
        "api_key_id": api_key.id,
        "endpoint": str(request.url.path),
        "method": request.method,
        "status_code": status_code,
        "response_time_ms": response_time,
        "timestamp": datetime.utcnow(),
        "user_agent": request.headers.get("user-agent"),
        "ip_address": request.client.host if request.client else None
    }
    
    API_USAGE.append(usage)
    
    # Keep only last 24 hours of usage data
    day_ago = datetime.utcnow() - timedelta(days=1)
    global API_USAGE
    API_USAGE = [u for u in API_USAGE if u["timestamp"] > day_ago]


# ============================================================================
# EXTERNAL API SCHEMAS
# ============================================================================

class ExternalEvaluationRequest(BaseModel):
    """Schema for external evaluation requests."""
    model_config = ConfigDict(protected_namespaces=())
    
    user_input: str = Field(..., description="User input text to evaluate")
    model_output: str = Field(..., description="Model response to evaluate")
    model_name: str = Field(..., description="Name of the model that generated the response")
    system_prompt: Optional[str] = Field(None, description="System prompt used")
    criteria: List[str] = Field(default=["relevance", "coherence"], description="Evaluation criteria")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    reference_answer: Optional[str] = Field(None, description="Reference answer for comparison")
    session_id: Optional[str] = Field(None, description="Session identifier")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class ExternalEvaluationResponse(BaseModel):
    """Schema for external evaluation responses."""
    model_config = ConfigDict(protected_namespaces=())
    
    evaluation_id: str = Field(..., description="Unique evaluation identifier")
    trace_id: str = Field(..., description="Associated trace identifier")
    overall_score: float = Field(..., ge=0, le=1, description="Overall evaluation score")
    criteria_scores: Dict[str, float] = Field(..., description="Scores for each criterion")
    reasoning: str = Field(..., description="Detailed reasoning for the evaluation")
    confidence: float = Field(..., ge=0, le=1, description="Confidence in the evaluation")
    evaluator_model: str = Field(..., description="Model used for evaluation")
    evaluation_time_ms: int = Field(..., description="Time taken for evaluation")
    cost_usd: Optional[float] = Field(None, description="Cost of the evaluation")


class ExternalTraceFilter(BaseModel):
    """Schema for filtering traces in external API."""
    model_config = ConfigDict(protected_namespaces=())
    
    model_names: Optional[List[str]] = Field(None, description="Filter by model names")
    date_from: Optional[str] = Field(None, description="Start date (ISO format)")
    date_to: Optional[str] = Field(None, description="End date (ISO format)")
    min_score: Optional[float] = Field(None, ge=0, le=1, description="Minimum evaluation score")
    max_score: Optional[float] = Field(None, ge=0, le=1, description="Maximum evaluation score")
    session_ids: Optional[List[str]] = Field(None, description="Filter by session IDs")
    has_evaluation: Optional[bool] = Field(None, description="Filter traces with/without evaluations")
    limit: int = Field(50, ge=1, le=500, description="Maximum number of results")
    offset: int = Field(0, ge=0, description="Number of results to skip")


class ExternalTraceResponse(BaseModel):
    """Schema for external trace responses."""
    model_config = ConfigDict(protected_namespaces=())
    
    trace_id: str
    timestamp: str
    user_input: str
    model_output: str
    model_name: str
    system_prompt: Optional[str]
    session_id: Optional[str]
    evaluation_score: Optional[float]
    evaluation_status: str
    latency_ms: Optional[int]
    cost_usd: Optional[float]
    metadata: Optional[Dict[str, Any]]


class ExternalBatchRequest(BaseModel):
    """Schema for batch operation requests."""
    model_config = ConfigDict(protected_namespaces=())
    
    operation: str = Field(..., description="Batch operation type (evaluate, export, analyze)")
    items: List[Dict[str, Any]] = Field(..., description="Items to process")
    options: Optional[Dict[str, Any]] = Field(None, description="Operation options")
    callback_url: Optional[str] = Field(None, description="Webhook URL for completion notification")


class ExternalBatchResponse(BaseModel):
    """Schema for batch operation responses."""
    model_config = ConfigDict(protected_namespaces=())
    
    batch_id: str = Field(..., description="Unique batch identifier")
    status: str = Field(..., description="Batch status (queued, processing, completed, failed)")
    total_items: int = Field(..., description="Total number of items")
    completed_items: int = Field(..., description="Number of completed items")
    failed_items: int = Field(..., description="Number of failed items")
    estimated_completion: Optional[str] = Field(None, description="Estimated completion time")
    results_url: Optional[str] = Field(None, description="URL to download results")


class APIUsageStats(BaseModel):
    """Schema for API usage statistics."""
    model_config = ConfigDict(protected_namespaces=())
    
    api_key_id: str
    current_period_usage: int
    rate_limit: int
    usage_remaining: int
    reset_time: str
    total_usage_today: int
    total_usage_month: int
    top_endpoints: List[Dict[str, Any]]
    average_response_time_ms: float


# ============================================================================
# API KEY MANAGEMENT ENDPOINTS
# ============================================================================

@router.post("/api-keys", response_model=Dict[str, Any])
async def create_api_key(
    request: APIKeyCreate,
    current_user: str = Depends(get_current_user_email)
):
    """Create a new API key for external integrations."""
    
    try:
        # Generate new API key
        api_key, key_hash = generate_api_key()
        key_id = str(uuid4())
        
        # Set rate limits based on tier
        rate_limits = {
            "free": 100,
            "premium": 1000,
            "enterprise": 10000
        }
        
        # Store API key info
        api_key_info = {
            "id": key_id,
            "key_hash": key_hash,
            "name": request.name,
            "user_email": current_user,
            "tier": request.tier,
            "rate_limit": rate_limits.get(request.tier, 100),
            "is_active": True,
            "created_at": datetime.utcnow(),
            "last_used_at": None,
            "usage_count": 0
        }
        
        API_KEYS[key_id] = api_key_info
        
        return {
            "api_key": api_key,  # Only shown once
            "key_id": key_id,
            "name": request.name,
            "tier": request.tier,
            "rate_limit": api_key_info["rate_limit"],
            "message": "API key created successfully. Store it securely - it won't be shown again."
        }
        
    except Exception as e:
        logger.error(f"API key creation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create API key: {str(e)}")


@router.get("/api-keys", response_model=List[APIKeyResponse])
async def list_api_keys(
    current_user: str = Depends(get_current_user_email)
):
    """List all API keys for the current user."""
    
    user_keys = []
    for key_id, key_info in API_KEYS.items():
        if key_info["user_email"] == current_user:
            # Calculate usage today
            today = datetime.utcnow().date()
            usage_today = len([
                u for u in API_USAGE 
                if u["api_key_id"] == key_id and u["timestamp"].date() == today
            ])
            
            user_keys.append(APIKeyResponse(
                id=key_id,
                name=key_info["name"],
                key_preview=f"llm-eval-{key_info['key_hash'][:8]}...",
                tier=key_info["tier"],
                rate_limit=key_info["rate_limit"],
                is_active=key_info["is_active"],
                created_at=key_info["created_at"].isoformat(),
                last_used_at=key_info["last_used_at"].isoformat() if key_info["last_used_at"] else None,
                usage_count=key_info["usage_count"],
                usage_today=usage_today
            ))
    
    return user_keys


@router.delete("/api-keys/{key_id}")
async def revoke_api_key(
    key_id: str,
    current_user: str = Depends(get_current_user_email)
):
    """Revoke an API key."""
    
    if key_id not in API_KEYS:
        raise HTTPException(status_code=404, detail="API key not found")
    
    key_info = API_KEYS[key_id]
    if key_info["user_email"] != current_user:
        raise HTTPException(status_code=403, detail="Not authorized to revoke this key")
    
    # Mark as inactive instead of deleting for audit purposes
    API_KEYS[key_id]["is_active"] = False
    
    return {"message": "API key revoked successfully"}


# ============================================================================
# EXTERNAL API ENDPOINTS
# ============================================================================

@router.get("/evaluations", response_model=List[ExternalEvaluationResponse])
@rate_limit(100)
async def list_evaluations(
    request: Request,
    limit: int = Query(50, ge=1, le=500, description="Maximum number of results"),
    offset: int = Query(0, ge=0, description="Number of results to skip"),
    model_name: Optional[str] = Query(None, description="Filter by model name"),
    min_score: Optional[float] = Query(None, ge=0, le=1, description="Minimum score filter"),
    api_key: APIKey = Depends(get_api_key_from_header),
    db: AsyncSession = Depends(get_db)
):
    """List evaluations with optional filtering."""
    
    start_time = time.time()
    
    try:
        # Check rate limit
        if not check_rate_limit(api_key, request):
            raise HTTPException(
                status_code=429, 
                detail="Rate limit exceeded",
                headers={"Retry-After": "3600"}
            )
        
        # Build query
        query = select(Evaluation).options(selectinload(Evaluation.trace))
        
        if model_name:
            query = query.join(Trace).where(Trace.model_name == model_name)
        
        if min_score is not None:
            query = query.where(Evaluation.score >= min_score)
        
        query = query.offset(offset).limit(limit)
        
        # Execute query
        result = await db.execute(query)
        evaluations = result.scalars().all()
        
        # Format response
        response_data = []
        for eval in evaluations:
            response_data.append(ExternalEvaluationResponse(
                evaluation_id=str(eval.id),
                trace_id=str(eval.trace_id),
                overall_score=eval.score or 0.0,
                criteria_scores=eval.metadata.get("criteria_scores", {}) if eval.metadata else {},
                reasoning=eval.critique or "No reasoning provided",
                confidence=eval.metadata.get("confidence", 0.5) if eval.metadata else 0.5,
                evaluator_model=eval.evaluator_type,
                evaluation_time_ms=eval.metadata.get("evaluation_time_ms", 0) if eval.metadata else 0,
                cost_usd=eval.metadata.get("cost_usd") if eval.metadata else None
            ))
        
        # Log usage
        response_time = (time.time() - start_time) * 1000
        log_api_usage(api_key, request, response_time, 200)
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"External API error in list_evaluations: {str(e)}")
        response_time = (time.time() - start_time) * 1000
        log_api_usage(api_key, request, response_time, 500)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/evaluations", response_model=ExternalEvaluationResponse)
@rate_limit(50)
async def create_evaluation(
    request: Request,
    eval_request: ExternalEvaluationRequest,
    api_key: APIKey = Depends(get_api_key_from_header),
    db: AsyncSession = Depends(get_db)
):
    """Create a new evaluation for a given input-output pair."""
    
    start_time = time.time()
    
    try:
        # Check rate limit
        if not check_rate_limit(api_key, request):
            raise HTTPException(
                status_code=429, 
                detail="Rate limit exceeded",
                headers={"Retry-After": "3600"}
            )
        
        # Create trace first
        from database.models import Trace
        trace = Trace(
            user_input=eval_request.user_input,
            model_output=eval_request.model_output,
            model_name=eval_request.model_name,
            system_prompt=eval_request.system_prompt,
            session_id=eval_request.session_id,
            status="pending",
            metadata=eval_request.metadata or {}
        )
        
        db.add(trace)
        await db.flush()  # Get the trace ID
        
        # Mock evaluation (in real implementation, call evaluation service)
        overall_score = 0.85  # Mock score
        criteria_scores = {criterion: 0.8 + (hash(criterion) % 20) / 100 for criterion in eval_request.criteria}
        reasoning = f"Evaluated based on {', '.join(eval_request.criteria)}. Response shows good quality."
        confidence = 0.9
        
        # Create evaluation
        evaluation = Evaluation(
            trace_id=trace.id,
            evaluator_type="external-api",
            evaluator_id=api_key.id,
            evaluator_email=api_key.user_email,
            score=overall_score,
            label="accepted" if overall_score > 0.7 else "rejected",
            critique=reasoning,
            metadata={
                "criteria_scores": criteria_scores,
                "confidence": confidence,
                "evaluation_time_ms": 1500,
                "cost_usd": 0.002,
                "api_key_id": api_key.id
            }
        )
        
        db.add(evaluation)
        await db.commit()
        
        response = ExternalEvaluationResponse(
            evaluation_id=str(evaluation.id),
            trace_id=str(trace.id),
            overall_score=overall_score,
            criteria_scores=criteria_scores,
            reasoning=reasoning,
            confidence=confidence,
            evaluator_model="external-api-v1",
            evaluation_time_ms=1500,
            cost_usd=0.002
        )
        
        # Log usage
        response_time = (time.time() - start_time) * 1000
        log_api_usage(api_key, request, response_time, 201)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"External API error in create_evaluation: {str(e)}")
        response_time = (time.time() - start_time) * 1000
        log_api_usage(api_key, request, response_time, 500)
        raise HTTPException(status_code=500, detail=f"Failed to create evaluation: {str(e)}")


@router.get("/traces", response_model=List[ExternalTraceResponse])
@rate_limit(200)
async def list_traces(
    request: Request,
    filter_params: ExternalTraceFilter = Depends(),
    api_key: APIKey = Depends(get_api_key_from_header),
    db: AsyncSession = Depends(get_db)
):
    """List traces with optional filtering."""
    
    start_time = time.time()
    
    try:
        # Check rate limit
        if not check_rate_limit(api_key, request):
            raise HTTPException(
                status_code=429, 
                detail="Rate limit exceeded",
                headers={"Retry-After": "3600"}
            )
        
        # Build query
        query = select(Trace).options(selectinload(Trace.evaluations))
        
        # Apply filters
        conditions = []
        
        if filter_params.model_names:
            conditions.append(Trace.model_name.in_(filter_params.model_names))
        
        if filter_params.session_ids:
            conditions.append(Trace.session_id.in_(filter_params.session_ids))
        
        if filter_params.date_from:
            try:
                date_from = datetime.fromisoformat(filter_params.date_from.replace('Z', '+00:00'))
                conditions.append(Trace.timestamp >= date_from)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date_from format. Use ISO format.")
        
        if filter_params.date_to:
            try:
                date_to = datetime.fromisoformat(filter_params.date_to.replace('Z', '+00:00'))
                conditions.append(Trace.timestamp <= date_to)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date_to format. Use ISO format.")
        
        if conditions:
            query = query.where(and_(*conditions))
        
        query = query.offset(filter_params.offset).limit(filter_params.limit)
        
        # Execute query
        result = await db.execute(query)
        traces = result.scalars().all()
        
        # Format response
        response_data = []
        for trace in traces:
            # Get evaluation info
            evaluation = trace.evaluations[0] if trace.evaluations else None
            evaluation_score = evaluation.score if evaluation else None
            evaluation_status = evaluation.label if evaluation else "pending"
            
            # Apply score filters after database query (for simplicity)
            if filter_params.min_score is not None and (not evaluation_score or evaluation_score < filter_params.min_score):
                continue
            if filter_params.max_score is not None and evaluation_score and evaluation_score > filter_params.max_score:
                continue
            if filter_params.has_evaluation is not None and bool(evaluation) != filter_params.has_evaluation:
                continue
            
            response_data.append(ExternalTraceResponse(
                trace_id=str(trace.id),
                timestamp=trace.timestamp.isoformat(),
                user_input=trace.user_input,
                model_output=trace.model_output,
                model_name=trace.model_name,
                system_prompt=trace.system_prompt,
                session_id=trace.session_id,
                evaluation_score=evaluation_score,
                evaluation_status=evaluation_status,
                latency_ms=trace.latency_ms,
                cost_usd=trace.cost_usd,
                metadata=trace.metadata
            ))
        
        # Log usage
        response_time = (time.time() - start_time) * 1000
        log_api_usage(api_key, request, response_time, 200)
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"External API error in list_traces: {str(e)}")
        response_time = (time.time() - start_time) * 1000
        log_api_usage(api_key, request, response_time, 500)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/traces", response_model=Dict[str, Any])
@rate_limit(100)
async def submit_trace(
    request: Request,
    trace_data: Dict[str, Any],
    api_key: APIKey = Depends(get_api_key_from_header),
    db: AsyncSession = Depends(get_db)
):
    """Submit a new trace for evaluation."""
    
    start_time = time.time()
    
    try:
        # Check rate limit
        if not check_rate_limit(api_key, request):
            raise HTTPException(
                status_code=429, 
                detail="Rate limit exceeded",
                headers={"Retry-After": "3600"}
            )
        
        # Validate required fields
        required_fields = ["user_input", "model_output", "model_name"]
        for field in required_fields:
            if field not in trace_data:
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
        
        # Create trace
        trace = Trace(
            user_input=trace_data["user_input"],
            model_output=trace_data["model_output"],
            model_name=trace_data["model_name"],
            system_prompt=trace_data.get("system_prompt"),
            session_id=trace_data.get("session_id"),
            latency_ms=trace_data.get("latency_ms"),
            cost_usd=trace_data.get("cost_usd"),
            status="submitted",
            metadata=trace_data.get("metadata", {})
        )
        
        db.add(trace)
        await db.commit()
        
        response = {
            "trace_id": str(trace.id),
            "status": "submitted",
            "message": "Trace submitted successfully",
            "timestamp": trace.timestamp.isoformat()
        }
        
        # Log usage
        response_time = (time.time() - start_time) * 1000
        log_api_usage(api_key, request, response_time, 201)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"External API error in submit_trace: {str(e)}")
        response_time = (time.time() - start_time) * 1000
        log_api_usage(api_key, request, response_time, 500)
        raise HTTPException(status_code=500, detail=f"Failed to submit trace: {str(e)}")


@router.get("/models", response_model=Dict[str, Any])
@rate_limit(500)
async def list_available_models(
    request: Request,
    api_key: APIKey = Depends(get_api_key_from_header)
):
    """Get list of available models and their capabilities."""
    
    start_time = time.time()
    
    try:
        # Check rate limit
        if not check_rate_limit(api_key, request):
            raise HTTPException(
                status_code=429, 
                detail="Rate limit exceeded",
                headers={"Retry-After": "3600"}
            )
        
        models = {
            "evaluator_models": [
                {
                    "id": "gpt-4",
                    "name": "GPT-4",
                    "provider": "OpenAI",
                    "capabilities": ["text-evaluation", "reasoning", "scoring"],
                    "cost_per_1k_tokens": 0.03,
                    "max_tokens": 8192,
                    "recommended_for": ["high-quality-evaluation", "complex-reasoning"]
                },
                {
                    "id": "claude-3-sonnet",
                    "name": "Claude 3 Sonnet",
                    "provider": "Anthropic",
                    "capabilities": ["text-evaluation", "reasoning", "scoring", "safety"],
                    "cost_per_1k_tokens": 0.015,
                    "max_tokens": 100000,
                    "recommended_for": ["safety-evaluation", "long-context"]
                },
                {
                    "id": "gemini-pro",
                    "name": "Gemini Pro",
                    "provider": "Google",
                    "capabilities": ["text-evaluation", "multimodal", "reasoning"],
                    "cost_per_1k_tokens": 0.001,
                    "max_tokens": 32768,
                    "recommended_for": ["cost-effective", "multimodal-evaluation"]
                }
            ],
            "evaluation_criteria": [
                "relevance",
                "coherence",
                "accuracy",
                "completeness",
                "clarity",
                "safety",
                "helpfulness",
                "factuality",
                "creativity",
                "conciseness"
            ],
            "supported_formats": ["text", "json", "structured"],
            "api_version": "v1",
            "rate_limits": {
                "free": "100 requests/hour",
                "premium": "1000 requests/hour",
                "enterprise": "10000 requests/hour"
            }
        }
        
        # Log usage
        response_time = (time.time() - start_time) * 1000
        log_api_usage(api_key, request, response_time, 200)
        
        return models
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"External API error in list_available_models: {str(e)}")
        response_time = (time.time() - start_time) * 1000
        log_api_usage(api_key, request, response_time, 500)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/batch", response_model=ExternalBatchResponse)
@rate_limit(10)
async def create_batch_operation(
    request: Request,
    batch_request: ExternalBatchRequest,
    api_key: APIKey = Depends(get_api_key_from_header)
):
    """Create a batch operation for processing multiple items."""
    
    start_time = time.time()
    
    try:
        # Check rate limit
        if not check_rate_limit(api_key, request):
            raise HTTPException(
                status_code=429, 
                detail="Rate limit exceeded",
                headers={"Retry-After": "3600"}
            )
        
        # Validate batch size
        if len(batch_request.items) > 1000:
            raise HTTPException(status_code=400, detail="Batch size cannot exceed 1000 items")
        
        # Create batch job
        batch_id = str(uuid4())
        
        # Mock batch processing (in real implementation, use background queue)
        estimated_completion = datetime.utcnow() + timedelta(minutes=len(batch_request.items) * 2)
        
        response = ExternalBatchResponse(
            batch_id=batch_id,
            status="queued",
            total_items=len(batch_request.items),
            completed_items=0,
            failed_items=0,
            estimated_completion=estimated_completion.isoformat(),
            results_url=f"/api/external/batch/{batch_id}/results"
        )
        
        # Log usage
        response_time = (time.time() - start_time) * 1000
        log_api_usage(api_key, request, response_time, 202)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"External API error in create_batch_operation: {str(e)}")
        response_time = (time.time() - start_time) * 1000
        log_api_usage(api_key, request, response_time, 500)
        raise HTTPException(status_code=500, detail=f"Failed to create batch operation: {str(e)}")


@router.get("/usage", response_model=APIUsageStats)
@rate_limit(1000)
async def get_usage_statistics(
    request: Request,
    api_key: APIKey = Depends(get_api_key_from_header)
):
    """Get API usage statistics for the current API key."""
    
    start_time = time.time()
    
    try:
        now = datetime.utcnow()
        hour_ago = now - timedelta(hours=1)
        today = now.date()
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        # Calculate usage statistics
        hourly_usage = [u for u in API_USAGE if u["api_key_id"] == api_key.id and u["timestamp"] > hour_ago]
        daily_usage = [u for u in API_USAGE if u["api_key_id"] == api_key.id and u["timestamp"].date() == today]
        monthly_usage = [u for u in API_USAGE if u["api_key_id"] == api_key.id and u["timestamp"] > month_start]
        
        # Top endpoints
        endpoint_counts = {}
        total_response_time = 0
        for usage in daily_usage:
            endpoint = usage["endpoint"]
            endpoint_counts[endpoint] = endpoint_counts.get(endpoint, 0) + 1
            total_response_time += usage["response_time_ms"]
        
        top_endpoints = [
            {"endpoint": endpoint, "count": count}
            for endpoint, count in sorted(endpoint_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        ]
        
        # Calculate reset time (next hour)
        reset_time = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
        
        stats = APIUsageStats(
            api_key_id=api_key.id,
            current_period_usage=len(hourly_usage),
            rate_limit=api_key.rate_limit,
            usage_remaining=max(0, api_key.rate_limit - len(hourly_usage)),
            reset_time=reset_time.isoformat(),
            total_usage_today=len(daily_usage),
            total_usage_month=len(monthly_usage),
            top_endpoints=top_endpoints,
            average_response_time_ms=total_response_time / len(daily_usage) if daily_usage else 0.0
        )
        
        # Log usage
        response_time = (time.time() - start_time) * 1000
        log_api_usage(api_key, request, response_time, 200)
        
        return stats
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"External API error in get_usage_statistics: {str(e)}")
        response_time = (time.time() - start_time) * 1000
        log_api_usage(api_key, request, response_time, 500)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# ============================================================================
# HEALTH AND STATUS ENDPOINTS
# ============================================================================

@router.get("/health")
async def external_api_health():
    """Health check endpoint for external API."""
    
    return {
        "status": "healthy",
        "service": "External Integration API",
        "version": "v1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "endpoints": [
            "GET /api/external/evaluations",
            "POST /api/external/evaluations", 
            "GET /api/external/traces",
            "POST /api/external/traces",
            "GET /api/external/models",
            "POST /api/external/batch",
            "GET /api/external/usage"
        ],
        "authentication": "API Key required (X-API-Key header or Authorization: Bearer <key>)",
        "rate_limiting": "Varies by tier and endpoint",
        "documentation": "/docs#/External%20Integration%20API"
    } 