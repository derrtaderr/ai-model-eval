from fastapi import APIRouter, HTTPException, Request, BackgroundTasks, Depends, Header
from pydantic import BaseModel, Field, validator
from typing import Dict, Any, Optional, List
import asyncio
import json
import hashlib
import hmac
import time
from datetime import datetime

from database.connection import get_db
from services.cache_service import cache_service
from database.models import Trace

router = APIRouter()

class WebhookTrace(BaseModel):
    """Model for incoming webhook trace data"""
    trace_id: str = Field(..., description="Unique identifier for the trace")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    model_name: str = Field(..., description="Name of the AI model")
    user_query: str = Field(..., description="User's input query")
    system_prompt: Optional[str] = Field(None, description="System prompt used")
    ai_response: str = Field(..., description="AI model's response")
    functions_called: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tokens_used: Optional[int] = Field(None, ge=0)
    response_time_ms: Optional[int] = Field(None, ge=0)
    cost: Optional[float] = Field(None, ge=0)
    
    @validator('trace_id')
    def validate_trace_id(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('trace_id cannot be empty')
        return v.strip()

class WebhookBatch(BaseModel):
    """Model for batch webhook submissions"""
    traces: List[WebhookTrace] = Field(..., min_items=1, max_items=100)
    batch_id: Optional[str] = Field(None)
    source: str = Field(..., description="Source system identifier")

class WebhookResponse(BaseModel):
    """Standard webhook response"""
    success: bool
    message: str
    processed_count: Optional[int] = None
    errors: Optional[List[str]] = None
    trace_ids: Optional[List[str]] = None

async def verify_webhook_signature(request: Request, signature: Optional[str] = Header(None)):
    """Verify webhook signature for security"""
    if not signature:
        return True  # For now, signatures are optional
    
    # In production, implement HMAC verification
    # secret = os.getenv("WEBHOOK_SECRET")
    # if secret:
    #     body = await request.body()
    #     expected_signature = hmac.new(
    #         secret.encode(), body, hashlib.sha256
    #     ).hexdigest()
    #     return hmac.compare_digest(f"sha256={expected_signature}", signature)
    return True

async def process_trace_async(trace_data: WebhookTrace, db_connection):
    """Process a single trace asynchronously"""
    try:
        # Convert to our internal trace format
        trace = Trace(
            id=trace_data.trace_id,
            timestamp=trace_data.timestamp.isoformat(),
            model_name=trace_data.model_name,
            user_query=trace_data.user_query,
            system_prompt=trace_data.system_prompt or "",
            ai_response=trace_data.ai_response,
            functions_called=trace_data.functions_called,
            metadata=trace_data.metadata,
            tokens_used=trace_data.tokens_used,
            response_time_ms=trace_data.response_time_ms,
            cost=trace_data.cost,
            evaluation_status="pending"
        )
        
        # Store in database
        cursor = db_connection.cursor()
        cursor.execute("""
            INSERT INTO traces (
                id, timestamp, model_name, user_query, system_prompt, 
                ai_response, functions_called, metadata, tokens_used, 
                response_time_ms, cost, evaluation_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trace.id, trace.timestamp, trace.model_name, trace.user_query,
            trace.system_prompt, trace.ai_response, json.dumps(trace.functions_called),
            json.dumps(trace.metadata), trace.tokens_used, trace.response_time_ms,
            trace.cost, trace.evaluation_status
        ))
        db_connection.commit()
        
        # Cache for real-time updates
        await cache_service.set(f"trace:{trace.id}", trace.dict(), expire=3600)
        
        # Publish real-time update
        await cache_service.publish("trace_updates", {
            "action": "new_trace",
            "trace_id": trace.id,
            "timestamp": trace.timestamp
        })
        
        return trace.id
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process trace: {str(e)}")

@router.post("/webhook/trace", response_model=WebhookResponse)
async def receive_trace_webhook(
    trace_data: WebhookTrace,
    background_tasks: BackgroundTasks,
    request: Request,
    signature_valid: bool = Depends(verify_webhook_signature),
    db_connection = Depends(get_db)
):
    """Receive a single trace via webhook"""
    if not signature_valid:
        raise HTTPException(status_code=401, detail="Invalid webhook signature")
    
    try:
        # Process trace in background
        trace_id = await process_trace_async(trace_data, db_connection)
        
        return WebhookResponse(
            success=True,
            message="Trace received and processed successfully",
            trace_ids=[trace_id]
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/webhook/batch", response_model=WebhookResponse)
async def receive_batch_webhook(
    batch_data: WebhookBatch,
    background_tasks: BackgroundTasks,
    request: Request,
    signature_valid: bool = Depends(verify_webhook_signature),
    db_connection = Depends(get_db)
):
    """Receive multiple traces via webhook batch"""
    if not signature_valid:
        raise HTTPException(status_code=401, detail="Invalid webhook signature")
    
    processed_traces = []
    errors = []
    
    try:
        for trace_data in batch_data.traces:
            try:
                trace_id = await process_trace_async(trace_data, db_connection)
                processed_traces.append(trace_id)
            except Exception as e:
                errors.append(f"Failed to process trace {trace_data.trace_id}: {str(e)}")
        
        return WebhookResponse(
            success=len(processed_traces) > 0,
            message=f"Processed {len(processed_traces)} of {len(batch_data.traces)} traces",
            processed_count=len(processed_traces),
            errors=errors if errors else None,
            trace_ids=processed_traces
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")

@router.get("/webhook/health")
async def webhook_health():
    """Health check for webhook endpoints"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "endpoints": [
            "/webhook/trace",
            "/webhook/batch"
        ]
    }

@router.get("/webhook/stats")
async def webhook_stats(db_connection = Depends(get_db)):
    """Get webhook processing statistics"""
    try:
        cursor = db_connection.cursor()
        
        # Get recent trace counts
        cursor.execute("""
            SELECT 
                COUNT(*) as total_traces,
                COUNT(CASE WHEN DATE(timestamp) = DATE('now') THEN 1 END) as today_traces,
                COUNT(CASE WHEN evaluation_status = 'pending' THEN 1 END) as pending_traces
            FROM traces
        """)
        
        stats = cursor.fetchone()
        
        return {
            "total_traces": stats[0],
            "today_traces": stats[1], 
            "pending_evaluation": stats[2],
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}") 