"""
LangSmith API endpoints for enhanced integration.
"""

from datetime import datetime
from typing import Dict, Any, Optional, List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Request, Header
from pydantic import BaseModel, Field

from auth.dependencies import get_current_user, require_permission
from database.models import User
from services.langsmith_connector import langsmith_connector, LangSmithSync


router = APIRouter(prefix="/langsmith", tags=["LangSmith Integration"])


class SyncRequest(BaseModel):
    """Request model for LangSmith sync operations."""
    project_name: Optional[str] = Field(None, description="LangSmith project to sync from")
    limit: int = Field(100, ge=1, le=1000, description="Maximum number of runs to sync")
    force_resync: bool = Field(False, description="Force re-sync of existing traces")


class SyncResponse(BaseModel):
    """Response model for sync operations."""
    status: str = Field(..., description="Sync operation status")
    project_name: str = Field(..., description="Project that was synced")
    total_synced: int = Field(..., description="Number of traces synced")
    last_sync: Optional[datetime] = Field(None, description="Last sync timestamp")
    errors: List[str] = Field(default_factory=list, description="Sync errors if any")


class EvaluationPushRequest(BaseModel):
    """Request model for pushing evaluations to LangSmith."""
    trace_id: UUID = Field(..., description="Local trace ID")
    score: Optional[float] = Field(None, description="Evaluation score")
    label: Optional[str] = Field(None, description="Evaluation label")
    critique: Optional[str] = Field(None, description="Evaluation critique")
    corrected_output: Optional[str] = Field(None, description="Corrected output if any")


class WebhookResponse(BaseModel):
    """Response model for webhook processing."""
    status: str = Field(..., description="Processing status")
    details: Dict[str, Any] = Field(default_factory=dict, description="Processing details")


@router.get("/status", response_model=Dict[str, Any])
async def get_langsmith_status(
    current_user: User = Depends(get_current_user)
):
    """
    Get LangSmith connection status and configuration.
    
    Returns current connection status, project info, and sync statistics.
    """
    try:
        status = await langsmith_connector.get_connection_status()
        sync_stats = await langsmith_connector.get_sync_stats()
        
        return {
            "connection": status,
            "sync_stats": sync_stats,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting LangSmith status: {str(e)}"
        )


@router.get("/projects", response_model=List[Dict[str, Any]])
async def get_langsmith_projects(
    current_user: User = Depends(get_current_user)
):
    """
    Get list of available LangSmith projects/datasets.
    
    Returns list of projects with metadata like name, description, and example count.
    """
    try:
        projects = await langsmith_connector.get_langsmith_projects()
        return projects
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching LangSmith projects: {str(e)}"
        )


@router.post("/sync", response_model=SyncResponse)
async def sync_from_langsmith(
    sync_request: SyncRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_permission("traces:write"))
):
    """
    Sync traces from LangSmith to the platform.
    
    Pulls recent traces from the specified LangSmith project and stores them locally.
    Supports incremental sync to avoid duplicates.
    """
    try:
        # Perform sync operation
        sync_result = await langsmith_connector.sync_traces_from_langsmith(
            project_name=sync_request.project_name,
            limit=sync_request.limit,
            team_id=current_user.team_id,
            force_resync=sync_request.force_resync
        )
        
        return SyncResponse(
            status="success",
            project_name=sync_result.project_name,
            total_synced=sync_result.total_synced,
            last_sync=sync_result.last_sync,
            errors=sync_result.errors
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error syncing from LangSmith: {str(e)}"
        )


@router.post("/sync/background")
async def sync_from_langsmith_background(
    sync_request: SyncRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_permission("traces:write"))
):
    """
    Start a background sync operation from LangSmith.
    
    For large sync operations, this endpoint starts the sync in the background
    and returns immediately. Use the status endpoint to monitor progress.
    """
    try:
        # Add sync task to background queue
        background_tasks.add_task(
            _background_sync_task,
            sync_request,
            current_user.team_id
        )
        
        return {
            "status": "started",
            "message": "Background sync started. Check status endpoint for progress.",
            "project_name": sync_request.project_name or langsmith_connector.project
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error starting background sync: {str(e)}"
        )


async def _background_sync_task(sync_request: SyncRequest, team_id: UUID):
    """Background task for LangSmith sync."""
    try:
        await langsmith_connector.sync_traces_from_langsmith(
            project_name=sync_request.project_name,
            limit=sync_request.limit,
            team_id=team_id,
            force_resync=sync_request.force_resync
        )
    except Exception as e:
        # Log error but don't raise since this is a background task
        import logging
        logging.error(f"Background sync failed: {e}")


@router.post("/push-evaluation")
async def push_evaluation_to_langsmith(
    evaluation_request: EvaluationPushRequest,
    current_user: User = Depends(require_permission("evaluations:write"))
):
    """
    Push evaluation results back to LangSmith as feedback.
    
    Sends human evaluation results to LangSmith to close the feedback loop
    and improve model training data.
    """
    try:
        evaluation_data = {
            "score": evaluation_request.score,
            "label": evaluation_request.label,
            "critique": evaluation_request.critique,
            "corrected_output": evaluation_request.corrected_output,
            "evaluator_id": str(current_user.id),
            "evaluator_type": "human",
            "evaluated_at": datetime.utcnow().isoformat()
        }
        
        success = await langsmith_connector.push_evaluation_to_langsmith(
            evaluation_request.trace_id,
            evaluation_data
        )
        
        if success:
            return {
                "status": "success",
                "message": "Evaluation successfully pushed to LangSmith",
                "trace_id": str(evaluation_request.trace_id)
            }
        else:
            return {
                "status": "failed",
                "message": "Failed to push evaluation to LangSmith",
                "trace_id": str(evaluation_request.trace_id)
            }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error pushing evaluation to LangSmith: {str(e)}"
        )


@router.post("/webhook", response_model=WebhookResponse)
async def handle_langsmith_webhook(
    request: Request,
    x_langsmith_signature: Optional[str] = Header(None)
):
    """
    Handle incoming LangSmith webhooks.
    
    Processes real-time events from LangSmith including:
    - run.created: New LLM runs
    - run.updated: Updated runs
    - feedback.created: New feedback/evaluations
    """
    try:
        # Get request payload
        payload = await request.json()
        
        # Process webhook
        result = await langsmith_connector.handle_webhook(
            payload,
            x_langsmith_signature or ""
        )
        
        return WebhookResponse(
            status=result.get("status", "unknown"),
            details=result
        )
    
    except HTTPException:
        # Re-raise HTTP exceptions (like auth failures)
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing LangSmith webhook: {str(e)}"
        )


@router.get("/sync-stats", response_model=Dict[str, Any])
async def get_sync_statistics(
    current_user: User = Depends(get_current_user)
):
    """
    Get detailed synchronization statistics.
    
    Returns comprehensive stats about LangSmith sync activity including
    total synced traces, recent activity, and sync history.
    """
    try:
        stats = await langsmith_connector.get_sync_stats()
        return stats
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting sync statistics: {str(e)}"
        )


@router.post("/test-connection")
async def test_langsmith_connection(
    current_user: User = Depends(require_permission("admin"))
):
    """
    Test LangSmith connection and configuration.
    
    Admin-only endpoint to verify LangSmith API connectivity and settings.
    """
    try:
        if not langsmith_connector.client:
            return {
                "status": "error",
                "message": "LangSmith client not configured. Check API key.",
                "connected": False
            }
        
        # Test connection
        connected = langsmith_connector._test_connection()
        
        if connected:
            status = await langsmith_connector.get_connection_status()
            return {
                "status": "success",
                "message": "LangSmith connection successful",
                "connected": True,
                "details": status
            }
        else:
            return {
                "status": "error",
                "message": "LangSmith connection failed",
                "connected": False
            }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error testing LangSmith connection: {str(e)}"
        )


@router.delete("/sync-cache")
async def clear_sync_cache(
    current_user: User = Depends(require_permission("admin"))
):
    """
    Clear LangSmith sync cache.
    
    Admin-only endpoint to reset sync timestamps and force a full re-sync
    on the next sync operation.
    """
    try:
        from utils.cache_service import cache_service
        
        # Clear all LangSmith sync cache keys
        cache_keys = [
            f"langsmith_last_sync:{langsmith_connector.project}",
            "langsmith_projects",
            "langsmith_status"
        ]
        
        for key in cache_keys:
            await cache_service.delete(key)
        
        return {
            "status": "success",
            "message": "LangSmith sync cache cleared",
            "cleared_keys": cache_keys
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error clearing sync cache: {str(e)}"
        ) 