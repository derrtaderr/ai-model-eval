"""
CI/CD Integration API for LLM Evaluation Platform.
Supports GitHub Actions, GitLab CI, and generic webhook integrations.
"""

import hashlib
import hmac
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from uuid import UUID

from fastapi import APIRouter, Request, HTTPException, Depends, BackgroundTasks, Header
from pydantic import BaseModel, Field, ConfigDict
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database.connection import get_db
from database.models import Trace, Evaluation, User
from auth.security import get_current_user_email

logger = logging.getLogger(__name__)
router = APIRouter()

# ============================================================================
# WEBHOOK SCHEMAS
# ============================================================================

class GitHubWebhookPayload(BaseModel):
    """Schema for GitHub webhook payloads."""
    model_config = ConfigDict(protected_namespaces=())
    
    action: str = Field(..., description="GitHub action (opened, synchronize, closed, etc.)")
    pull_request: Optional[Dict[str, Any]] = Field(None, description="PR data if applicable")
    commits: Optional[List[Dict[str, Any]]] = Field(None, description="Commit data if applicable")
    repository: Dict[str, Any] = Field(..., description="Repository information")
    sender: Dict[str, Any] = Field(..., description="User who triggered the event")


class GitLabWebhookPayload(BaseModel):
    """Schema for GitLab webhook payloads."""
    model_config = ConfigDict(protected_namespaces=())
    
    object_kind: str = Field(..., description="GitLab event type (push, merge_request, etc.)")
    event_type: Optional[str] = Field(None, description="Event type for merge requests")
    project: Dict[str, Any] = Field(..., description="Project information")
    commits: Optional[List[Dict[str, Any]]] = Field(None, description="Commit data")
    merge_request: Optional[Dict[str, Any]] = Field(None, description="MR data if applicable")
    user: Dict[str, Any] = Field(..., description="User who triggered the event")


class TriggerRequest(BaseModel):
    """Schema for generic trigger requests."""
    model_config = ConfigDict(protected_namespaces=())
    
    trigger_type: str = Field(..., description="Type of trigger (evaluation, export, test)")
    payload: Dict[str, Any] = Field(..., description="Trigger payload data")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    callback_url: Optional[str] = Field(None, description="Callback URL for results")


class IntegrationResponse(BaseModel):
    """Schema for integration responses."""
    model_config = ConfigDict(protected_namespaces=())
    
    status: str = Field(..., description="Integration status (success, pending, failed)")
    job_id: Optional[str] = Field(None, description="Background job ID if applicable")
    message: str = Field(..., description="Response message")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Response metadata")


class IntegrationStatus(BaseModel):
    """Schema for integration status."""
    model_config = ConfigDict(protected_namespaces=())
    
    github_webhook_active: bool = Field(..., description="GitHub webhook status")
    gitlab_webhook_active: bool = Field(..., description="GitLab webhook status")
    active_jobs: int = Field(..., description="Number of active background jobs")
    last_github_event: Optional[datetime] = Field(None, description="Last GitHub event timestamp")
    last_gitlab_event: Optional[datetime] = Field(None, description="Last GitLab event timestamp")
    health_check_passed: bool = Field(..., description="Overall health status")


# ============================================================================
# WEBHOOK SECURITY
# ============================================================================

def verify_github_signature(payload: bytes, signature: str, secret: str) -> bool:
    """Verify GitHub webhook signature."""
    if not signature or not signature.startswith("sha256="):
        return False
    
    expected_signature = hmac.new(
        secret.encode('utf-8'),
        payload,
        hashlib.sha256
    ).hexdigest()
    
    return hmac.compare_digest(f"sha256={expected_signature}", signature)


def verify_gitlab_token(provided_token: str, expected_token: str) -> bool:
    """Verify GitLab webhook token."""
    return hmac.compare_digest(provided_token, expected_token)


# ============================================================================
# WEBHOOK HANDLERS
# ============================================================================

@router.post("/webhooks/github", response_model=IntegrationResponse)
async def handle_github_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_github_event: str = Header(..., alias="X-GitHub-Event"),
    x_hub_signature_256: Optional[str] = Header(None, alias="X-Hub-Signature-256"),
    db: AsyncSession = Depends(get_db)
):
    """Handle GitHub webhook events."""
    
    try:
        # Get the raw payload
        payload = await request.body()
        
        # Verify signature (in production, you'd get the secret from environment)
        github_secret = "your-github-webhook-secret"  # TODO: Move to environment
        if x_hub_signature_256 and not verify_github_signature(payload, x_hub_signature_256, github_secret):
            raise HTTPException(status_code=401, detail="Invalid signature")
        
        # Parse the payload
        try:
            payload_data = json.loads(payload)
            webhook_payload = GitHubWebhookPayload(**payload_data)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid payload: {str(e)}")
        
        # Handle different event types
        job_id = None
        message = f"Processed GitHub {x_github_event} event"
        
        if x_github_event == "pull_request":
            job_id = await handle_pull_request_event(webhook_payload, background_tasks, db)
            message = f"Triggered evaluation for PR #{webhook_payload.pull_request.get('number', 'unknown')}"
            
        elif x_github_event == "push":
            job_id = await handle_push_event(webhook_payload, background_tasks, db)
            message = f"Triggered evaluation for push to {webhook_payload.repository['name']}"
            
        elif x_github_event == "workflow_run":
            job_id = await handle_workflow_event(webhook_payload, background_tasks, db)
            message = "Processed workflow completion event"
        
        return IntegrationResponse(
            status="success",
            job_id=job_id,
            message=message,
            metadata={
                "event_type": x_github_event,
                "repository": webhook_payload.repository.get("full_name"),
                "sender": webhook_payload.sender.get("login")
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"GitHub webhook error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Webhook processing failed: {str(e)}")


@router.post("/webhooks/gitlab", response_model=IntegrationResponse)
async def handle_gitlab_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_gitlab_event: str = Header(..., alias="X-Gitlab-Event"),
    x_gitlab_token: Optional[str] = Header(None, alias="X-Gitlab-Token"),
    db: AsyncSession = Depends(get_db)
):
    """Handle GitLab webhook events."""
    
    try:
        # Get the raw payload
        payload = await request.body()
        
        # Verify token (in production, you'd get the token from environment)
        gitlab_token = "your-gitlab-webhook-token"  # TODO: Move to environment
        if x_gitlab_token and not verify_gitlab_token(x_gitlab_token, gitlab_token):
            raise HTTPException(status_code=401, detail="Invalid token")
        
        # Parse the payload
        try:
            payload_data = json.loads(payload)
            webhook_payload = GitLabWebhookPayload(**payload_data)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid payload: {str(e)}")
        
        # Handle different event types
        job_id = None
        message = f"Processed GitLab {x_gitlab_event} event"
        
        if webhook_payload.object_kind == "merge_request":
            job_id = await handle_merge_request_event(webhook_payload, background_tasks, db)
            message = f"Triggered evaluation for MR !{webhook_payload.merge_request.get('iid', 'unknown')}"
            
        elif webhook_payload.object_kind == "push":
            job_id = await handle_gitlab_push_event(webhook_payload, background_tasks, db)
            message = f"Triggered evaluation for push to {webhook_payload.project['name']}"
            
        elif webhook_payload.object_kind == "pipeline":
            job_id = await handle_pipeline_event(webhook_payload, background_tasks, db)
            message = "Processed pipeline event"
        
        return IntegrationResponse(
            status="success",
            job_id=job_id,
            message=message,
            metadata={
                "event_type": webhook_payload.object_kind,
                "project": webhook_payload.project.get("path_with_namespace"),
                "user": webhook_payload.user.get("username")
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"GitLab webhook error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Webhook processing failed: {str(e)}")


@router.post("/trigger", response_model=IntegrationResponse)
async def generic_trigger(
    trigger_request: TriggerRequest,
    background_tasks: BackgroundTasks,
    current_user: str = Depends(get_current_user_email),
    db: AsyncSession = Depends(get_db)
):
    """Generic trigger endpoint for external integrations."""
    
    try:
        job_id = None
        message = f"Processed {trigger_request.trigger_type} trigger"
        
        if trigger_request.trigger_type == "evaluation":
            job_id = await trigger_evaluation(trigger_request, background_tasks, db, current_user)
            message = "Triggered batch evaluation"
            
        elif trigger_request.trigger_type == "export":
            job_id = await trigger_export(trigger_request, background_tasks, db, current_user)
            message = "Triggered data export"
            
        elif trigger_request.trigger_type == "test":
            job_id = await trigger_test_suite(trigger_request, background_tasks, db, current_user)
            message = "Triggered test suite execution"
            
        else:
            raise HTTPException(status_code=400, detail=f"Unknown trigger type: {trigger_request.trigger_type}")
        
        return IntegrationResponse(
            status="success" if job_id else "pending",
            job_id=job_id,
            message=message,
            metadata=trigger_request.metadata or {}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Generic trigger error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Trigger processing failed: {str(e)}")


@router.get("/status", response_model=IntegrationStatus)
async def get_integration_status(
    current_user: str = Depends(get_current_user_email)
):
    """Get integration system status."""
    
    try:
        # Check integration health
        # In a real implementation, you'd check actual webhook endpoints, 
        # background job queues, etc.
        
        return IntegrationStatus(
            github_webhook_active=True,
            gitlab_webhook_active=True,
            active_jobs=0,  # TODO: Get from actual job queue
            last_github_event=None,  # TODO: Get from database
            last_gitlab_event=None,  # TODO: Get from database
            health_check_passed=True
        )
        
    except Exception as e:
        logger.error(f"Integration status error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")


# ============================================================================
# EVENT HANDLERS
# ============================================================================

async def handle_pull_request_event(
    payload: GitHubWebhookPayload, 
    background_tasks: BackgroundTasks, 
    db: AsyncSession
) -> Optional[str]:
    """Handle GitHub pull request events."""
    
    if payload.action in ["opened", "synchronize", "reopened"]:
        # Trigger evaluation for the PR
        job_id = f"github-pr-{payload.pull_request['id']}-{datetime.utcnow().timestamp()}"
        
        # Add background task for evaluation
        background_tasks.add_task(
            run_pr_evaluation,
            job_id,
            payload.pull_request,
            payload.repository
        )
        
        return job_id
    
    return None


async def handle_push_event(
    payload: GitHubWebhookPayload, 
    background_tasks: BackgroundTasks, 
    db: AsyncSession
) -> Optional[str]:
    """Handle GitHub push events."""
    
    # Only process pushes to main/master branches
    ref = payload.commits[0].get('ref', '') if payload.commits else ''
    if 'main' in ref or 'master' in ref:
        job_id = f"github-push-{payload.repository['id']}-{datetime.utcnow().timestamp()}"
        
        background_tasks.add_task(
            run_push_evaluation,
            job_id,
            payload.commits,
            payload.repository
        )
        
        return job_id
    
    return None


async def handle_workflow_event(
    payload: GitHubWebhookPayload, 
    background_tasks: BackgroundTasks, 
    db: AsyncSession
) -> Optional[str]:
    """Handle GitHub workflow events."""
    
    # Process workflow completion events
    job_id = f"github-workflow-{datetime.utcnow().timestamp()}"
    
    background_tasks.add_task(
        process_workflow_results,
        job_id,
        payload
    )
    
    return job_id


async def handle_merge_request_event(
    payload: GitLabWebhookPayload, 
    background_tasks: BackgroundTasks, 
    db: AsyncSession
) -> Optional[str]:
    """Handle GitLab merge request events."""
    
    if payload.event_type in ["open", "update", "reopen"]:
        job_id = f"gitlab-mr-{payload.merge_request['id']}-{datetime.utcnow().timestamp()}"
        
        background_tasks.add_task(
            run_mr_evaluation,
            job_id,
            payload.merge_request,
            payload.project
        )
        
        return job_id
    
    return None


async def handle_gitlab_push_event(
    payload: GitLabWebhookPayload, 
    background_tasks: BackgroundTasks, 
    db: AsyncSession
) -> Optional[str]:
    """Handle GitLab push events."""
    
    # Process pushes to default branch
    if payload.project.get('default_branch') in str(payload.commits[0].get('ref', '')) if payload.commits else False:
        job_id = f"gitlab-push-{payload.project['id']}-{datetime.utcnow().timestamp()}"
        
        background_tasks.add_task(
            run_gitlab_push_evaluation,
            job_id,
            payload.commits,
            payload.project
        )
        
        return job_id
    
    return None


async def handle_pipeline_event(
    payload: GitLabWebhookPayload, 
    background_tasks: BackgroundTasks, 
    db: AsyncSession
) -> Optional[str]:
    """Handle GitLab pipeline events."""
    
    job_id = f"gitlab-pipeline-{datetime.utcnow().timestamp()}"
    
    background_tasks.add_task(
        process_pipeline_results,
        job_id,
        payload
    )
    
    return job_id


# ============================================================================
# TRIGGER HANDLERS
# ============================================================================

async def trigger_evaluation(
    request: TriggerRequest, 
    background_tasks: BackgroundTasks, 
    db: AsyncSession,
    user_email: str
) -> Optional[str]:
    """Trigger batch evaluation from external request."""
    
    job_id = f"trigger-eval-{datetime.utcnow().timestamp()}"
    
    background_tasks.add_task(
        run_triggered_evaluation,
        job_id,
        request.payload,
        user_email
    )
    
    return job_id


async def trigger_export(
    request: TriggerRequest, 
    background_tasks: BackgroundTasks, 
    db: AsyncSession,
    user_email: str
) -> Optional[str]:
    """Trigger data export from external request."""
    
    job_id = f"trigger-export-{datetime.utcnow().timestamp()}"
    
    background_tasks.add_task(
        run_triggered_export,
        job_id,
        request.payload,
        user_email
    )
    
    return job_id


async def trigger_test_suite(
    request: TriggerRequest, 
    background_tasks: BackgroundTasks, 
    db: AsyncSession,
    user_email: str
) -> Optional[str]:
    """Trigger test suite execution from external request."""
    
    job_id = f"trigger-test-{datetime.utcnow().timestamp()}"
    
    background_tasks.add_task(
        run_triggered_tests,
        job_id,
        request.payload,
        user_email
    )
    
    return job_id


# ============================================================================
# BACKGROUND TASKS
# ============================================================================

async def run_pr_evaluation(job_id: str, pr_data: Dict[str, Any], repo_data: Dict[str, Any]):
    """Background task to run PR evaluation."""
    logger.info(f"Starting PR evaluation job {job_id} for PR #{pr_data.get('number')}")
    
    # In a real implementation, this would:
    # 1. Extract code changes from the PR
    # 2. Run model evaluations on affected components
    # 3. Compare against baseline performance
    # 4. Post results back to GitHub as a comment
    
    # Placeholder implementation
    await asyncio.sleep(2)  # Simulate processing
    logger.info(f"Completed PR evaluation job {job_id}")


async def run_push_evaluation(job_id: str, commits: List[Dict[str, Any]], repo_data: Dict[str, Any]):
    """Background task to run push evaluation."""
    logger.info(f"Starting push evaluation job {job_id} for {len(commits)} commits")
    
    # Implementation would analyze commits and trigger appropriate evaluations
    await asyncio.sleep(1)
    logger.info(f"Completed push evaluation job {job_id}")


async def process_workflow_results(job_id: str, workflow_data: Dict[str, Any]):
    """Background task to process workflow results."""
    logger.info(f"Processing workflow results job {job_id}")
    
    # Implementation would extract results from workflow and update database
    await asyncio.sleep(1)
    logger.info(f"Completed workflow processing job {job_id}")


async def run_mr_evaluation(job_id: str, mr_data: Dict[str, Any], project_data: Dict[str, Any]):
    """Background task to run merge request evaluation."""
    logger.info(f"Starting MR evaluation job {job_id} for MR !{mr_data.get('iid')}")
    
    # Similar to PR evaluation but for GitLab
    await asyncio.sleep(2)
    logger.info(f"Completed MR evaluation job {job_id}")


async def run_gitlab_push_evaluation(job_id: str, commits: List[Dict[str, Any]], project_data: Dict[str, Any]):
    """Background task to run GitLab push evaluation."""
    logger.info(f"Starting GitLab push evaluation job {job_id}")
    
    await asyncio.sleep(1)
    logger.info(f"Completed GitLab push evaluation job {job_id}")


async def process_pipeline_results(job_id: str, pipeline_data: Dict[str, Any]):
    """Background task to process pipeline results."""
    logger.info(f"Processing pipeline results job {job_id}")
    
    await asyncio.sleep(1)
    logger.info(f"Completed pipeline processing job {job_id}")


async def run_triggered_evaluation(job_id: str, payload: Dict[str, Any], user_email: str):
    """Background task for triggered evaluation."""
    logger.info(f"Running triggered evaluation job {job_id} for user {user_email}")
    
    # Implementation would use the batch evaluation system
    await asyncio.sleep(3)
    logger.info(f"Completed triggered evaluation job {job_id}")


async def run_triggered_export(job_id: str, payload: Dict[str, Any], user_email: str):
    """Background task for triggered export."""
    logger.info(f"Running triggered export job {job_id} for user {user_email}")
    
    # Implementation would use the export system we built in subtask 9.1
    await asyncio.sleep(2)
    logger.info(f"Completed triggered export job {job_id}")


async def run_triggered_tests(job_id: str, payload: Dict[str, Any], user_email: str):
    """Background task for triggered test suite."""
    logger.info(f"Running triggered tests job {job_id} for user {user_email}")
    
    # Implementation would run test suites
    await asyncio.sleep(2)
    logger.info(f"Completed triggered tests job {job_id}")


# Add missing import for asyncio 