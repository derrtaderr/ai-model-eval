"""
Enhanced LangSmith connector service for production-grade LLM evaluation platform.

This service provides comprehensive LangSmith integration including:
- Bidirectional sync (pull traces from LangSmith, push evaluations back)
- Real-time webhook handling
- Advanced error handling and retry logic
- Project management
- Enhanced metadata mapping
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from uuid import UUID, uuid4
import hashlib
import hmac
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func, or_
from sqlalchemy.orm import selectinload
from decouple import config
import httpx
from fastapi import HTTPException

from langsmith import Client
from langsmith.schemas import Run, Dataset, Example

from database.models import Trace, Evaluation, TraceTag, User, Team
from database.connection import AsyncSessionLocal
from services.trace_logger import trace_logger
from utils.cache_service import cache_service


logger = logging.getLogger(__name__)


@dataclass
class LangSmithSync:
    """Configuration for LangSmith sync operations."""
    project_name: str
    last_sync: Optional[datetime] = None
    total_synced: int = 0
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class LangSmithConnector:
    """Enhanced LangSmith connector with production features."""
    
    def __init__(self):
        self.api_key = config("LANGCHAIN_API_KEY", default=None)
        self.project = config("LANGCHAIN_PROJECT", default="llm-eval-platform")
        self.webhook_secret = config("LANGSMITH_WEBHOOK_SECRET", default=None)
        self.base_url = config("LANGSMITH_BASE_URL", default="https://api.smith.langchain.com")
        
        # Initialize client
        if self.api_key:
            self.client = Client(api_key=self.api_key)
            self._test_connection()
        else:
            self.client = None
            logger.warning("LangSmith API key not configured. Integration disabled.")
    
    def _test_connection(self) -> bool:
        """Test LangSmith connection."""
        try:
            # Simple test call
            list(self.client.list_datasets(limit=1))
            logger.info("LangSmith connection successful")
            return True
        except Exception as e:
            logger.error(f"LangSmith connection failed: {e}")
            return False
    
    async def get_connection_status(self) -> Dict[str, Any]:
        """Get current connection status and configuration."""
        if not self.client:
            return {
                "connected": False,
                "error": "No API key configured",
                "project": None,
                "last_sync": None
            }
        
        try:
            # Test connection with a simple call
            projects = list(self.client.list_datasets(limit=1))
            
            # Get last sync info from cache
            last_sync_key = f"langsmith_last_sync:{self.project}"
            last_sync = await cache_service.get(last_sync_key)
            
            return {
                "connected": True,
                "project": self.project,
                "last_sync": last_sync,
                "base_url": self.base_url,
                "webhook_configured": bool(self.webhook_secret)
            }
        except Exception as e:
            return {
                "connected": False,
                "error": str(e),
                "project": self.project,
                "last_sync": None
            }
    
    async def sync_traces_from_langsmith(
        self, 
        project_name: Optional[str] = None,
        limit: int = 100,
        team_id: Optional[UUID] = None,
        force_resync: bool = False
    ) -> LangSmithSync:
        """
        Enhanced sync from LangSmith with better error handling and progress tracking.
        
        Args:
            project_name: LangSmith project to sync from
            limit: Maximum number of runs to sync
            team_id: Team to associate synced traces with
            force_resync: Whether to re-sync existing traces
            
        Returns:
            LangSmithSync object with sync results
        """
        if not self.client:
            raise ValueError("LangSmith client not configured")
        
        project_name = project_name or self.project
        sync_result = LangSmithSync(project_name=project_name)
        
        try:
            # Get last sync time to avoid duplicates
            last_sync_key = f"langsmith_last_sync:{project_name}"
            last_sync = None
            
            if not force_resync:
                last_sync_str = await cache_service.get(last_sync_key)
                if last_sync_str:
                    last_sync = datetime.fromisoformat(last_sync_str)
                    sync_result.last_sync = last_sync
            
            # Build filter criteria
            filter_criteria = {"project_name": project_name}
            if last_sync and not force_resync:
                filter_criteria["start_time"] = last_sync
            
            logger.info(f"Starting LangSmith sync for project: {project_name}")
            
            # Get runs from LangSmith
            runs = list(self.client.list_runs(
                project_name=project_name,
                limit=limit,
                start_time=last_sync if not force_resync else None
            ))
            
            logger.info(f"Retrieved {len(runs)} runs from LangSmith")
            
            async with AsyncSessionLocal() as session:
                synced_count = 0
                
                for run in runs:
                    try:
                        # Check if trace already exists
                        if not force_resync:
                            existing_query = select(Trace).where(
                                Trace.langsmith_run_id == str(run.id)
                            )
                            existing_result = await session.execute(existing_query)
                            if existing_result.scalar_one_or_none():
                                continue
                        
                        # Convert LangSmith run to our trace format
                        trace_data = await self._convert_langsmith_run_to_trace(
                            run, team_id, session
                        )
                        
                        if trace_data:
                            # Create or update trace
                            if force_resync:
                                # Update existing trace
                                existing_query = select(Trace).where(
                                    Trace.langsmith_run_id == str(run.id)
                                )
                                existing_result = await session.execute(existing_query)
                                existing_trace = existing_result.scalar_one_or_none()
                                
                                if existing_trace:
                                    for key, value in trace_data.items():
                                        if key != 'id':
                                            setattr(existing_trace, key, value)
                                else:
                                    trace = Trace(**trace_data)
                                    session.add(trace)
                            else:
                                trace = Trace(**trace_data)
                                session.add(trace)
                            
                            synced_count += 1
                        
                    except Exception as e:
                        error_msg = f"Error processing run {run.id}: {str(e)}"
                        logger.error(error_msg)
                        sync_result.errors.append(error_msg)
                        continue
                
                await session.commit()
                sync_result.total_synced = synced_count
                
                # Update last sync time
                now = datetime.utcnow()
                await cache_service.set(
                    last_sync_key, 
                    now.isoformat(), 
                    ttl=86400 * 7  # 7 days
                )
                sync_result.last_sync = now
                
                logger.info(f"LangSmith sync completed: {synced_count} traces synced")
        
        except Exception as e:
            error_msg = f"LangSmith sync failed: {str(e)}"
            logger.error(error_msg)
            sync_result.errors.append(error_msg)
            raise
        
        return sync_result
    
    async def _convert_langsmith_run_to_trace(
        self, 
        run: Run, 
        team_id: Optional[UUID],
        session: AsyncSession
    ) -> Optional[Dict[str, Any]]:
        """Convert a LangSmith run to our trace format."""
        try:
            inputs = run.inputs or {}
            outputs = run.outputs or {}
            
            # Extract user input
            user_input = ""
            if "messages" in inputs:
                # Handle chat format
                messages = inputs["messages"]
                if isinstance(messages, list) and messages:
                    user_messages = [m for m in messages if m.get("role") == "user"]
                    if user_messages:
                        user_input = user_messages[-1].get("content", "")
            elif "input" in inputs:
                user_input = str(inputs["input"])
            elif "question" in inputs:
                user_input = str(inputs["question"])
            
            # Extract model output
            model_output = ""
            if "output" in outputs:
                output = outputs["output"]
                if isinstance(output, dict):
                    model_output = output.get("content", str(output))
                else:
                    model_output = str(output)
            elif "answer" in outputs:
                model_output = str(outputs["answer"])
            
            # Extract system prompt
            system_prompt = None
            if "messages" in inputs:
                messages = inputs["messages"]
                if isinstance(messages, list):
                    system_messages = [m for m in messages if m.get("role") == "system"]
                    if system_messages:
                        system_prompt = system_messages[0].get("content")
            elif "system_prompt" in inputs:
                system_prompt = str(inputs["system_prompt"])
            
            # Extract model name
            model_name = "unknown"
            if run.extra:
                model_name = (
                    run.extra.get("model_name") or 
                    run.extra.get("model") or 
                    run.extra.get("llm", {}).get("model_name") or
                    "unknown"
                )
            
            # Calculate latency
            latency_ms = None
            if run.end_time and run.start_time:
                latency_ms = int((run.end_time - run.start_time).total_seconds() * 1000)
            elif run.total_time:
                latency_ms = int(run.total_time * 1000)
            
            # Extract token usage
            token_count = None
            if run.extra and "token_usage" in run.extra:
                token_usage = run.extra["token_usage"]
                token_count = {
                    "input": token_usage.get("prompt_tokens", 0),
                    "output": token_usage.get("completion_tokens", 0),
                    "total": token_usage.get("total_tokens", 0)
                }
            
            # Enhanced metadata
            metadata = {
                "langsmith": {
                    "run_id": str(run.id),
                    "run_type": run.run_type,
                    "parent_run_id": str(run.parent_run_id) if run.parent_run_id else None,
                    "session_id": run.session_id,
                    "project_name": getattr(run, 'project_name', None),
                    "tags": run.tags or [],
                    "extra": run.extra or {},
                    "feedback": getattr(run, 'feedback_stats', {}),
                    "error": run.error if hasattr(run, 'error') else None
                },
                "sync_info": {
                    "synced_at": datetime.utcnow().isoformat(),
                    "sync_version": "v2.0"
                }
            }
            
            # Extract tools/functions used
            if run.run_type == "tool" or (run.extra and "tools" in run.extra):
                tools_used = []
                if run.extra and "tools" in run.extra:
                    tools_used = run.extra["tools"]
                elif run.name:
                    tools_used = [run.name]
                
                if tools_used:
                    metadata["tools_used"] = tools_used
            
            # Determine status
            status = "completed"
            if run.error:
                status = "error"
                model_output = f"ERROR: {run.error}"
            elif hasattr(run, 'status'):
                status = run.status
            
            return {
                "user_input": user_input,
                "model_output": model_output,
                "model_name": model_name,
                "system_prompt": system_prompt,
                "session_id": run.session_id,
                "metadata": metadata,
                "langsmith_run_id": str(run.id),
                "latency_ms": latency_ms,
                "token_count": token_count,
                "cost_usd": run.total_cost,
                "timestamp": run.start_time or datetime.utcnow(),
                "status": status,
                "team_id": team_id
            }
        
        except Exception as e:
            logger.error(f"Error converting LangSmith run {run.id}: {e}")
            return None
    
    async def push_evaluation_to_langsmith(
        self, 
        trace_id: UUID, 
        evaluation: Dict[str, Any]
    ) -> bool:
        """
        Push evaluation results back to LangSmith as feedback.
        
        Args:
            trace_id: Local trace ID
            evaluation: Evaluation data to push
            
        Returns:
            True if successful, False otherwise
        """
        if not self.client:
            logger.warning("LangSmith client not configured")
            return False
        
        try:
            async with AsyncSessionLocal() as session:
                # Get the trace with LangSmith run ID
                query = select(Trace).where(Trace.id == trace_id)
                result = await session.execute(query)
                trace = result.scalar_one_or_none()
                
                if not trace or not trace.langsmith_run_id:
                    logger.warning(f"Trace {trace_id} not found or missing LangSmith run ID")
                    return False
                
                # Prepare feedback data
                feedback_data = {
                    "key": "human_evaluation",
                    "score": evaluation.get("score"),
                    "value": evaluation.get("label"),
                    "comment": evaluation.get("critique"),
                    "correction": evaluation.get("corrected_output"),
                    "source_info": {
                        "source": "llm-eval-platform",
                        "evaluator_id": evaluation.get("evaluator_id"),
                        "evaluation_type": evaluation.get("evaluator_type", "human"),
                        "timestamp": evaluation.get("evaluated_at", datetime.utcnow().isoformat())
                    }
                }
                
                # Create feedback in LangSmith
                self.client.create_feedback(
                    run_id=trace.langsmith_run_id,
                    key=feedback_data["key"],
                    score=feedback_data["score"],
                    value=feedback_data["value"],
                    comment=feedback_data["comment"],
                    correction=feedback_data.get("correction"),
                    source_info=feedback_data["source_info"]
                )
                
                logger.info(f"Successfully pushed evaluation for trace {trace_id} to LangSmith")
                return True
        
        except Exception as e:
            logger.error(f"Error pushing evaluation to LangSmith: {e}")
            return False
    
    async def handle_webhook(self, payload: Dict[str, Any], signature: str) -> Dict[str, Any]:
        """
        Handle incoming LangSmith webhook.
        
        Args:
            payload: Webhook payload
            signature: Webhook signature for verification
            
        Returns:
            Processing result
        """
        # Verify webhook signature if secret is configured
        if self.webhook_secret and not self._verify_webhook_signature(payload, signature):
            raise HTTPException(status_code=401, detail="Invalid webhook signature")
        
        event_type = payload.get("event_type")
        data = payload.get("data", {})
        
        logger.info(f"Processing LangSmith webhook: {event_type}")
        
        try:
            if event_type == "run.created":
                return await self._handle_run_created(data)
            elif event_type == "run.updated":
                return await self._handle_run_updated(data)
            elif event_type == "feedback.created":
                return await self._handle_feedback_created(data)
            else:
                logger.warning(f"Unknown webhook event type: {event_type}")
                return {"status": "ignored", "reason": f"Unknown event type: {event_type}"}
        
        except Exception as e:
            logger.error(f"Error processing webhook: {e}")
            return {"status": "error", "error": str(e)}
    
    def _verify_webhook_signature(self, payload: Dict[str, Any], signature: str) -> bool:
        """Verify webhook signature."""
        if not self.webhook_secret:
            return True
        
        try:
            payload_bytes = json.dumps(payload, sort_keys=True).encode()
            expected_signature = hmac.new(
                self.webhook_secret.encode(),
                payload_bytes,
                hashlib.sha256
            ).hexdigest()
            
            return hmac.compare_digest(f"sha256={expected_signature}", signature)
        except Exception as e:
            logger.error(f"Error verifying webhook signature: {e}")
            return False
    
    async def _handle_run_created(self, run_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle run.created webhook event."""
        try:
            # Convert webhook data to Run object (simplified)
            run_id = run_data.get("id")
            
            if not run_id:
                return {"status": "ignored", "reason": "No run ID in webhook data"}
            
            # Fetch full run details from LangSmith
            run = self.client.read_run(run_id)
            
            # Convert and store the trace
            async with AsyncSessionLocal() as session:
                trace_data = await self._convert_langsmith_run_to_trace(run, None, session)
                
                if trace_data:
                    trace = Trace(**trace_data)
                    session.add(trace)
                    await session.commit()
                    
                    return {
                        "status": "success", 
                        "trace_id": str(trace.id),
                        "run_id": run_id
                    }
                else:
                    return {"status": "ignored", "reason": "Could not convert run to trace"}
        
        except Exception as e:
            logger.error(f"Error handling run.created webhook: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _handle_run_updated(self, run_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle run.updated webhook event."""
        try:
            run_id = run_data.get("id")
            
            if not run_id:
                return {"status": "ignored", "reason": "No run ID in webhook data"}
            
            async with AsyncSessionLocal() as session:
                # Find existing trace
                query = select(Trace).where(Trace.langsmith_run_id == str(run_id))
                result = await session.execute(query)
                existing_trace = result.scalar_one_or_none()
                
                if not existing_trace:
                    # Treat as new run
                    return await self._handle_run_created(run_data)
                
                # Fetch updated run details
                run = self.client.read_run(run_id)
                trace_data = await self._convert_langsmith_run_to_trace(run, existing_trace.team_id, session)
                
                if trace_data:
                    # Update existing trace
                    for key, value in trace_data.items():
                        if key != 'id':
                            setattr(existing_trace, key, value)
                    
                    await session.commit()
                    
                    return {
                        "status": "updated", 
                        "trace_id": str(existing_trace.id),
                        "run_id": run_id
                    }
                else:
                    return {"status": "ignored", "reason": "Could not convert updated run to trace"}
        
        except Exception as e:
            logger.error(f"Error handling run.updated webhook: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _handle_feedback_created(self, feedback_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle feedback.created webhook event."""
        try:
            run_id = feedback_data.get("run_id")
            
            if not run_id:
                return {"status": "ignored", "reason": "No run ID in feedback data"}
            
            async with AsyncSessionLocal() as session:
                # Find corresponding trace
                query = select(Trace).where(Trace.langsmith_run_id == str(run_id))
                result = await session.execute(query)
                trace = result.scalar_one_or_none()
                
                if not trace:
                    return {"status": "ignored", "reason": "Trace not found for run ID"}
                
                # Create evaluation record from feedback
                evaluation = Evaluation(
                    trace_id=trace.id,
                    evaluator_type="langsmith_feedback",
                    score=feedback_data.get("score"),
                    label=feedback_data.get("value"),
                    critique=feedback_data.get("comment"),
                    metadata={
                        "langsmith_feedback": feedback_data,
                        "source": "langsmith_webhook"
                    },
                    evaluated_at=datetime.utcnow()
                )
                
                session.add(evaluation)
                await session.commit()
                
                return {
                    "status": "success",
                    "evaluation_id": str(evaluation.id),
                    "trace_id": str(trace.id)
                }
        
        except Exception as e:
            logger.error(f"Error handling feedback.created webhook: {e}")
            return {"status": "error", "error": str(e)}
    
    async def get_langsmith_projects(self) -> List[Dict[str, Any]]:
        """Get list of available LangSmith projects."""
        if not self.client:
            return []
        
        try:
            projects = []
            datasets = list(self.client.list_datasets(limit=50))
            
            for dataset in datasets:
                projects.append({
                    "id": str(dataset.id),
                    "name": dataset.name,
                    "description": dataset.description,
                    "created_at": dataset.created_at.isoformat() if dataset.created_at else None,
                    "example_count": getattr(dataset, 'example_count', 0)
                })
            
            return projects
        
        except Exception as e:
            logger.error(f"Error fetching LangSmith projects: {e}")
            return []
    
    async def get_sync_stats(self) -> Dict[str, Any]:
        """Get synchronization statistics."""
        try:
            async with AsyncSessionLocal() as session:
                # Count traces with LangSmith run IDs
                langsmith_traces_query = select(func.count(Trace.id)).where(
                    Trace.langsmith_run_id.isnot(None)
                )
                result = await session.execute(langsmith_traces_query)
                langsmith_traces_count = result.scalar()
                
                # Get last sync time
                last_sync_key = f"langsmith_last_sync:{self.project}"
                last_sync = await cache_service.get(last_sync_key)
                
                # Get recent sync activity (last 24 hours)
                yesterday = datetime.utcnow() - timedelta(days=1)
                recent_traces_query = select(func.count(Trace.id)).where(
                    and_(
                        Trace.langsmith_run_id.isnot(None),
                        Trace.timestamp >= yesterday
                    )
                )
                result = await session.execute(recent_traces_query)
                recent_synced_count = result.scalar()
                
                return {
                    "total_langsmith_traces": langsmith_traces_count,
                    "recent_synced_count": recent_synced_count,
                    "last_sync": last_sync,
                    "project": self.project,
                    "connected": bool(self.client)
                }
        
        except Exception as e:
            logger.error(f"Error getting sync stats: {e}")
            return {
                "total_langsmith_traces": 0,
                "recent_synced_count": 0,
                "last_sync": None,
                "project": self.project,
                "connected": False,
                "error": str(e)
            }


# Global connector instance
langsmith_connector = LangSmithConnector() 