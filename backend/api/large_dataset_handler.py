"""
Large Dataset Handling and Performance Optimization Module.
Provides streaming exports, background job processing, and efficient pagination
for handling datasets with millions of records.
"""

import asyncio
import gzip
import json
import logging
import time
from datetime import datetime, timedelta
from io import BytesIO, StringIO
from typing import AsyncGenerator, Dict, Any, List, Optional, Union, Callable
from uuid import uuid4
import csv

from fastapi import APIRouter, Depends, HTTPException, Response, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, ConfigDict
from sqlalchemy import select, func, text, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from database.connection import get_db
from database.models import Trace, Evaluation, User
from auth.security import get_current_user_email

logger = logging.getLogger(__name__)
router = APIRouter()

# ============================================================================
# MODELS AND SCHEMAS
# ============================================================================

class StreamingExportRequest(BaseModel):
    """Request schema for streaming exports."""
    model_config = ConfigDict(protected_namespaces=())
    
    format: str = Field("json", description="Export format (json, csv, jsonl)")
    filters: Optional[Dict[str, Any]] = Field(None, description="Filter criteria")
    fields: Optional[List[str]] = Field(None, description="Specific fields to export")
    compress: bool = Field(False, description="Whether to gzip compress the output")
    chunk_size: int = Field(1000, ge=100, le=10000, description="Number of records per chunk")
    include_evaluations: bool = Field(True, description="Include evaluation data")
    date_from: Optional[str] = Field(None, description="Start date filter (ISO format)")
    date_to: Optional[str] = Field(None, description="End date filter (ISO format)")


class BackgroundJobRequest(BaseModel):
    """Request schema for background job creation."""
    model_config = ConfigDict(protected_namespaces=())
    
    job_type: str = Field(..., description="Type of job (export, analysis, cleanup)")
    parameters: Dict[str, Any] = Field(..., description="Job parameters")
    priority: str = Field("normal", description="Job priority (low, normal, high)")
    callback_url: Optional[str] = Field(None, description="Webhook URL for completion notification")
    email_notification: Optional[str] = Field(None, description="Email for completion notification")
    estimated_duration: Optional[int] = Field(None, description="Estimated duration in seconds")


class BackgroundJobStatus(BaseModel):
    """Schema for background job status."""
    model_config = ConfigDict(protected_namespaces=())
    
    job_id: str
    status: str = Field(..., description="Job status (queued, running, completed, failed, cancelled)")
    progress: float = Field(..., ge=0, le=1, description="Job progress (0-1)")
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
    result_url: Optional[str] = None
    estimated_completion: Optional[str] = None
    processed_records: int = 0
    total_records: Optional[int] = None


class PaginationRequest(BaseModel):
    """Advanced pagination request."""
    model_config = ConfigDict(protected_namespaces=())
    
    cursor: Optional[str] = Field(None, description="Cursor for pagination")
    limit: int = Field(100, ge=1, le=10000, description="Number of records per page")
    sort_by: str = Field("timestamp", description="Field to sort by")
    sort_order: str = Field("desc", description="Sort order (asc, desc)")
    filters: Optional[Dict[str, Any]] = Field(None, description="Filter criteria")
    include_total: bool = Field(False, description="Whether to include total count")


class PaginatedResponse(BaseModel):
    """Paginated response with metadata."""
    model_config = ConfigDict(protected_namespaces=())
    
    data: List[Dict[str, Any]]
    pagination: Dict[str, Any]
    metadata: Dict[str, Any]


# ============================================================================
# BACKGROUND JOB MANAGEMENT
# ============================================================================

# In-memory job storage (use Redis/database in production)
BACKGROUND_JOBS = {}
JOB_RESULTS = {}

class BackgroundJobManager:
    """Manages background job execution and status tracking."""
    
    def __init__(self):
        self.jobs = BACKGROUND_JOBS
        self.results = JOB_RESULTS
        self._running_jobs = set()
    
    async def create_job(
        self,
        job_type: str,
        parameters: Dict[str, Any],
        user_email: str,
        priority: str = "normal",
        callback_url: Optional[str] = None,
        email_notification: Optional[str] = None
    ) -> str:
        """Create a new background job."""
        
        job_id = str(uuid4())
        
        job_data = {
            "id": job_id,
            "type": job_type,
            "parameters": parameters,
            "user_email": user_email,
            "priority": priority,
            "status": "queued",
            "progress": 0.0,
            "created_at": datetime.utcnow(),
            "started_at": None,
            "completed_at": None,
            "error_message": None,
            "result_url": None,
            "callback_url": callback_url,
            "email_notification": email_notification,
            "processed_records": 0,
            "total_records": None
        }
        
        self.jobs[job_id] = job_data
        
        # Start job execution in background
        asyncio.create_task(self._execute_job(job_id))
        
        return job_id
    
    async def get_job_status(self, job_id: str) -> Optional[BackgroundJobStatus]:
        """Get job status by ID."""
        
        job_data = self.jobs.get(job_id)
        if not job_data:
            return None
        
        return BackgroundJobStatus(
            job_id=job_id,
            status=job_data["status"],
            progress=job_data["progress"],
            created_at=job_data["created_at"].isoformat(),
            started_at=job_data["started_at"].isoformat() if job_data["started_at"] else None,
            completed_at=job_data["completed_at"].isoformat() if job_data["completed_at"] else None,
            error_message=job_data["error_message"],
            result_url=job_data["result_url"],
            estimated_completion=self._calculate_estimated_completion(job_data),
            processed_records=job_data["processed_records"],
            total_records=job_data["total_records"]
        )
    
    async def cancel_job(self, job_id: str, user_email: str) -> bool:
        """Cancel a running job."""
        
        job_data = self.jobs.get(job_id)
        if not job_data or job_data["user_email"] != user_email:
            return False
        
        if job_data["status"] in ["completed", "failed", "cancelled"]:
            return False
        
        job_data["status"] = "cancelled"
        job_data["completed_at"] = datetime.utcnow()
        
        return True
    
    async def list_user_jobs(
        self, 
        user_email: str, 
        status_filter: Optional[str] = None,
        limit: int = 50
    ) -> List[BackgroundJobStatus]:
        """List jobs for a specific user."""
        
        user_jobs = [
            job for job in self.jobs.values() 
            if job["user_email"] == user_email
        ]
        
        if status_filter:
            user_jobs = [job for job in user_jobs if job["status"] == status_filter]
        
        # Sort by creation time (newest first)
        user_jobs.sort(key=lambda x: x["created_at"], reverse=True)
        
        # Apply limit
        user_jobs = user_jobs[:limit]
        
        # Convert to status objects
        status_list = []
        for job in user_jobs:
            status_list.append(await self.get_job_status(job["id"]))
        
        return [status for status in status_list if status]
    
    def _calculate_estimated_completion(self, job_data: Dict) -> Optional[str]:
        """Calculate estimated completion time based on progress."""
        
        if job_data["status"] != "running" or job_data["progress"] <= 0:
            return None
        
        started_at = job_data["started_at"]
        if not started_at:
            return None
        
        elapsed = (datetime.utcnow() - started_at).total_seconds()
        estimated_total = elapsed / job_data["progress"]
        estimated_completion = started_at + timedelta(seconds=estimated_total)
        
        return estimated_completion.isoformat()
    
    async def _execute_job(self, job_id: str):
        """Execute a background job."""
        
        job_data = self.jobs.get(job_id)
        if not job_data or job_id in self._running_jobs:
            return
        
        self._running_jobs.add(job_id)
        
        try:
            job_data["status"] = "running"
            job_data["started_at"] = datetime.utcnow()
            
            job_type = job_data["type"]
            parameters = job_data["parameters"]
            
            if job_type == "export":
                await self._execute_export_job(job_id, parameters)
            elif job_type == "analysis":
                await self._execute_analysis_job(job_id, parameters)
            elif job_type == "cleanup":
                await self._execute_cleanup_job(job_id, parameters)
            else:
                raise ValueError(f"Unknown job type: {job_type}")
            
            job_data["status"] = "completed"
            job_data["progress"] = 1.0
            
        except Exception as e:
            logger.error(f"Job {job_id} failed: {str(e)}")
            job_data["status"] = "failed"
            job_data["error_message"] = str(e)
        
        finally:
            job_data["completed_at"] = datetime.utcnow()
            self._running_jobs.discard(job_id)
            
            # Send notifications if configured
            await self._send_job_notifications(job_id)
    
    async def _execute_export_job(self, job_id: str, parameters: Dict[str, Any]):
        """Execute an export job."""
        
        job_data = self.jobs[job_id]
        
        # Mock export processing with progress updates
        total_records = parameters.get("total_records", 100000)
        chunk_size = parameters.get("chunk_size", 1000)
        
        job_data["total_records"] = total_records
        
        # Simulate processing chunks
        for i in range(0, total_records, chunk_size):
            if job_data["status"] == "cancelled":
                break
            
            # Simulate processing time
            await asyncio.sleep(0.1)
            
            processed = min(i + chunk_size, total_records)
            job_data["processed_records"] = processed
            job_data["progress"] = processed / total_records
            
            logger.info(f"Export job {job_id}: processed {processed}/{total_records} records")
        
        # Generate result file URL
        result_filename = f"export_{job_id}.{parameters.get('format', 'json')}"
        job_data["result_url"] = f"/api/large-dataset/download/{result_filename}"
    
    async def _execute_analysis_job(self, job_id: str, parameters: Dict[str, Any]):
        """Execute an analysis job."""
        
        job_data = self.jobs[job_id]
        
        # Mock analysis processing
        analysis_steps = parameters.get("steps", ["load_data", "analyze", "generate_report"])
        
        for i, step in enumerate(analysis_steps):
            if job_data["status"] == "cancelled":
                break
            
            logger.info(f"Analysis job {job_id}: executing step {step}")
            
            # Simulate step processing
            await asyncio.sleep(2)
            
            job_data["progress"] = (i + 1) / len(analysis_steps)
        
        # Generate result
        result_filename = f"analysis_{job_id}.json"
        job_data["result_url"] = f"/api/large-dataset/download/{result_filename}"
    
    async def _execute_cleanup_job(self, job_id: str, parameters: Dict[str, Any]):
        """Execute a cleanup job."""
        
        job_data = self.jobs[job_id]
        
        # Mock cleanup processing
        cleanup_tasks = parameters.get("tasks", ["old_logs", "temp_files", "cache"])
        
        for i, task in enumerate(cleanup_tasks):
            if job_data["status"] == "cancelled":
                break
            
            logger.info(f"Cleanup job {job_id}: executing task {task}")
            
            # Simulate cleanup processing
            await asyncio.sleep(1)
            
            job_data["progress"] = (i + 1) / len(cleanup_tasks)
    
    async def _send_job_notifications(self, job_id: str):
        """Send notifications for job completion."""
        
        job_data = self.jobs[job_id]
        
        # Email notification
        if job_data.get("email_notification"):
            logger.info(f"Sending email notification for job {job_id} to {job_data['email_notification']}")
            # Implement email sending logic here
        
        # Webhook notification
        if job_data.get("callback_url"):
            logger.info(f"Sending webhook notification for job {job_id} to {job_data['callback_url']}")
            # Implement webhook sending logic here


# Global job manager instance
job_manager = BackgroundJobManager()


# ============================================================================
# STREAMING EXPORT FUNCTIONALITY
# ============================================================================

class StreamingExporter:
    """Handles streaming exports for large datasets."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def stream_traces(
        self,
        format: str = "json",
        filters: Optional[Dict[str, Any]] = None,
        fields: Optional[List[str]] = None,
        chunk_size: int = 1000,
        include_evaluations: bool = True,
        compress: bool = False
    ) -> AsyncGenerator[bytes, None]:
        """Stream traces in the specified format."""
        
        # Build query
        query = select(Trace)
        
        if include_evaluations:
            query = query.options(selectinload(Trace.evaluations))
        
        # Apply filters
        if filters:
            conditions = []
            
            if filters.get("model_names"):
                conditions.append(Trace.model_name.in_(filters["model_names"]))
            
            if filters.get("date_from"):
                date_from = datetime.fromisoformat(filters["date_from"].replace('Z', '+00:00'))
                conditions.append(Trace.timestamp >= date_from)
            
            if filters.get("date_to"):
                date_to = datetime.fromisoformat(filters["date_to"].replace('Z', '+00:00'))
                conditions.append(Trace.timestamp <= date_to)
            
            if filters.get("session_ids"):
                conditions.append(Trace.session_id.in_(filters["session_ids"]))
            
            if conditions:
                query = query.where(and_(*conditions))
        
        # Order by timestamp for consistent pagination
        query = query.order_by(Trace.timestamp.desc())
        
        # Stream data
        if format.lower() == "json":
            async for chunk in self._stream_json(query, chunk_size, fields, compress):
                yield chunk
        elif format.lower() == "csv":
            async for chunk in self._stream_csv(query, chunk_size, fields, compress):
                yield chunk
        elif format.lower() == "jsonl":
            async for chunk in self._stream_jsonl(query, chunk_size, fields, compress):
                yield chunk
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    async def _stream_json(
        self,
        query,
        chunk_size: int,
        fields: Optional[List[str]],
        compress: bool
    ) -> AsyncGenerator[bytes, None]:
        """Stream data as JSON array."""
        
        # Start JSON array
        start_data = b'{"data": ['
        if compress:
            yield gzip.compress(start_data)
        else:
            yield start_data
        
        first_chunk = True
        offset = 0
        
        while True:
            # Get chunk of data
            chunk_query = query.offset(offset).limit(chunk_size)
            result = await self.db.execute(chunk_query)
            traces = result.scalars().all()
            
            if not traces:
                break
            
            # Convert to JSON
            chunk_data = []
            for trace in traces:
                trace_dict = self._trace_to_dict(trace, fields)
                chunk_data.append(trace_dict)
            
            # Add comma separator (except for first chunk)
            json_chunk = ""
            if not first_chunk:
                json_chunk = ","
            
            json_chunk += json.dumps(chunk_data)[1:-1]  # Remove array brackets
            
            chunk_bytes = json_chunk.encode('utf-8')
            if compress:
                yield gzip.compress(chunk_bytes)
            else:
                yield chunk_bytes
            
            first_chunk = False
            offset += chunk_size
            
            # Small delay to prevent overwhelming the database
            await asyncio.sleep(0.01)
        
        # End JSON array
        end_data = b']}'
        if compress:
            yield gzip.compress(end_data)
        else:
            yield end_data
    
    async def _stream_csv(
        self,
        query,
        chunk_size: int,
        fields: Optional[List[str]],
        compress: bool
    ) -> AsyncGenerator[bytes, None]:
        """Stream data as CSV."""
        
        # Get first chunk to determine headers
        first_chunk_query = query.limit(1)
        result = await self.db.execute(first_chunk_query)
        first_trace = result.scalar_one_or_none()
        
        if not first_trace:
            return
        
        # Determine CSV headers
        sample_dict = self._trace_to_dict(first_trace, fields)
        headers = list(sample_dict.keys())
        
        # Write CSV header
        output = StringIO()
        writer = csv.DictWriter(output, fieldnames=headers)
        writer.writeheader()
        header_data = output.getvalue().encode('utf-8')
        
        if compress:
            yield gzip.compress(header_data)
        else:
            yield header_data
        
        # Stream data chunks
        offset = 0
        
        while True:
            chunk_query = query.offset(offset).limit(chunk_size)
            result = await self.db.execute(chunk_query)
            traces = result.scalars().all()
            
            if not traces:
                break
            
            # Convert chunk to CSV
            output = StringIO()
            writer = csv.DictWriter(output, fieldnames=headers)
            
            for trace in traces:
                trace_dict = self._trace_to_dict(trace, fields)
                writer.writerow(trace_dict)
            
            chunk_data = output.getvalue().encode('utf-8')
            
            if compress:
                yield gzip.compress(chunk_data)
            else:
                yield chunk_data
            
            offset += chunk_size
            await asyncio.sleep(0.01)
    
    async def _stream_jsonl(
        self,
        query,
        chunk_size: int,
        fields: Optional[List[str]],
        compress: bool
    ) -> AsyncGenerator[bytes, None]:
        """Stream data as JSON Lines."""
        
        offset = 0
        
        while True:
            chunk_query = query.offset(offset).limit(chunk_size)
            result = await self.db.execute(chunk_query)
            traces = result.scalars().all()
            
            if not traces:
                break
            
            # Convert chunk to JSONL
            jsonl_lines = []
            for trace in traces:
                trace_dict = self._trace_to_dict(trace, fields)
                jsonl_lines.append(json.dumps(trace_dict))
            
            chunk_data = ('\n'.join(jsonl_lines) + '\n').encode('utf-8')
            
            if compress:
                yield gzip.compress(chunk_data)
            else:
                yield chunk_data
            
            offset += chunk_size
            await asyncio.sleep(0.01)
    
    def _trace_to_dict(self, trace: Trace, fields: Optional[List[str]] = None) -> Dict[str, Any]:
        """Convert trace to dictionary with optional field filtering."""
        
        trace_dict = {
            "id": str(trace.id),
            "timestamp": trace.timestamp.isoformat(),
            "user_input": trace.user_input,
            "model_output": trace.model_output,
            "model_name": trace.model_name,
            "system_prompt": trace.system_prompt,
            "session_id": trace.session_id,
            "status": trace.status,
            "latency_ms": trace.latency_ms,
            "cost_usd": trace.cost_usd,
            "metadata": trace.metadata
        }
        
        # Add evaluation data if available
        if hasattr(trace, 'evaluations') and trace.evaluations:
            eval_data = []
            for evaluation in trace.evaluations:
                eval_dict = {
                    "id": str(evaluation.id),
                    "evaluator_type": evaluation.evaluator_type,
                    "score": evaluation.score,
                    "label": evaluation.label,
                    "critique": evaluation.critique,
                    "metadata": evaluation.metadata
                }
                eval_data.append(eval_dict)
            trace_dict["evaluations"] = eval_data
        
        # Filter fields if specified
        if fields:
            trace_dict = {k: v for k, v in trace_dict.items() if k in fields}
        
        return trace_dict


# ============================================================================
# ADVANCED PAGINATION
# ============================================================================

class AdvancedPaginator:
    """Provides cursor-based pagination for large datasets."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def paginate_traces(
        self,
        cursor: Optional[str] = None,
        limit: int = 100,
        sort_by: str = "timestamp",
        sort_order: str = "desc",
        filters: Optional[Dict[str, Any]] = None,
        include_total: bool = False
    ) -> PaginatedResponse:
        """Paginate traces with cursor-based pagination."""
        
        # Build base query
        query = select(Trace).options(selectinload(Trace.evaluations))
        
        # Apply filters
        if filters:
            conditions = self._build_filter_conditions(filters)
            if conditions:
                query = query.where(and_(*conditions))
        
        # Apply sorting
        sort_column = getattr(Trace, sort_by, Trace.timestamp)
        if sort_order.lower() == "desc":
            query = query.order_by(sort_column.desc())
        else:
            query = query.order_by(sort_column.asc())
        
        # Apply cursor pagination
        if cursor:
            cursor_value = self._decode_cursor(cursor)
            if sort_order.lower() == "desc":
                query = query.where(sort_column < cursor_value)
            else:
                query = query.where(sort_column > cursor_value)
        
        # Execute query with limit + 1 to check for more results
        query = query.limit(limit + 1)
        result = await self.db.execute(query)
        traces = result.scalars().all()
        
        # Check if there are more results
        has_more = len(traces) > limit
        if has_more:
            traces = traces[:limit]
        
        # Generate next cursor
        next_cursor = None
        if has_more and traces:
            last_trace = traces[-1]
            cursor_value = getattr(last_trace, sort_by)
            next_cursor = self._encode_cursor(cursor_value)
        
        # Convert traces to dictionaries
        data = []
        for trace in traces:
            trace_dict = {
                "id": str(trace.id),
                "timestamp": trace.timestamp.isoformat(),
                "user_input": trace.user_input,
                "model_output": trace.model_output,
                "model_name": trace.model_name,
                "system_prompt": trace.system_prompt,
                "session_id": trace.session_id,
                "status": trace.status,
                "latency_ms": trace.latency_ms,
                "cost_usd": trace.cost_usd,
                "metadata": trace.metadata
            }
            
            # Add evaluation data
            if trace.evaluations:
                eval_data = []
                for evaluation in trace.evaluations:
                    eval_dict = {
                        "id": str(evaluation.id),
                        "evaluator_type": evaluation.evaluator_type,
                        "score": evaluation.score,
                        "label": evaluation.label,
                        "critique": evaluation.critique
                    }
                    eval_data.append(eval_dict)
                trace_dict["evaluations"] = eval_data
            
            data.append(trace_dict)
        
        # Get total count if requested (expensive operation)
        total_count = None
        if include_total:
            count_query = select(func.count(Trace.id))
            if filters:
                conditions = self._build_filter_conditions(filters)
                if conditions:
                    count_query = count_query.where(and_(*conditions))
            
            total_result = await self.db.execute(count_query)
            total_count = total_result.scalar()
        
        # Build pagination metadata
        pagination = {
            "cursor": cursor,
            "next_cursor": next_cursor,
            "has_more": has_more,
            "limit": limit,
            "sort_by": sort_by,
            "sort_order": sort_order
        }
        
        if total_count is not None:
            pagination["total_count"] = total_count
        
        metadata = {
            "query_time": time.time(),
            "record_count": len(data)
        }
        
        return PaginatedResponse(
            data=data,
            pagination=pagination,
            metadata=metadata
        )
    
    def _build_filter_conditions(self, filters: Dict[str, Any]) -> List:
        """Build SQLAlchemy filter conditions from filter dictionary."""
        
        conditions = []
        
        if filters.get("model_names"):
            conditions.append(Trace.model_name.in_(filters["model_names"]))
        
        if filters.get("date_from"):
            date_from = datetime.fromisoformat(filters["date_from"].replace('Z', '+00:00'))
            conditions.append(Trace.timestamp >= date_from)
        
        if filters.get("date_to"):
            date_to = datetime.fromisoformat(filters["date_to"].replace('Z', '+00:00'))
            conditions.append(Trace.timestamp <= date_to)
        
        if filters.get("session_ids"):
            conditions.append(Trace.session_id.in_(filters["session_ids"]))
        
        if filters.get("status"):
            conditions.append(Trace.status == filters["status"])
        
        if filters.get("min_cost"):
            conditions.append(Trace.cost_usd >= filters["min_cost"])
        
        if filters.get("max_cost"):
            conditions.append(Trace.cost_usd <= filters["max_cost"])
        
        return conditions
    
    def _encode_cursor(self, value) -> str:
        """Encode cursor value to string."""
        if isinstance(value, datetime):
            return value.isoformat()
        return str(value)
    
    def _decode_cursor(self, cursor: str):
        """Decode cursor string to value."""
        try:
            # Try to parse as datetime first
            return datetime.fromisoformat(cursor.replace('Z', '+00:00'))
        except ValueError:
            # Return as string if not a datetime
            return cursor


# ============================================================================
# API ENDPOINTS
# ============================================================================

@router.post("/stream-export")
async def create_streaming_export(
    request: StreamingExportRequest,
    db: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user_email)
):
    """Create a streaming export of traces."""
    
    try:
        exporter = StreamingExporter(db)
        
        # Determine content type and filename
        content_type = "application/json"
        filename = f"traces_export_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        if request.format.lower() == "csv":
            content_type = "text/csv"
            filename += ".csv"
        elif request.format.lower() == "jsonl":
            content_type = "application/x-jsonlines"
            filename += ".jsonl"
        else:
            filename += ".json"
        
        if request.compress:
            content_type = "application/gzip"
            filename += ".gz"
        
        # Create streaming response
        stream = exporter.stream_traces(
            format=request.format,
            filters=request.filters,
            fields=request.fields,
            chunk_size=request.chunk_size,
            include_evaluations=request.include_evaluations,
            compress=request.compress
        )
        
        headers = {
            "Content-Disposition": f"attachment; filename={filename}",
            "X-Export-Format": request.format,
            "X-Export-Compressed": str(request.compress).lower()
        }
        
        return StreamingResponse(
            stream,
            media_type=content_type,
            headers=headers
        )
        
    except Exception as e:
        logger.error(f"Streaming export error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


@router.post("/background-job", response_model=Dict[str, Any])
async def create_background_job(
    request: BackgroundJobRequest,
    background_tasks: BackgroundTasks,
    current_user: str = Depends(get_current_user_email)
):
    """Create a background job for long-running operations."""
    
    try:
        job_id = await job_manager.create_job(
            job_type=request.job_type,
            parameters=request.parameters,
            user_email=current_user,
            priority=request.priority,
            callback_url=request.callback_url,
            email_notification=request.email_notification
        )
        
        return {
            "job_id": job_id,
            "status": "queued",
            "message": "Background job created successfully",
            "check_status_url": f"/api/large-dataset/job/{job_id}"
        }
        
    except Exception as e:
        logger.error(f"Background job creation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create job: {str(e)}")


@router.get("/job/{job_id}", response_model=BackgroundJobStatus)
async def get_job_status(
    job_id: str,
    current_user: str = Depends(get_current_user_email)
):
    """Get status of a background job."""
    
    status = await job_manager.get_job_status(job_id)
    if not status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Check if user owns the job
    job_data = job_manager.jobs.get(job_id)
    if job_data and job_data["user_email"] != current_user:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return status


@router.delete("/job/{job_id}")
async def cancel_job(
    job_id: str,
    current_user: str = Depends(get_current_user_email)
):
    """Cancel a background job."""
    
    success = await job_manager.cancel_job(job_id, current_user)
    if not success:
        raise HTTPException(status_code=404, detail="Job not found or cannot be cancelled")
    
    return {"message": "Job cancelled successfully"}


@router.get("/jobs", response_model=List[BackgroundJobStatus])
async def list_user_jobs(
    status_filter: Optional[str] = None,
    limit: int = 50,
    current_user: str = Depends(get_current_user_email)
):
    """List background jobs for the current user."""
    
    jobs = await job_manager.list_user_jobs(
        user_email=current_user,
        status_filter=status_filter,
        limit=limit
    )
    
    return jobs


@router.post("/paginate", response_model=PaginatedResponse)
async def paginate_traces(
    request: PaginationRequest,
    db: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user_email)
):
    """Get paginated traces with advanced filtering."""
    
    try:
        paginator = AdvancedPaginator(db)
        
        result = await paginator.paginate_traces(
            cursor=request.cursor,
            limit=request.limit,
            sort_by=request.sort_by,
            sort_order=request.sort_order,
            filters=request.filters,
            include_total=request.include_total
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Pagination error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Pagination failed: {str(e)}")


@router.get("/performance-metrics")
async def get_performance_metrics(
    current_user: str = Depends(get_current_user_email)
):
    """Get system performance metrics."""
    
    # Mock performance metrics (in production, collect from monitoring systems)
    return {
        "database": {
            "active_connections": 25,
            "max_connections": 100,
            "avg_query_time_ms": 45.2,
            "slow_queries_count": 3
        },
        "memory": {
            "usage_mb": 512,
            "available_mb": 2048,
            "usage_percentage": 25.0
        },
        "jobs": {
            "running": len([j for j in job_manager.jobs.values() if j["status"] == "running"]),
            "queued": len([j for j in job_manager.jobs.values() if j["status"] == "queued"]),
            "completed_today": len([
                j for j in job_manager.jobs.values() 
                if j["status"] == "completed" and 
                j["completed_at"] and 
                j["completed_at"].date() == datetime.utcnow().date()
            ])
        },
        "exports": {
            "active_streams": 2,
            "total_exported_today": 15420,
            "largest_export_records": 1250000
        },
        "cache": {
            "hit_rate": 0.85,
            "entries": 1250,
            "memory_usage_mb": 64
        }
    }


@router.get("/health")
async def large_dataset_health():
    """Health check for large dataset handling service."""
    
    return {
        "status": "healthy",
        "service": "Large Dataset Handler",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "features": [
            "Streaming Exports",
            "Background Job Processing", 
            "Advanced Pagination",
            "Performance Monitoring"
        ],
        "limits": {
            "max_export_records": 10000000,
            "max_concurrent_jobs": 50,
            "max_page_size": 10000,
            "streaming_chunk_size": 1000
        }
    } 