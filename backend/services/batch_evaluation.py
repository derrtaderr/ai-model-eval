"""
Batch Evaluation Processing Service
Scalable system for processing thousands of traces with model-based evaluation.
Part of Task 6.4 - Build Batch Processing Capabilities.
"""

import asyncio
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import deque

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_

from database.models import Trace, Evaluation
from database.connection import get_db
from services.evaluator_models import evaluator_manager, EvaluationRequest, EvaluationResult, EvaluationCriteria
from config.settings import get_settings

logger = logging.getLogger(__name__)

class BatchStrategy(str, Enum):
    """Batch processing strategies."""
    FIFO = "fifo"  # First In, First Out
    PRIORITY = "priority"  # Priority-based processing
    CHUNKED = "chunked"  # Process in optimized chunks
    COST_OPTIMIZED = "cost_optimized"  # Minimize API costs

class BatchStatus(str, Enum):
    """Batch processing status."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TaskPriority(str, Enum):
    """Task priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class BatchTask:
    """Individual task in a batch."""
    id: str
    trace_id: str
    evaluation_request: EvaluationRequest
    priority: TaskPriority = TaskPriority.MEDIUM
    retries: int = 0
    max_retries: int = 3
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None
    result: Optional[EvaluationResult] = None
    estimated_cost: float = 0.0
    actual_cost: float = 0.0

@dataclass
class BatchJob:
    """Batch processing job."""
    id: str
    name: Optional[str]
    description: Optional[str]
    tasks: List[BatchTask]
    strategy: BatchStrategy
    status: BatchStatus = BatchStatus.PENDING
    parallel_workers: int = 5
    created_by: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    paused_at: Optional[str] = None
    progress: Dict[str, Any] = field(default_factory=dict)
    settings: Dict[str, Any] = field(default_factory=dict)
    
    # Statistics
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    skipped_tasks: int = 0
    
    # Cost tracking
    estimated_total_cost: float = 0.0
    actual_total_cost: float = 0.0
    
    # Error tracking
    errors: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class BatchProgress:
    """Real-time batch processing progress."""
    job_id: str
    status: BatchStatus
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    skipped_tasks: int
    current_workers: int
    progress_percentage: float
    estimated_time_remaining_seconds: Optional[int]
    throughput_tasks_per_minute: float
    average_task_duration_ms: float
    estimated_total_cost: float
    actual_total_cost: float
    last_updated: str = field(default_factory=lambda: datetime.utcnow().isoformat())

class BatchProcessor:
    """Core batch processing engine."""
    
    def __init__(self, max_workers: int = 10):
        self.max_workers = max_workers
        self.settings = get_settings()
        
        # Job management
        self.active_jobs: Dict[str, BatchJob] = {}
        self.job_queues: Dict[str, deque] = {}
        self.job_workers: Dict[str, List[asyncio.Task]] = {}
        self.job_locks: Dict[str, asyncio.Lock] = {}
        
        # Global stats
        self.total_processed = 0
        self.total_errors = 0
        self.start_time = datetime.utcnow()
        
        # Shutdown flag
        self._shutdown_requested = False
        
        logger.info(f"BatchProcessor initialized with max_workers={max_workers}")
    
    async def create_batch_job(
        self,
        trace_ids: List[str],
        criteria: List[EvaluationCriteria],
        evaluator_model: Optional[str] = None,
        strategy: BatchStrategy = BatchStrategy.FIFO,
        parallel_workers: int = 5,
        job_name: Optional[str] = None,
        description: Optional[str] = None,
        priority: TaskPriority = TaskPriority.MEDIUM,
        use_calibration: bool = True,
        custom_prompt: Optional[str] = None,
        created_by: Optional[str] = None
    ) -> BatchJob:
        """Create a new batch processing job."""
        
        job_id = str(uuid.uuid4())
        
        # Fetch trace data from database
        async with get_db() as session:
            traces_query = select(Trace).where(Trace.id.in_(trace_ids))
            result = await session.execute(traces_query)
            traces = result.scalars().all()
        
        if not traces:
            raise ValueError("No valid traces found for the provided IDs")
        
        # Create tasks for each trace-criteria combination
        tasks = []
        total_estimated_cost = 0.0
        
        for trace in traces:
            for criterion in criteria:
                task_id = str(uuid.uuid4())
                
                # Create evaluation request
                eval_request = EvaluationRequest(
                    trace_id=trace.id,
                    user_input=trace.user_input,
                    model_output=trace.model_output,
                    system_prompt=trace.system_prompt,
                    context=trace.metadata or {},
                    criteria=[criterion],
                    custom_prompt=custom_prompt
                )
                
                # Estimate cost
                if evaluator_model and evaluator_model in evaluator_manager.evaluators:
                    evaluator = evaluator_manager.evaluators[evaluator_model]
                    estimated_cost = await evaluator.get_cost_estimate(eval_request)
                else:
                    # Use default estimator
                    estimated_cost = await evaluator_manager.evaluators.get("openai/gpt-4", evaluator_manager.evaluators["openai/gpt-3.5-turbo"]).get_cost_estimate(eval_request)
                
                task = BatchTask(
                    id=task_id,
                    trace_id=trace.id,
                    evaluation_request=eval_request,
                    priority=priority,
                    estimated_cost=estimated_cost
                )
                
                tasks.append(task)
                total_estimated_cost += estimated_cost
        
        # Create batch job
        job = BatchJob(
            id=job_id,
            name=job_name or f"Batch Evaluation {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}",
            description=description,
            tasks=tasks,
            strategy=strategy,
            parallel_workers=min(parallel_workers, self.max_workers),
            created_by=created_by,
            total_tasks=len(tasks),
            estimated_total_cost=total_estimated_cost,
            settings={
                "evaluator_model": evaluator_model,
                "use_calibration": use_calibration,
                "criteria": [c.value for c in criteria]
            }
        )
        
        # Initialize job management structures
        self.active_jobs[job_id] = job
        self.job_queues[job_id] = deque(self._sort_tasks_by_strategy(tasks, strategy))
        self.job_workers[job_id] = []
        self.job_locks[job_id] = asyncio.Lock()
        
        logger.info(f"Created batch job {job_id} with {len(tasks)} tasks, estimated cost: ${total_estimated_cost:.4f}")
        
        return job
    
    def _sort_tasks_by_strategy(self, tasks: List[BatchTask], strategy: BatchStrategy) -> List[BatchTask]:
        """Sort tasks based on processing strategy."""
        if strategy == BatchStrategy.FIFO:
            return tasks  # Maintain original order
        elif strategy == BatchStrategy.PRIORITY:
            priority_order = {TaskPriority.CRITICAL: 0, TaskPriority.HIGH: 1, TaskPriority.MEDIUM: 2, TaskPriority.LOW: 3}
            return sorted(tasks, key=lambda t: priority_order.get(t.priority, 2))
        elif strategy == BatchStrategy.CHUNKED:
            # Group by trace_id to optimize context switching
            grouped = {}
            for task in tasks:
                if task.trace_id not in grouped:
                    grouped[task.trace_id] = []
                grouped[task.trace_id].append(task)
            
            # Flatten groups
            result = []
            for trace_tasks in grouped.values():
                result.extend(trace_tasks)
            return result
        elif strategy == BatchStrategy.COST_OPTIMIZED:
            # Sort by estimated cost (cheapest first) to maximize throughput
            return sorted(tasks, key=lambda t: t.estimated_cost)
        else:
            return tasks
    
    async def start_batch_job(self, job_id: str) -> bool:
        """Start processing a batch job."""
        if job_id not in self.active_jobs:
            raise ValueError(f"Job {job_id} not found")
        
        job = self.active_jobs[job_id]
        
        if job.status != BatchStatus.PENDING:
            raise ValueError(f"Job {job_id} is not in pending state (current: {job.status})")
        
        # Update job status
        job.status = BatchStatus.RUNNING
        job.started_at = datetime.utcnow().isoformat()
        
        # Start worker tasks
        async with self.job_locks[job_id]:
            for i in range(job.parallel_workers):
                worker_task = asyncio.create_task(
                    self._worker_loop(job_id, f"worker-{i}")
                )
                self.job_workers[job_id].append(worker_task)
        
        logger.info(f"Started batch job {job_id} with {job.parallel_workers} workers")
        return True
    
    async def _worker_loop(self, job_id: str, worker_name: str):
        """Main worker loop for processing tasks."""
        logger.info(f"Worker {worker_name} started for job {job_id}")
        
        try:
            while not self._shutdown_requested:
                # Get next task
                task = await self._get_next_task(job_id)
                if task is None:
                    # No more tasks, exit
                    break
                
                # Process task
                await self._process_task(job_id, task, worker_name)
                
                # Check if job should be paused
                job = self.active_jobs[job_id]
                if job.status == BatchStatus.PAUSED:
                    logger.info(f"Worker {worker_name} paused for job {job_id}")
                    while job.status == BatchStatus.PAUSED and not self._shutdown_requested:
                        await asyncio.sleep(1)
                    logger.info(f"Worker {worker_name} resumed for job {job_id}")
                
                if job.status == BatchStatus.CANCELLED:
                    break
        
        except Exception as e:
            logger.error(f"Worker {worker_name} error in job {job_id}: {e}")
        finally:
            logger.info(f"Worker {worker_name} finished for job {job_id}")
    
    async def _get_next_task(self, job_id: str) -> Optional[BatchTask]:
        """Get the next task to process."""
        async with self.job_locks[job_id]:
            queue = self.job_queues[job_id]
            if queue:
                return queue.popleft()
            return None
    
    async def _process_task(self, job_id: str, task: BatchTask, worker_name: str):
        """Process a single task."""
        job = self.active_jobs[job_id]
        
        # Update task status
        task.started_at = datetime.utcnow().isoformat()
        
        try:
            start_time = time.time()
            
            # Get evaluator model
            evaluator_model = job.settings.get("evaluator_model")
            use_calibration = job.settings.get("use_calibration", True)
            
            # Perform evaluation
            if use_calibration:
                result = await evaluator_manager.evaluate_single_with_calibration(
                    task.evaluation_request,
                    evaluator_model
                )
            else:
                result = await evaluator_manager.evaluate_single(
                    task.evaluation_request,
                    evaluator_model
                )
            
            # Update task with result
            task.result = result
            task.completed_at = datetime.utcnow().isoformat()
            task.actual_cost = result.cost_usd or 0.0
            
            # Save to database
            await self._save_evaluation_result(result)
            
            # Update job statistics
            async with self.job_locks[job_id]:
                job.completed_tasks += 1
                job.actual_total_cost += task.actual_cost
                self.total_processed += 1
            
            duration_ms = (time.time() - start_time) * 1000
            logger.debug(f"Worker {worker_name} completed task {task.id} in {duration_ms:.0f}ms")
            
        except Exception as e:
            # Handle task failure
            task.error = str(e)
            task.retries += 1
            
            if task.retries < task.max_retries:
                # Retry the task
                logger.warning(f"Task {task.id} failed (attempt {task.retries}/{task.max_retries}): {e}")
                async with self.job_locks[job_id]:
                    # Add back to queue for retry
                    self.job_queues[job_id].append(task)
            else:
                # Task permanently failed
                logger.error(f"Task {task.id} permanently failed after {task.retries} attempts: {e}")
                async with self.job_locks[job_id]:
                    job.failed_tasks += 1
                    job.errors.append({
                        "task_id": task.id,
                        "trace_id": task.trace_id,
                        "error": str(e),
                        "attempts": task.retries,
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    self.total_errors += 1
        
        # Check if job is complete
        await self._check_job_completion(job_id)
    
    async def _save_evaluation_result(self, result: EvaluationResult):
        """Save evaluation result to database."""
        try:
            async with get_db() as session:
                # Create metadata with additional evaluation details
                eval_metadata = {
                    "criteria": result.criteria.value,
                    "evaluator_model": result.evaluator_model,
                    "confidence": result.confidence,
                    "reasoning": result.reasoning,
                    "evaluation_time_ms": result.evaluation_time_ms,
                    "cost_usd": result.cost_usd
                }
                
                # Add original metadata if present
                if result.metadata:
                    eval_metadata.update(result.metadata)
                
                evaluation = Evaluation(
                    trace_id=result.trace_id,
                    evaluator_type="ai_model",
                    evaluator_id=None,  # AI models don't have user IDs
                    score=result.score,
                    label=None,  # Could be set based on score thresholds if needed
                    critique=result.reasoning,
                    eval_metadata=eval_metadata
                )
                
                session.add(evaluation)
                await session.commit()
        except Exception as e:
            logger.error(f"Failed to save evaluation result for trace {result.trace_id}: {e}")
    
    async def _check_job_completion(self, job_id: str):
        """Check if a job has completed and update status."""
        job = self.active_jobs[job_id]
        
        async with self.job_locks[job_id]:
            total_completed = job.completed_tasks + job.failed_tasks + job.skipped_tasks
            queue_size = len(self.job_queues[job_id])
            
            if total_completed >= job.total_tasks or queue_size == 0:
                # Job is complete
                job.status = BatchStatus.COMPLETED
                job.completed_at = datetime.utcnow().isoformat()
                
                # Cancel remaining workers
                for worker_task in self.job_workers[job_id]:
                    if not worker_task.done():
                        worker_task.cancel()
                
                logger.info(f"Batch job {job_id} completed: {job.completed_tasks} successful, {job.failed_tasks} failed")
    
    async def pause_batch_job(self, job_id: str) -> bool:
        """Pause a running batch job."""
        if job_id not in self.active_jobs:
            raise ValueError(f"Job {job_id} not found")
        
        job = self.active_jobs[job_id]
        
        if job.status != BatchStatus.RUNNING:
            raise ValueError(f"Job {job_id} is not running")
        
        job.status = BatchStatus.PAUSED
        job.paused_at = datetime.utcnow().isoformat()
        
        logger.info(f"Paused batch job {job_id}")
        return True
    
    async def resume_batch_job(self, job_id: str) -> bool:
        """Resume a paused batch job."""
        if job_id not in self.active_jobs:
            raise ValueError(f"Job {job_id} not found")
        
        job = self.active_jobs[job_id]
        
        if job.status != BatchStatus.PAUSED:
            raise ValueError(f"Job {job_id} is not paused")
        
        job.status = BatchStatus.RUNNING
        
        logger.info(f"Resumed batch job {job_id}")
        return True
    
    async def cancel_batch_job(self, job_id: str) -> bool:
        """Cancel a batch job."""
        if job_id not in self.active_jobs:
            raise ValueError(f"Job {job_id} not found")
        
        job = self.active_jobs[job_id]
        
        job.status = BatchStatus.CANCELLED
        
        # Cancel all workers
        for worker_task in self.job_workers.get(job_id, []):
            if not worker_task.done():
                worker_task.cancel()
        
        logger.info(f"Cancelled batch job {job_id}")
        return True
    
    async def get_batch_progress(self, job_id: str) -> BatchProgress:
        """Get real-time progress for a batch job."""
        if job_id not in self.active_jobs:
            raise ValueError(f"Job {job_id} not found")
        
        job = self.active_jobs[job_id]
        
        # Calculate metrics
        total_completed = job.completed_tasks + job.failed_tasks + job.skipped_tasks
        progress_percentage = (total_completed / job.total_tasks * 100) if job.total_tasks > 0 else 0
        
        # Calculate throughput and ETA
        if job.started_at:
            elapsed_seconds = (datetime.utcnow() - datetime.fromisoformat(job.started_at.replace('Z', '+00:00'))).total_seconds()
            throughput = (total_completed / elapsed_seconds * 60) if elapsed_seconds > 0 else 0
            
            remaining_tasks = job.total_tasks - total_completed
            eta_seconds = (remaining_tasks / throughput * 60) if throughput > 0 else None
        else:
            throughput = 0
            eta_seconds = None
        
        # Calculate average task duration
        avg_duration = 0
        if job.completed_tasks > 0:
            total_duration = sum(
                (datetime.fromisoformat(task.completed_at.replace('Z', '+00:00')) - 
                 datetime.fromisoformat(task.started_at.replace('Z', '+00:00'))).total_seconds() * 1000
                for task in job.tasks if task.completed_at and task.started_at
            )
            avg_duration = total_duration / job.completed_tasks
        
        return BatchProgress(
            job_id=job_id,
            status=job.status,
            total_tasks=job.total_tasks,
            completed_tasks=job.completed_tasks,
            failed_tasks=job.failed_tasks,
            skipped_tasks=job.skipped_tasks,
            current_workers=len([w for w in self.job_workers.get(job_id, []) if not w.done()]),
            progress_percentage=progress_percentage,
            estimated_time_remaining_seconds=int(eta_seconds) if eta_seconds else None,
            throughput_tasks_per_minute=throughput,
            average_task_duration_ms=avg_duration,
            estimated_total_cost=job.estimated_total_cost,
            actual_total_cost=job.actual_total_cost
        )
    
    async def list_batch_jobs(self, status_filter: Optional[BatchStatus] = None) -> List[BatchJob]:
        """List all batch jobs, optionally filtered by status."""
        jobs = list(self.active_jobs.values())
        
        if status_filter:
            jobs = [job for job in jobs if job.status == status_filter]
        
        return jobs
    
    async def get_batch_job(self, job_id: str) -> BatchJob:
        """Get a specific batch job."""
        if job_id not in self.active_jobs:
            raise ValueError(f"Job {job_id} not found")
        
        return self.active_jobs[job_id]
    
    async def cleanup_completed_jobs(self, older_than_hours: int = 24):
        """Clean up completed jobs older than specified hours."""
        cutoff_time = datetime.utcnow() - timedelta(hours=older_than_hours)
        
        jobs_to_remove = []
        for job_id, job in self.active_jobs.items():
            if job.status in [BatchStatus.COMPLETED, BatchStatus.CANCELLED, BatchStatus.FAILED]:
                if job.completed_at:
                    completed_time = datetime.fromisoformat(job.completed_at.replace('Z', '+00:00'))
                    if completed_time < cutoff_time:
                        jobs_to_remove.append(job_id)
        
        for job_id in jobs_to_remove:
            # Clean up job data
            self.active_jobs.pop(job_id, None)
            self.job_queues.pop(job_id, None)
            self.job_workers.pop(job_id, None)
            self.job_locks.pop(job_id, None)
            
            logger.info(f"Cleaned up completed job {job_id}")
        
        return len(jobs_to_remove)
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Get system-wide batch processing statistics."""
        active_jobs = len(self.active_jobs)
        running_jobs = len([job for job in self.active_jobs.values() if job.status == BatchStatus.RUNNING])
        total_workers = sum(len(workers) for workers in self.job_workers.values())
        
        uptime_seconds = (datetime.utcnow() - self.start_time).total_seconds()
        
        return {
            "active_jobs": active_jobs,
            "running_jobs": running_jobs,
            "total_workers": total_workers,
            "max_workers": self.max_workers,
            "total_processed": self.total_processed,
            "total_errors": self.total_errors,
            "uptime_seconds": uptime_seconds,
            "success_rate": (self.total_processed / (self.total_processed + self.total_errors)) if (self.total_processed + self.total_errors) > 0 else 0,
            "throughput_per_hour": (self.total_processed / uptime_seconds * 3600) if uptime_seconds > 0 else 0
        }
    
    async def shutdown(self):
        """Gracefully shutdown the batch processor."""
        logger.info("Shutting down batch processor...")
        self._shutdown_requested = True
        
        # Cancel all running workers
        for job_id, workers in self.job_workers.items():
            for worker in workers:
                if not worker.done():
                    worker.cancel()
        
        # Wait for workers to finish
        all_workers = []
        for workers in self.job_workers.values():
            all_workers.extend(workers)
        
        if all_workers:
            await asyncio.gather(*all_workers, return_exceptions=True)
        
        logger.info("Batch processor shutdown complete")

# Global batch processor instance
batch_processor = BatchProcessor() 