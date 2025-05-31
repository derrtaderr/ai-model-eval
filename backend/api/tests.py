"""
API endpoints for test case management and execution.
"""

from typing import Dict, List, Any, Optional
from uuid import UUID
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from ..auth.security import get_current_user_email
from ..database.connection import get_db
from ..database.models import TestCase, TestRun, Trace
from ..testing.test_runner import test_runner, TestSuiteResult
from ..testing.assertions import ASSERTION_REGISTRY


router = APIRouter()


class TestCaseCreate(BaseModel):
    """Schema for creating a test case."""
    name: str = Field(..., description="Name of the test case")
    description: Optional[str] = Field(None, description="Description of what this test validates")
    input_data: Dict[str, Any] = Field(..., description="Input data for the test")
    expected_output: Optional[Any] = Field(None, description="Expected output or validation criteria")
    assertion_type: str = Field(..., description="Type of assertion (contains, regex, sentiment, etc.)")
    assertion_config: Optional[Dict[str, Any]] = Field(None, description="Configuration for the assertion")
    tags: Optional[List[str]] = Field(None, description="Tags for categorizing the test")


class TestCaseResponse(BaseModel):
    """Schema for test case response."""
    id: str
    name: str
    description: Optional[str]
    input_data: Dict[str, Any]
    expected_output: Optional[Any]
    assertion_type: str
    assertion_config: Optional[Dict[str, Any]]
    tags: Optional[List[str]]
    is_active: bool
    created_at: datetime
    created_by: Optional[str]


class TestRunRequest(BaseModel):
    """Schema for running tests."""
    test_case_ids: Optional[List[str]] = Field(None, description="Specific test case IDs to run")
    trace_ids: Optional[List[str]] = Field(None, description="Trace IDs to test against")
    model_outputs: Optional[Dict[str, str]] = Field(None, description="Manual model outputs to test")
    suite_name: Optional[str] = Field("API Test Suite", description="Name for the test suite")
    parallel: bool = Field(True, description="Run tests in parallel")
    filters: Optional[Dict[str, Any]] = Field(None, description="Filters for test case selection")


class TestRunResponse(BaseModel):
    """Schema for test run response."""
    id: str
    test_case_id: str
    trace_id: Optional[str]
    status: str
    result: Optional[Dict[str, Any]]
    error_message: Optional[str]
    execution_time_ms: Optional[int]
    executed_at: datetime


class RegressionTestRequest(BaseModel):
    """Schema for regression test request."""
    baseline_trace_ids: List[str] = Field(..., description="Baseline trace IDs")
    current_trace_ids: List[str] = Field(..., description="Current trace IDs to compare")
    test_case_filters: Optional[Dict[str, Any]] = Field(None, description="Filters for test case selection")


@router.post("/test-cases", response_model=Dict[str, str], status_code=status.HTTP_201_CREATED)
async def create_test_case(
    test_case_data: TestCaseCreate,
    current_user_email: str = Depends(get_current_user_email),
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new test case.
    
    Test cases define automated validations that can be run against LLM outputs.
    """
    try:
        # Validate assertion type
        if test_case_data.assertion_type not in ASSERTION_REGISTRY:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid assertion type. Available types: {list(ASSERTION_REGISTRY.keys())}"
            )
        
        # Create test case
        test_case = TestCase(
            name=test_case_data.name,
            description=test_case_data.description,
            input_data=test_case_data.input_data,
            expected_output=test_case_data.expected_output,
            assertion_type=test_case_data.assertion_type,
            assertion_config=test_case_data.assertion_config,
            tags=test_case_data.tags,
            is_active=True,
            created_at=datetime.utcnow(),
            # TODO: Link to actual user when user management is implemented
            created_by=None
        )
        
        db.add(test_case)
        await db.commit()
        await db.refresh(test_case)
        
        return {"test_case_id": str(test_case.id), "message": "Test case created successfully"}
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create test case: {str(e)}"
        )


@router.get("/test-cases", response_model=List[TestCaseResponse])
async def get_test_cases(
    limit: int = Query(100, ge=1, le=1000, description="Number of test cases to return"),
    offset: int = Query(0, ge=0, description="Number of test cases to skip"),
    active_only: bool = Query(True, description="Only return active test cases"),
    tags: Optional[str] = Query(None, description="Filter by tags (comma-separated)"),
    assertion_type: Optional[str] = Query(None, description="Filter by assertion type"),
    current_user_email: str = Depends(get_current_user_email),
    db: AsyncSession = Depends(get_db)
):
    """
    Retrieve test cases with optional filtering.
    
    Returns a list of test cases that can be filtered by various criteria.
    """
    try:
        query = select(TestCase)
        
        # Apply filters
        if active_only:
            query = query.where(TestCase.is_active == True)
        
        if assertion_type:
            query = query.where(TestCase.assertion_type == assertion_type)
        
        if tags:
            tag_list = [tag.strip() for tag in tags.split(",")]
            # Filter test cases that have any of the specified tags
            for tag in tag_list:
                query = query.where(TestCase.tags.op('? || ?')(tag))
        
        query = query.order_by(TestCase.created_at.desc()).limit(limit).offset(offset)
        
        result = await db.execute(query)
        test_cases = result.scalars().all()
        
        # Convert to response format
        response = []
        for tc in test_cases:
            response.append(TestCaseResponse(
                id=str(tc.id),
                name=tc.name,
                description=tc.description,
                input_data=tc.input_data,
                expected_output=tc.expected_output,
                assertion_type=tc.assertion_type,
                assertion_config=tc.assertion_config,
                tags=tc.tags,
                is_active=tc.is_active,
                created_at=tc.created_at,
                created_by=str(tc.created_by) if tc.created_by else None
            ))
        
        return response
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve test cases: {str(e)}"
        )


@router.get("/test-cases/{test_case_id}", response_model=TestCaseResponse)
async def get_test_case(
    test_case_id: str,
    current_user_email: str = Depends(get_current_user_email),
    db: AsyncSession = Depends(get_db)
):
    """
    Get a specific test case by its ID.
    
    Returns detailed information about a single test case.
    """
    try:
        test_case_uuid = UUID(test_case_id)
        
        query = select(TestCase).where(TestCase.id == test_case_uuid)
        result = await db.execute(query)
        test_case = result.scalar_one_or_none()
        
        if not test_case:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Test case not found"
            )
        
        return TestCaseResponse(
            id=str(test_case.id),
            name=test_case.name,
            description=test_case.description,
            input_data=test_case.input_data,
            expected_output=test_case.expected_output,
            assertion_type=test_case.assertion_type,
            assertion_config=test_case.assertion_config,
            tags=test_case.tags,
            is_active=test_case.is_active,
            created_at=test_case.created_at,
            created_by=str(test_case.created_by) if test_case.created_by else None
        )
    
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid test case ID format"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve test case: {str(e)}"
        )


@router.post("/test-runs", response_model=Dict[str, Any])
async def run_tests(
    test_run_request: TestRunRequest,
    current_user_email: str = Depends(get_current_user_email),
    db: AsyncSession = Depends(get_db)
):
    """
    Execute test cases against traces or provided outputs.
    
    Can run specific test cases or all applicable tests for given traces.
    """
    try:
        # Get test cases to run
        test_cases = []
        
        if test_run_request.test_case_ids:
            # Run specific test cases
            test_case_uuids = [UUID(tc_id) for tc_id in test_run_request.test_case_ids]
            query = select(TestCase).where(
                and_(TestCase.id.in_(test_case_uuids), TestCase.is_active == True)
            )
            result = await db.execute(query)
            test_cases = result.scalars().all()
        else:
            # Run all active test cases
            query = select(TestCase).where(TestCase.is_active == True)
            
            # Apply filters if provided
            if test_run_request.filters:
                if "tags" in test_run_request.filters:
                    tag_filter = test_run_request.filters["tags"]
                    if isinstance(tag_filter, list):
                        for tag in tag_filter:
                            query = query.where(TestCase.tags.op('? || ?')(tag))
                if "assertion_type" in test_run_request.filters:
                    query = query.where(TestCase.assertion_type == test_run_request.filters["assertion_type"])
            
            result = await db.execute(query)
            test_cases = result.scalars().all()
        
        if not test_cases:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No test cases found to execute"
            )
        
        # Prepare model outputs
        model_outputs = {}
        
        if test_run_request.model_outputs:
            # Use provided outputs
            model_outputs = test_run_request.model_outputs
        elif test_run_request.trace_ids:
            # Get outputs from traces
            trace_uuids = [UUID(trace_id) for trace_id in test_run_request.trace_ids]
            trace_query = select(Trace).where(Trace.id.in_(trace_uuids))
            trace_result = await db.execute(trace_query)
            traces = trace_result.scalars().all()
            
            # Map each test case to each trace output
            for test_case in test_cases:
                for trace in traces:
                    model_outputs[f"{test_case.id}_{trace.id}"] = trace.model_output
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either trace_ids or model_outputs must be provided"
            )
        
        # Run the test suite
        suite_result = await test_runner.run_test_suite(
            test_cases=test_cases,
            model_outputs=model_outputs,
            suite_name=test_run_request.suite_name,
            parallel=test_run_request.parallel
        )
        
        # Convert dataclass to dict for JSON response
        return {
            "suite_name": suite_result.suite_name,
            "total_tests": suite_result.total_tests,
            "passed_tests": suite_result.passed_tests,
            "failed_tests": suite_result.failed_tests,
            "error_tests": suite_result.error_tests,
            "execution_time_ms": suite_result.execution_time_ms,
            "summary": suite_result.summary,
            "test_results": [
                {
                    "test_case_id": result.test_case_id,
                    "test_run_id": result.test_run_id,
                    "status": result.status,
                    "assertion_results": result.assertion_results,
                    "execution_time_ms": result.execution_time_ms,
                    "error_message": result.error_message,
                    "trace_id": result.trace_id
                }
                for result in suite_result.test_results
            ]
        }
    
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to run tests: {str(e)}"
        )


@router.post("/test-runs/trace/{trace_id}", response_model=Dict[str, Any])
async def run_tests_for_trace(
    trace_id: str,
    test_case_filters: Optional[Dict[str, Any]] = None,
    current_user_email: str = Depends(get_current_user_email),
    db: AsyncSession = Depends(get_db)
):
    """
    Run all applicable test cases against a specific trace.
    
    Automatically executes relevant tests for the given trace.
    """
    try:
        trace_uuid = UUID(trace_id)
        
        # Verify trace exists
        trace_query = select(Trace).where(Trace.id == trace_uuid)
        trace_result = await db.execute(trace_query)
        trace = trace_result.scalar_one_or_none()
        
        if not trace:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Trace not found"
            )
        
        # Run tests for the trace
        suite_result = await test_runner.run_tests_for_trace(
            trace_id=trace_id,
            test_case_filters=test_case_filters
        )
        
        # Convert dataclass to dict for JSON response
        return {
            "suite_name": suite_result.suite_name,
            "total_tests": suite_result.total_tests,
            "passed_tests": suite_result.passed_tests,
            "failed_tests": suite_result.failed_tests,
            "error_tests": suite_result.error_tests,
            "execution_time_ms": suite_result.execution_time_ms,
            "summary": suite_result.summary,
            "test_results": [
                {
                    "test_case_id": result.test_case_id,
                    "test_run_id": result.test_run_id,
                    "status": result.status,
                    "assertion_results": result.assertion_results,
                    "execution_time_ms": result.execution_time_ms,
                    "error_message": result.error_message,
                    "trace_id": result.trace_id
                }
                for result in suite_result.test_results
            ]
        }
    
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid trace ID format"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to run tests for trace: {str(e)}"
        )


@router.post("/test-runs/regression", response_model=Dict[str, Any])
async def run_regression_tests(
    regression_request: RegressionTestRequest,
    current_user_email: str = Depends(get_current_user_email),
    db: AsyncSession = Depends(get_db)
):
    """
    Run regression tests comparing baseline vs current traces.
    
    Compares test results between baseline and current implementations.
    """
    try:
        # Run regression analysis
        regression_result = await test_runner.run_regression_tests(
            baseline_trace_ids=regression_request.baseline_trace_ids,
            current_trace_ids=regression_request.current_trace_ids,
            test_case_filters=regression_request.test_case_filters
        )
        
        return regression_result
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to run regression tests: {str(e)}"
        )


@router.get("/test-runs", response_model=List[TestRunResponse])
async def get_test_runs(
    limit: int = Query(100, ge=1, le=1000, description="Number of test runs to return"),
    offset: int = Query(0, ge=0, description="Number of test runs to skip"),
    test_case_id: Optional[str] = Query(None, description="Filter by test case ID"),
    trace_id: Optional[str] = Query(None, description="Filter by trace ID"),
    status: Optional[str] = Query(None, description="Filter by status (passed, failed, error)"),
    current_user_email: str = Depends(get_current_user_email),
    db: AsyncSession = Depends(get_db)
):
    """
    Retrieve test run history with optional filtering.
    
    Returns a list of test runs that can be filtered by various criteria.
    """
    try:
        query = select(TestRun)
        
        # Apply filters
        if test_case_id:
            test_case_uuid = UUID(test_case_id)
            query = query.where(TestRun.test_case_id == test_case_uuid)
        
        if trace_id:
            trace_uuid = UUID(trace_id)
            query = query.where(TestRun.trace_id == trace_uuid)
        
        if status:
            query = query.where(TestRun.status == status)
        
        query = query.order_by(TestRun.executed_at.desc()).limit(limit).offset(offset)
        
        result = await db.execute(query)
        test_runs = result.scalars().all()
        
        # Convert to response format
        response = []
        for tr in test_runs:
            response.append(TestRunResponse(
                id=str(tr.id),
                test_case_id=str(tr.test_case_id),
                trace_id=str(tr.trace_id) if tr.trace_id else None,
                status=tr.status,
                result=tr.result,
                error_message=tr.error_message,
                execution_time_ms=tr.execution_time_ms,
                executed_at=tr.executed_at
            ))
        
        return response
    
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid ID format: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve test runs: {str(e)}"
        )


@router.get("/assertions/types", response_model=List[str])
async def get_assertion_types(
    current_user_email: str = Depends(get_current_user_email)
):
    """
    Get available assertion types for test cases.
    
    Returns a list of all supported assertion types.
    """
    return list(ASSERTION_REGISTRY.keys())


@router.delete("/test-cases/{test_case_id}", response_model=Dict[str, str])
async def delete_test_case(
    test_case_id: str,
    current_user_email: str = Depends(get_current_user_email),
    db: AsyncSession = Depends(get_db)
):
    """
    Delete a test case (soft delete - marks as inactive).
    
    Marks the test case as inactive rather than permanently deleting it.
    """
    try:
        test_case_uuid = UUID(test_case_id)
        
        query = select(TestCase).where(TestCase.id == test_case_uuid)
        result = await db.execute(query)
        test_case = result.scalar_one_or_none()
        
        if not test_case:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Test case not found"
            )
        
        # Soft delete (mark as inactive)
        test_case.is_active = False
        await db.commit()
        
        return {"message": "Test case deleted successfully", "test_case_id": test_case_id}
    
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid test case ID format"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete test case: {str(e)}"
        ) 