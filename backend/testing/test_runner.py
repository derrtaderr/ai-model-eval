"""
Test runner service for executing test suites against LLM outputs.
"""

import asyncio
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from uuid import UUID, uuid4
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from database.models import TestCase, TestRun, Trace
from database.connection import AsyncSessionLocal
from .assertions import create_assertion, AssertionOutcome, AssertionResult


@dataclass
class TestExecutionResult:
    """Result of executing a single test case."""
    test_case_id: str
    test_run_id: str
    status: str  # "passed", "failed", "error"
    assertion_results: List[Dict[str, Any]]
    execution_time_ms: int
    error_message: Optional[str] = None
    trace_id: Optional[str] = None


@dataclass
class TestSuiteResult:
    """Result of executing a test suite."""
    suite_name: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    error_tests: int
    execution_time_ms: int
    test_results: List[TestExecutionResult]
    summary: Dict[str, Any]


class TestRunner:
    """Engine for running LLM test cases."""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def run_test_case(
        self,
        test_case: TestCase,
        model_output: str,
        trace_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> TestExecutionResult:
        """
        Run a single test case against model output.
        
        Args:
            test_case: The test case to execute
            model_output: The output from the LLM to test
            trace_id: Optional trace ID to link the test run
            context: Additional context for the test
            
        Returns:
            TestExecutionResult with the outcome
        """
        start_time = datetime.now()
        test_run_id = str(uuid4())
        assertion_results = []
        overall_status = "passed"
        error_message = None
        
        try:
            # Create assertion from test case configuration
            assertion = create_assertion(
                assertion_type=test_case.assertion_type,
                name=test_case.name,
                config=test_case.assertion_config or {}
            )
            
            # Execute assertion
            outcome = await assertion.evaluate(
                model_output=model_output,
                expected=test_case.expected_output,
                context=context or {}
            )
            
            # Convert outcome to dict for storage
            assertion_result = {
                "assertion_type": test_case.assertion_type,
                "result": outcome.result.value,
                "message": outcome.message,
                "expected": outcome.expected,
                "actual": outcome.actual,
                "details": outcome.details,
                "execution_time_ms": outcome.execution_time_ms
            }
            assertion_results.append(assertion_result)
            
            # Determine overall test status
            if outcome.result == AssertionResult.FAILED:
                overall_status = "failed"
            elif outcome.result == AssertionResult.ERROR:
                overall_status = "error"
                error_message = outcome.message
        
        except Exception as e:
            overall_status = "error"
            error_message = f"Test execution error: {str(e)}"
            assertion_results.append({
                "assertion_type": test_case.assertion_type,
                "result": "error",
                "message": error_message,
                "expected": None,
                "actual": None,
                "details": None,
                "execution_time_ms": None
            })
        
        execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
        
        # Store test run in database
        await self._store_test_run(
            test_case_id=test_case.id,
            test_run_id=test_run_id,
            status=overall_status,
            result=assertion_results[0] if assertion_results else None,
            error_message=error_message,
            execution_time_ms=execution_time,
            trace_id=trace_id
        )
        
        return TestExecutionResult(
            test_case_id=str(test_case.id),
            test_run_id=test_run_id,
            status=overall_status,
            assertion_results=assertion_results,
            execution_time_ms=execution_time,
            error_message=error_message,
            trace_id=trace_id
        )
    
    async def run_test_suite(
        self,
        test_cases: List[TestCase],
        model_outputs: Dict[str, str],
        suite_name: str = "Default Suite",
        parallel: bool = True,
        context: Optional[Dict[str, Any]] = None
    ) -> TestSuiteResult:
        """
        Run a suite of test cases.
        
        Args:
            test_cases: List of test cases to execute
            model_outputs: Dict mapping test case IDs to model outputs
            suite_name: Name for the test suite
            parallel: Whether to run tests in parallel
            context: Additional context for all tests
            
        Returns:
            TestSuiteResult with aggregated results
        """
        start_time = datetime.now()
        
        if parallel:
            # Run tests in parallel
            tasks = []
            for test_case in test_cases:
                model_output = model_outputs.get(str(test_case.id), "")
                if model_output:  # Only run if we have output for this test
                    task = self.run_test_case(test_case, model_output, context=context)
                    tasks.append(task)
            
            test_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle any exceptions
            valid_results = []
            for i, result in enumerate(test_results):
                if isinstance(result, Exception):
                    # Create error result for failed test
                    error_result = TestExecutionResult(
                        test_case_id=str(test_cases[i].id),
                        test_run_id=str(uuid4()),
                        status="error",
                        assertion_results=[],
                        execution_time_ms=0,
                        error_message=str(result)
                    )
                    valid_results.append(error_result)
                else:
                    valid_results.append(result)
            
            test_results = valid_results
        else:
            # Run tests sequentially
            test_results = []
            for test_case in test_cases:
                model_output = model_outputs.get(str(test_case.id), "")
                if model_output:  # Only run if we have output for this test
                    result = await self.run_test_case(test_case, model_output, context=context)
                    test_results.append(result)
        
        # Calculate summary statistics
        total_tests = len(test_results)
        passed_tests = sum(1 for r in test_results if r.status == "passed")
        failed_tests = sum(1 for r in test_results if r.status == "failed")
        error_tests = sum(1 for r in test_results if r.status == "error")
        
        execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
        
        summary = {
            "pass_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "failure_rate": failed_tests / total_tests if total_tests > 0 else 0,
            "error_rate": error_tests / total_tests if total_tests > 0 else 0,
            "average_execution_time_ms": sum(r.execution_time_ms for r in test_results) / total_tests if total_tests > 0 else 0,
            "fastest_test_ms": min(r.execution_time_ms for r in test_results) if test_results else 0,
            "slowest_test_ms": max(r.execution_time_ms for r in test_results) if test_results else 0
        }
        
        return TestSuiteResult(
            suite_name=suite_name,
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            error_tests=error_tests,
            execution_time_ms=execution_time,
            test_results=test_results,
            summary=summary
        )
    
    async def run_tests_for_trace(
        self,
        trace_id: str,
        test_case_filters: Optional[Dict[str, Any]] = None
    ) -> TestSuiteResult:
        """
        Run all applicable test cases against a specific trace.
        
        Args:
            trace_id: ID of the trace to test
            test_case_filters: Optional filters for selecting test cases
            
        Returns:
            TestSuiteResult with results for all applicable tests
        """
        async with AsyncSessionLocal() as session:
            # Get the trace
            trace_query = select(Trace).where(Trace.id == trace_id)
            trace_result = await session.execute(trace_query)
            trace = trace_result.scalar_one_or_none()
            
            if not trace:
                raise ValueError(f"Trace {trace_id} not found")
            
            # Get applicable test cases
            test_case_query = select(TestCase).where(TestCase.is_active == True)
            
            # Apply filters if provided
            if test_case_filters:
                if "tags" in test_case_filters:
                    # Filter by tags (stored in JSON field)
                    tag_filter = test_case_filters["tags"]
                    if isinstance(tag_filter, list):
                        # Test case tags should contain any of the specified tags
                        for tag in tag_filter:
                            test_case_query = test_case_query.where(
                                TestCase.tags.op('? || ?')(tag)
                            )
            
            test_case_result = await session.execute(test_case_query)
            test_cases = test_case_result.scalars().all()
            
            # Create model outputs dict
            model_outputs = {str(tc.id): trace.model_output for tc in test_cases}
            
            # Run the test suite
            context = {
                "trace_id": trace_id,
                "model_name": trace.model_name,
                "user_input": trace.user_input,
                "metadata": trace.metadata or {}
            }
            
            return await self.run_test_suite(
                test_cases=test_cases,
                model_outputs=model_outputs,
                suite_name=f"Tests for Trace {trace_id}",
                context=context
            )
    
    async def run_regression_tests(
        self,
        baseline_trace_ids: List[str],
        current_trace_ids: List[str],
        test_case_filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run regression tests comparing baseline vs current traces.
        
        Args:
            baseline_trace_ids: IDs of baseline traces
            current_trace_ids: IDs of current traces to compare
            test_case_filters: Optional filters for test case selection
            
        Returns:
            Dict with comparison results and regression analysis
        """
        # Run tests for baseline traces
        baseline_results = []
        for trace_id in baseline_trace_ids:
            result = await self.run_tests_for_trace(trace_id, test_case_filters)
            baseline_results.append(result)
        
        # Run tests for current traces
        current_results = []
        for trace_id in current_trace_ids:
            result = await self.run_tests_for_trace(trace_id, test_case_filters)
            current_results.append(result)
        
        # Calculate regression metrics
        baseline_pass_rates = [r.summary["pass_rate"] for r in baseline_results]
        current_pass_rates = [r.summary["pass_rate"] for r in current_results]
        
        avg_baseline_pass_rate = sum(baseline_pass_rates) / len(baseline_pass_rates) if baseline_pass_rates else 0
        avg_current_pass_rate = sum(current_pass_rates) / len(current_pass_rates) if current_pass_rates else 0
        
        regression_detected = avg_current_pass_rate < avg_baseline_pass_rate - 0.05  # 5% threshold
        
        return {
            "baseline_results": baseline_results,
            "current_results": current_results,
            "regression_analysis": {
                "average_baseline_pass_rate": avg_baseline_pass_rate,
                "average_current_pass_rate": avg_current_pass_rate,
                "pass_rate_delta": avg_current_pass_rate - avg_baseline_pass_rate,
                "regression_detected": regression_detected,
                "threshold": 0.05
            },
            "comparison_summary": {
                "baseline_trace_count": len(baseline_trace_ids),
                "current_trace_count": len(current_trace_ids),
                "test_cases_executed": len(baseline_results[0].test_results) if baseline_results else 0
            }
        }
    
    async def _store_test_run(
        self,
        test_case_id: UUID,
        test_run_id: str,
        status: str,
        result: Optional[Dict[str, Any]],
        error_message: Optional[str],
        execution_time_ms: int,
        trace_id: Optional[str] = None
    ):
        """Store test run results in the database."""
        async with AsyncSessionLocal() as session:
            test_run = TestRun(
                id=test_run_id,
                test_case_id=test_case_id,
                trace_id=trace_id,
                status=status,
                result=result,
                error_message=error_message,
                execution_time_ms=execution_time_ms,
                executed_at=datetime.utcnow()
            )
            
            session.add(test_run)
            await session.commit()
    
    def __del__(self):
        """Cleanup executor on deletion."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)


# Global test runner instance
test_runner = TestRunner() 