"""
Integration tests for LangSmith connector.

Tests the enhanced LangSmith integration including:
- Connection and configuration
- Trace synchronization
- Webhook handling
- Error scenarios
- Performance
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from services.langsmith_connector import LangSmithConnector, LangSmithSync
from database.models import Trace, Evaluation, User, Team
from api.langsmith import router
from main import app


@pytest.fixture
def langsmith_connector():
    """Create a test LangSmith connector."""
    with patch('services.langsmith_connector.config') as mock_config:
        mock_config.return_value = "test-api-key"
        connector = LangSmithConnector()
        connector.client = Mock()
        return connector


@pytest.fixture
def test_client():
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def mock_langsmith_run():
    """Create a mock LangSmith run object."""
    run = Mock()
    run.id = str(uuid4())
    run.inputs = {"input": "test user input", "messages": [{"role": "user", "content": "Hello"}]}
    run.outputs = {"output": "test model output"}
    run.extra = {"model_name": "gpt-4", "token_usage": {"prompt_tokens": 10, "completion_tokens": 20}}
    run.start_time = datetime.utcnow()
    run.end_time = datetime.utcnow() + timedelta(seconds=2)
    run.total_time = 2.0
    run.total_cost = 0.001
    run.session_id = str(uuid4())
    run.run_type = "llm"
    run.tags = ["test"]
    run.error = None
    return run


class TestLangSmithConnectorInit:
    """Test LangSmith connector initialization."""

    def test_init_with_api_key(self):
        """Test connector initialization with API key."""
        with patch('services.langsmith_connector.config') as mock_config:
            mock_config.side_effect = lambda key, default=None: {
                "LANGCHAIN_API_KEY": "test-key",
                "LANGCHAIN_PROJECT": "test-project",
                "LANGSMITH_WEBHOOK_SECRET": "test-secret",
                "LANGSMITH_BASE_URL": "https://api.smith.langchain.com"
            }.get(key, default)
            
            with patch('services.langsmith_connector.Client') as mock_client:
                connector = LangSmithConnector()
                
                assert connector.api_key == "test-key"
                assert connector.project == "test-project"
                assert connector.webhook_secret == "test-secret"
                assert connector.client is not None
                mock_client.assert_called_once_with(api_key="test-key")

    def test_init_without_api_key(self):
        """Test connector initialization without API key."""
        with patch('services.langsmith_connector.config') as mock_config:
            mock_config.return_value = None
            
            connector = LangSmithConnector()
            
            assert connector.api_key is None
            assert connector.client is None

    def test_connection_test_success(self, langsmith_connector):
        """Test successful connection test."""
        langsmith_connector.client.list_datasets.return_value = []
        
        result = langsmith_connector._test_connection()
        
        assert result is True

    def test_connection_test_failure(self, langsmith_connector):
        """Test failed connection test."""
        langsmith_connector.client.list_datasets.side_effect = Exception("Connection failed")
        
        result = langsmith_connector._test_connection()
        
        assert result is False


class TestConnectionStatus:
    """Test connection status functionality."""

    @pytest.mark.asyncio
    async def test_get_connection_status_connected(self, langsmith_connector):
        """Test getting connection status when connected."""
        langsmith_connector.client.list_datasets.return_value = []
        
        with patch('services.langsmith_connector.cache_service') as mock_cache:
            mock_cache.get.return_value = "2023-01-01T12:00:00"
            
            status = await langsmith_connector.get_connection_status()
            
            assert status["connected"] is True
            assert status["project"] == langsmith_connector.project
            assert status["last_sync"] == "2023-01-01T12:00:00"

    @pytest.mark.asyncio
    async def test_get_connection_status_disconnected(self):
        """Test getting connection status when disconnected."""
        connector = LangSmithConnector()
        connector.client = None
        
        status = await connector.get_connection_status()
        
        assert status["connected"] is False
        assert status["error"] == "No API key configured"

    @pytest.mark.asyncio
    async def test_get_connection_status_error(self, langsmith_connector):
        """Test getting connection status with error."""
        langsmith_connector.client.list_datasets.side_effect = Exception("API error")
        
        status = await langsmith_connector.get_connection_status()
        
        assert status["connected"] is False
        assert "API error" in status["error"]


class TestTraceSync:
    """Test trace synchronization functionality."""

    @pytest.mark.asyncio
    async def test_sync_traces_success(self, langsmith_connector, mock_langsmith_run):
        """Test successful trace synchronization."""
        langsmith_connector.client.list_runs.return_value = [mock_langsmith_run]
        
        with patch('services.langsmith_connector.AsyncSessionLocal') as mock_session_local:
            mock_session = AsyncMock()
            mock_session_local.return_value.__aenter__.return_value = mock_session
            mock_session.execute.return_value.scalar_one_or_none.return_value = None  # No existing trace
            
            with patch('services.langsmith_connector.cache_service') as mock_cache:
                mock_cache.get.return_value = None
                mock_cache.set.return_value = None
                
                result = await langsmith_connector.sync_traces_from_langsmith(
                    project_name="test-project",
                    limit=10
                )
                
                assert result.project_name == "test-project"
                assert result.total_synced == 1
                assert len(result.errors) == 0

    @pytest.mark.asyncio
    async def test_sync_traces_with_existing(self, langsmith_connector, mock_langsmith_run):
        """Test sync with existing traces (should skip)."""
        langsmith_connector.client.list_runs.return_value = [mock_langsmith_run]
        
        with patch('services.langsmith_connector.AsyncSessionLocal') as mock_session_local:
            mock_session = AsyncMock()
            mock_session_local.return_value.__aenter__.return_value = mock_session
            
            # Mock existing trace found
            existing_trace = Mock()
            mock_session.execute.return_value.scalar_one_or_none.return_value = existing_trace
            
            with patch('services.langsmith_connector.cache_service') as mock_cache:
                mock_cache.get.return_value = None
                
                result = await langsmith_connector.sync_traces_from_langsmith(
                    project_name="test-project",
                    force_resync=False
                )
                
                assert result.total_synced == 0  # Should skip existing

    @pytest.mark.asyncio
    async def test_sync_traces_force_resync(self, langsmith_connector, mock_langsmith_run):
        """Test sync with force resync."""
        langsmith_connector.client.list_runs.return_value = [mock_langsmith_run]
        
        with patch('services.langsmith_connector.AsyncSessionLocal') as mock_session_local:
            mock_session = AsyncMock()
            mock_session_local.return_value.__aenter__.return_value = mock_session
            
            # Mock existing trace
            existing_trace = Mock()
            existing_trace.team_id = uuid4()
            mock_session.execute.return_value.scalar_one_or_none.return_value = existing_trace
            
            with patch('services.langsmith_connector.cache_service') as mock_cache:
                mock_cache.get.return_value = None
                mock_cache.set.return_value = None
                
                result = await langsmith_connector.sync_traces_from_langsmith(
                    project_name="test-project",
                    force_resync=True
                )
                
                assert result.total_synced == 1  # Should update existing

    @pytest.mark.asyncio
    async def test_convert_langsmith_run_to_trace(self, langsmith_connector, mock_langsmith_run):
        """Test converting LangSmith run to trace format."""
        team_id = uuid4()
        
        with patch('services.langsmith_connector.AsyncSessionLocal') as mock_session_local:
            mock_session = AsyncMock()
            
            trace_data = await langsmith_connector._convert_langsmith_run_to_trace(
                mock_langsmith_run, team_id, mock_session
            )
            
            assert trace_data is not None
            assert trace_data["user_input"] == "Hello"  # From messages
            assert trace_data["model_output"] == "test model output"
            assert trace_data["model_name"] == "gpt-4"
            assert trace_data["team_id"] == team_id
            assert trace_data["langsmith_run_id"] == str(mock_langsmith_run.id)
            assert "langsmith" in trace_data["metadata"]

    @pytest.mark.asyncio
    async def test_sync_without_client(self):
        """Test sync without LangSmith client configured."""
        connector = LangSmithConnector()
        connector.client = None
        
        with pytest.raises(ValueError, match="LangSmith client not configured"):
            await connector.sync_traces_from_langsmith()


class TestEvaluationPush:
    """Test pushing evaluations to LangSmith."""

    @pytest.mark.asyncio
    async def test_push_evaluation_success(self, langsmith_connector):
        """Test successful evaluation push."""
        trace_id = uuid4()
        evaluation_data = {
            "score": 0.8,
            "label": "good",
            "critique": "Well done",
            "evaluator_id": str(uuid4())
        }
        
        with patch('services.langsmith_connector.AsyncSessionLocal') as mock_session_local:
            mock_session = AsyncMock()
            mock_session_local.return_value.__aenter__.return_value = mock_session
            
            # Mock trace with LangSmith run ID
            mock_trace = Mock()
            mock_trace.langsmith_run_id = str(uuid4())
            mock_session.execute.return_value.scalar_one_or_none.return_value = mock_trace
            
            langsmith_connector.client.create_feedback.return_value = True
            
            result = await langsmith_connector.push_evaluation_to_langsmith(
                trace_id, evaluation_data
            )
            
            assert result is True
            langsmith_connector.client.create_feedback.assert_called_once()

    @pytest.mark.asyncio
    async def test_push_evaluation_no_trace(self, langsmith_connector):
        """Test push evaluation with non-existent trace."""
        trace_id = uuid4()
        evaluation_data = {"score": 0.8}
        
        with patch('services.langsmith_connector.AsyncSessionLocal') as mock_session_local:
            mock_session = AsyncMock()
            mock_session_local.return_value.__aenter__.return_value = mock_session
            mock_session.execute.return_value.scalar_one_or_none.return_value = None
            
            result = await langsmith_connector.push_evaluation_to_langsmith(
                trace_id, evaluation_data
            )
            
            assert result is False

    @pytest.mark.asyncio
    async def test_push_evaluation_without_client(self):
        """Test push evaluation without client."""
        connector = LangSmithConnector()
        connector.client = None
        
        result = await connector.push_evaluation_to_langsmith(uuid4(), {})
        
        assert result is False


class TestWebhookHandling:
    """Test webhook handling functionality."""

    @pytest.mark.asyncio
    async def test_handle_run_created_webhook(self, langsmith_connector, mock_langsmith_run):
        """Test handling run.created webhook."""
        payload = {
            "event_type": "run.created",
            "data": {"id": str(mock_langsmith_run.id)}
        }
        
        langsmith_connector.client.read_run.return_value = mock_langsmith_run
        
        with patch('services.langsmith_connector.AsyncSessionLocal') as mock_session_local:
            mock_session = AsyncMock()
            mock_session_local.return_value.__aenter__.return_value = mock_session
            mock_trace = Mock()
            mock_trace.id = uuid4()
            mock_session.add.return_value = None
            
            with patch.object(langsmith_connector, '_convert_langsmith_run_to_trace') as mock_convert:
                mock_convert.return_value = {"user_input": "test", "model_output": "test"}
                
                result = await langsmith_connector.handle_webhook(payload, "")
                
                assert result["status"] == "success"
                assert "trace_id" in result

    @pytest.mark.asyncio
    async def test_handle_feedback_created_webhook(self, langsmith_connector):
        """Test handling feedback.created webhook."""
        run_id = str(uuid4())
        payload = {
            "event_type": "feedback.created",
            "data": {
                "run_id": run_id,
                "score": 0.9,
                "value": "excellent",
                "comment": "Great response"
            }
        }
        
        with patch('services.langsmith_connector.AsyncSessionLocal') as mock_session_local:
            mock_session = AsyncMock()
            mock_session_local.return_value.__aenter__.return_value = mock_session
            
            # Mock trace found
            mock_trace = Mock()
            mock_trace.id = uuid4()
            mock_session.execute.return_value.scalar_one_or_none.return_value = mock_trace
            
            result = await langsmith_connector.handle_webhook(payload, "")
            
            assert result["status"] == "success"
            assert "evaluation_id" in result

    @pytest.mark.asyncio
    async def test_handle_unknown_webhook(self, langsmith_connector):
        """Test handling unknown webhook event."""
        payload = {
            "event_type": "unknown.event",
            "data": {}
        }
        
        result = await langsmith_connector.handle_webhook(payload, "")
        
        assert result["status"] == "ignored"
        assert "Unknown event type" in result["reason"]

    def test_verify_webhook_signature_valid(self, langsmith_connector):
        """Test webhook signature verification with valid signature."""
        langsmith_connector.webhook_secret = "test-secret"
        payload = {"test": "data"}
        
        # Mock the signature verification
        with patch('hmac.compare_digest') as mock_compare:
            mock_compare.return_value = True
            
            result = langsmith_connector._verify_webhook_signature(payload, "valid-signature")
            
            assert result is True

    def test_verify_webhook_signature_invalid(self, langsmith_connector):
        """Test webhook signature verification with invalid signature."""
        langsmith_connector.webhook_secret = "test-secret"
        payload = {"test": "data"}
        
        with patch('hmac.compare_digest') as mock_compare:
            mock_compare.return_value = False
            
            result = langsmith_connector._verify_webhook_signature(payload, "invalid-signature")
            
            assert result is False


class TestAPIEndpoints:
    """Test LangSmith API endpoints."""

    def test_get_status_endpoint(self, test_client):
        """Test the status endpoint."""
        with patch('api.langsmith.langsmith_connector') as mock_connector:
            mock_connector.get_connection_status.return_value = {
                "connected": True,
                "project": "test"
            }
            mock_connector.get_sync_stats.return_value = {
                "total_langsmith_traces": 100
            }
            
            # Mock authentication
            with patch('auth.dependencies.get_current_user') as mock_auth:
                mock_user = Mock()
                mock_auth.return_value = mock_user
                
                response = test_client.get("/api/v1/langsmith/status")
                
                assert response.status_code == 200
                data = response.json()
                assert "connection" in data
                assert "sync_stats" in data

    def test_sync_endpoint(self, test_client):
        """Test the sync endpoint."""
        with patch('api.langsmith.langsmith_connector') as mock_connector:
            mock_sync_result = LangSmithSync(
                project_name="test",
                total_synced=5,
                errors=[]
            )
            mock_connector.sync_traces_from_langsmith.return_value = mock_sync_result
            
            with patch('auth.dependencies.require_permission') as mock_auth:
                mock_user = Mock()
                mock_user.team_id = uuid4()
                mock_auth.return_value = mock_user
                
                response = test_client.post(
                    "/api/v1/langsmith/sync",
                    json={"project_name": "test", "limit": 10}
                )
                
                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "success"
                assert data["total_synced"] == 5

    def test_push_evaluation_endpoint(self, test_client):
        """Test the push evaluation endpoint."""
        with patch('api.langsmith.langsmith_connector') as mock_connector:
            mock_connector.push_evaluation_to_langsmith.return_value = True
            
            with patch('auth.dependencies.require_permission') as mock_auth:
                mock_user = Mock()
                mock_user.id = uuid4()
                mock_auth.return_value = mock_user
                
                response = test_client.post(
                    "/api/v1/langsmith/push-evaluation",
                    json={
                        "trace_id": str(uuid4()),
                        "score": 0.8,
                        "label": "good"
                    }
                )
                
                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "success"

    def test_webhook_endpoint(self, test_client):
        """Test the webhook endpoint."""
        with patch('api.langsmith.langsmith_connector') as mock_connector:
            mock_connector.handle_webhook.return_value = {
                "status": "success",
                "trace_id": str(uuid4())
            }
            
            response = test_client.post(
                "/api/v1/langsmith/webhook",
                json={
                    "event_type": "run.created",
                    "data": {"id": str(uuid4())}
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"


class TestErrorHandling:
    """Test error handling scenarios."""

    @pytest.mark.asyncio
    async def test_sync_with_api_error(self, langsmith_connector):
        """Test sync with API error."""
        langsmith_connector.client.list_runs.side_effect = Exception("API Error")
        
        with pytest.raises(Exception, match="API Error"):
            await langsmith_connector.sync_traces_from_langsmith()

    @pytest.mark.asyncio
    async def test_convert_run_with_malformed_data(self, langsmith_connector):
        """Test converting run with malformed data."""
        malformed_run = Mock()
        malformed_run.id = "invalid-id"
        malformed_run.inputs = None
        malformed_run.outputs = None
        malformed_run.extra = None
        malformed_run.start_time = None
        malformed_run.end_time = None
        malformed_run.total_time = None
        malformed_run.total_cost = None
        malformed_run.session_id = None
        malformed_run.error = None
        
        with patch('services.langsmith_connector.AsyncSessionLocal') as mock_session_local:
            mock_session = AsyncMock()
            
            result = await langsmith_connector._convert_langsmith_run_to_trace(
                malformed_run, None, mock_session
            )
            
            # Should handle gracefully and return valid trace data
            assert result is not None
            assert result["user_input"] == ""
            assert result["model_output"] == ""
            assert result["model_name"] == "unknown"

    @pytest.mark.asyncio
    async def test_webhook_with_invalid_signature(self, langsmith_connector):
        """Test webhook with invalid signature."""
        langsmith_connector.webhook_secret = "secret"
        
        with patch.object(langsmith_connector, '_verify_webhook_signature') as mock_verify:
            mock_verify.return_value = False
            
            from fastapi import HTTPException
            with pytest.raises(HTTPException) as exc_info:
                await langsmith_connector.handle_webhook({}, "invalid-sig")
            
            assert exc_info.value.status_code == 401


class TestPerformance:
    """Test performance characteristics."""

    @pytest.mark.asyncio
    async def test_large_sync_performance(self, langsmith_connector):
        """Test performance with large number of runs."""
        # Create 1000 mock runs
        mock_runs = []
        for i in range(1000):
            run = Mock()
            run.id = f"run-{i}"
            run.inputs = {"input": f"test input {i}"}
            run.outputs = {"output": f"test output {i}"}
            run.extra = {"model_name": "gpt-4"}
            run.start_time = datetime.utcnow()
            run.end_time = datetime.utcnow()
            run.total_time = 1.0
            run.total_cost = 0.001
            run.session_id = str(uuid4())
            run.run_type = "llm"
            run.tags = []
            run.error = None
            mock_runs.append(run)
        
        langsmith_connector.client.list_runs.return_value = mock_runs
        
        with patch('services.langsmith_connector.AsyncSessionLocal') as mock_session_local:
            mock_session = AsyncMock()
            mock_session_local.return_value.__aenter__.return_value = mock_session
            mock_session.execute.return_value.scalar_one_or_none.return_value = None
            
            with patch('services.langsmith_connector.cache_service') as mock_cache:
                mock_cache.get.return_value = None
                mock_cache.set.return_value = None
                
                import time
                start_time = time.time()
                
                result = await langsmith_connector.sync_traces_from_langsmith(
                    limit=1000
                )
                
                end_time = time.time()
                duration = end_time - start_time
                
                # Should process 1000 runs in reasonable time (< 10 seconds)
                assert duration < 10.0
                assert result.total_synced == 1000

    @pytest.mark.asyncio
    async def test_concurrent_webhook_handling(self, langsmith_connector):
        """Test handling multiple webhooks concurrently."""
        payloads = [
            {
                "event_type": "run.created",
                "data": {"id": f"run-{i}"}
            }
            for i in range(10)
        ]
        
        langsmith_connector.client.read_run.return_value = Mock()
        
        with patch('services.langsmith_connector.AsyncSessionLocal') as mock_session_local:
            mock_session = AsyncMock()
            mock_session_local.return_value.__aenter__.return_value = mock_session
            
            with patch.object(langsmith_connector, '_convert_langsmith_run_to_trace') as mock_convert:
                mock_convert.return_value = {"user_input": "test"}
                
                # Process webhooks concurrently
                tasks = [
                    langsmith_connector.handle_webhook(payload, "")
                    for payload in payloads
                ]
                
                results = await asyncio.gather(*tasks)
                
                # All should succeed
                assert len(results) == 10
                for result in results:
                    assert result["status"] == "success"


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 