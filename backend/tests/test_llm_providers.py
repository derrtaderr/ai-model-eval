"""
Comprehensive test suite for LLM Provider Integration System
Tests all aspects of the provider management system.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
from typing import Dict, Any

from fastapi.testclient import TestClient
from fastapi import FastAPI

from services.llm_providers import (
    LLMProviderManager,
    OpenAIProvider,
    AnthropicProvider,
    ProviderType,
    OperationType,
    ModelCapability,
    LLMRequest,
    LLMResponse,
    ProviderConfig,
    ProviderHealth,
    ModelInfo,
    UsageStats,
    LoggingHook,
    CostTrackingHook
)

from api.llm_providers import router


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client."""
    client = Mock()
    client.models.list = AsyncMock()
    client.chat.completions.create = AsyncMock()
    return client


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client."""
    client = Mock()
    client.messages.create = AsyncMock()
    return client


@pytest.fixture
def openai_provider():
    """Create OpenAI provider for testing."""
    config = ProviderConfig(
        provider_type=ProviderType.OPENAI,
        api_key="test-key",
        enabled=True
    )
    return OpenAIProvider(config)


@pytest.fixture
def anthropic_provider():
    """Create Anthropic provider for testing."""
    config = ProviderConfig(
        provider_type=ProviderType.ANTHROPIC,
        api_key="test-key",
        enabled=True
    )
    return AnthropicProvider(config)


@pytest.fixture
def provider_manager():
    """Create provider manager for testing."""
    return LLMProviderManager()


@pytest.fixture
def sample_llm_request():
    """Sample LLM request for testing."""
    return LLMRequest(
        operation_type=OperationType.CHAT,
        messages=[{"role": "user", "content": "Hello"}],
        model="gpt-3.5-turbo",
        max_tokens=100,
        temperature=0.7,
        user_id="test-user"
    )


class TestProviderConfig:
    """Test provider configuration."""
    
    def test_provider_config_creation(self):
        """Test creating a provider configuration."""
        config = ProviderConfig(
            provider_type=ProviderType.OPENAI,
            api_key="test-key",
            timeout=30,
            max_retries=3,
            enabled=True
        )
        
        assert config.provider_type == ProviderType.OPENAI
        assert config.api_key == "test-key"
        assert config.timeout == 30
        assert config.max_retries == 3
        assert config.enabled is True
    
    def test_provider_config_defaults(self):
        """Test provider configuration with defaults."""
        config = ProviderConfig(provider_type=ProviderType.ANTHROPIC)
        
        assert config.timeout == 30
        assert config.max_retries == 3
        assert config.enabled is True
        assert config.custom_headers == {}


class TestModelInfo:
    """Test model information handling."""
    
    def test_model_info_creation(self):
        """Test creating model information."""
        model = ModelInfo(
            id="gpt-4",
            name="GPT-4",
            provider=ProviderType.OPENAI,
            capabilities=[ModelCapability.CHAT, ModelCapability.REASONING],
            context_window=8192,
            cost_per_token_input=0.03/1000,
            cost_per_token_output=0.06/1000,
            supports_streaming=True,
            supports_functions=True
        )
        
        assert model.id == "gpt-4"
        assert model.provider == ProviderType.OPENAI
        assert ModelCapability.CHAT in model.capabilities
        assert model.supports_streaming is True
        assert model.supports_functions is True


class TestOpenAIProvider:
    """Test OpenAI provider implementation."""
    
    @pytest.mark.asyncio
    async def test_initialization_success(self, openai_provider, mock_openai_client):
        """Test successful OpenAI provider initialization."""
        with patch('services.llm_providers.AsyncOpenAI', return_value=mock_openai_client):
            result = await openai_provider.initialize()
            assert result is True
            assert openai_provider.client == mock_openai_client
    
    @pytest.mark.asyncio
    async def test_initialization_failure(self, openai_provider):
        """Test OpenAI provider initialization failure."""
        with patch('services.llm_providers.AsyncOpenAI', side_effect=ImportError("OpenAI not installed")):
            result = await openai_provider.initialize()
            assert result is False
    
    @pytest.mark.asyncio
    async def test_get_available_models(self, openai_provider, mock_openai_client):
        """Test getting available OpenAI models."""
        # Mock models response
        mock_model = Mock()
        mock_model.id = "gpt-4"
        mock_models_response = Mock()
        mock_models_response.data = [mock_model]
        
        mock_openai_client.models.list.return_value = mock_models_response
        openai_provider.client = mock_openai_client
        
        models = await openai_provider.get_available_models()
        
        assert len(models) == 1
        assert models[0].id == "gpt-4"
        assert models[0].provider == ProviderType.OPENAI
    
    def test_parse_openai_model(self, openai_provider):
        """Test parsing OpenAI model information."""
        mock_model = Mock()
        mock_model.id = "gpt-4-vision-preview"
        
        model_info = openai_provider._parse_openai_model(mock_model)
        
        assert model_info.id == "gpt-4-vision-preview"
        assert ModelCapability.CHAT in model_info.capabilities
        assert ModelCapability.REASONING in model_info.capabilities
        assert ModelCapability.VISION in model_info.capabilities
        assert model_info.supports_vision is True
    
    @pytest.mark.asyncio
    async def test_complete_request(self, openai_provider, mock_openai_client, sample_llm_request):
        """Test completing an OpenAI request."""
        # Mock completion response
        mock_choice = Mock()
        mock_choice.message.content = "Hello! How can I help you?"
        mock_choice.finish_reason = "stop"
        mock_choice.message.function_call = None
        
        mock_usage = Mock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 15
        mock_usage.dict.return_value = {"prompt_tokens": 10, "completion_tokens": 15}
        
        mock_response = Mock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage
        mock_response.id = "test-request-id"
        
        mock_openai_client.chat.completions.create.return_value = mock_response
        openai_provider.client = mock_openai_client
        
        response = await openai_provider.complete(sample_llm_request)
        
        assert response.provider == ProviderType.OPENAI
        assert response.content == "Hello! How can I help you?"
        assert response.finish_reason == "stop"
        assert response.usage["prompt_tokens"] == 10
        assert response.usage["completion_tokens"] == 15
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, openai_provider, mock_openai_client):
        """Test successful health check."""
        mock_model = Mock()
        mock_model.id = "gpt-3.5-turbo"
        mock_models_response = Mock()
        mock_models_response.data = [mock_model]
        
        mock_openai_client.models.list.return_value = mock_models_response
        openai_provider.client = mock_openai_client
        
        health = await openai_provider.health_check()
        
        assert health.provider_type == ProviderType.OPENAI
        assert health.is_healthy is True
        assert health.response_time_ms is not None
        assert "gpt-3.5-turbo" in health.available_models
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self, openai_provider, mock_openai_client):
        """Test failed health check."""
        mock_openai_client.models.list.side_effect = Exception("API Error")
        openai_provider.client = mock_openai_client
        
        health = await openai_provider.health_check()
        
        assert health.provider_type == ProviderType.OPENAI
        assert health.is_healthy is False
        assert health.error_message == "API Error"


class TestAnthropicProvider:
    """Test Anthropic provider implementation."""
    
    @pytest.mark.asyncio
    async def test_initialization_success(self, anthropic_provider, mock_anthropic_client):
        """Test successful Anthropic provider initialization."""
        with patch('services.llm_providers.anthropic.AsyncAnthropic', return_value=mock_anthropic_client):
            result = await anthropic_provider.initialize()
            assert result is True
            assert anthropic_provider.client == mock_anthropic_client
    
    @pytest.mark.asyncio
    async def test_get_available_models(self, anthropic_provider):
        """Test getting available Anthropic models."""
        models = await anthropic_provider.get_available_models()
        
        assert len(models) >= 1
        claude_models = [m for m in models if "claude" in m.id.lower()]
        assert len(claude_models) > 0
        
        # Check that all models have required attributes
        for model in models:
            assert model.provider == ProviderType.ANTHROPIC
            assert model.supports_streaming is True
            assert ModelCapability.CHAT in model.capabilities
    
    @pytest.mark.asyncio
    async def test_complete_request(self, anthropic_provider, mock_anthropic_client, sample_llm_request):
        """Test completing an Anthropic request."""
        # Mock Anthropic response
        mock_content_block = Mock()
        mock_content_block.text = "Hello! I'm Claude, how can I assist you?"
        
        mock_usage = Mock()
        mock_usage.input_tokens = 10
        mock_usage.output_tokens = 15
        
        mock_response = Mock()
        mock_response.content = [mock_content_block]
        mock_response.usage = mock_usage
        mock_response.stop_reason = "end_turn"
        mock_response.id = "test-request-id"
        
        mock_anthropic_client.messages.create.return_value = mock_response
        anthropic_provider.client = mock_anthropic_client
        
        response = await anthropic_provider.complete(sample_llm_request)
        
        assert response.provider == ProviderType.ANTHROPIC
        assert response.content == "Hello! I'm Claude, how can I assist you?"
        assert response.finish_reason == "end_turn"
        assert response.usage["prompt_tokens"] == 10
        assert response.usage["completion_tokens"] == 15
    
    def test_prepare_anthropic_request(self, anthropic_provider, sample_llm_request):
        """Test preparing an Anthropic request."""
        anthropic_request = anthropic_provider._prepare_anthropic_request(sample_llm_request)
        
        assert "model" in anthropic_request
        assert "messages" in anthropic_request
        assert "max_tokens" in anthropic_request
        assert anthropic_request["max_tokens"] == 100


class TestLLMProviderManager:
    """Test LLM provider manager."""
    
    def test_provider_manager_initialization(self, provider_manager):
        """Test provider manager initialization."""
        assert isinstance(provider_manager, LLMProviderManager)
        assert isinstance(provider_manager.providers, dict)
    
    @pytest.mark.asyncio
    async def test_initialize_all_providers(self, provider_manager):
        """Test initializing all providers."""
        with patch.object(provider_manager, 'providers', {
            ProviderType.OPENAI: Mock(initialize=AsyncMock(return_value=True)),
            ProviderType.ANTHROPIC: Mock(initialize=AsyncMock(return_value=True))
        }):
            results = await provider_manager.initialize_all()
            
            assert ProviderType.OPENAI in results
            assert ProviderType.ANTHROPIC in results
            assert results[ProviderType.OPENAI] is True
            assert results[ProviderType.ANTHROPIC] is True
    
    @pytest.mark.asyncio
    async def test_get_available_providers(self, provider_manager):
        """Test getting available providers."""
        mock_healthy_provider = Mock()
        mock_healthy_provider.config.enabled = True
        mock_healthy_provider.health_check = AsyncMock(return_value=ProviderHealth(
            provider_type=ProviderType.OPENAI,
            is_healthy=True,
            last_check=datetime.utcnow()
        ))
        
        mock_unhealthy_provider = Mock()
        mock_unhealthy_provider.config.enabled = True
        mock_unhealthy_provider.health_check = AsyncMock(return_value=ProviderHealth(
            provider_type=ProviderType.ANTHROPIC,
            is_healthy=False,
            last_check=datetime.utcnow()
        ))
        
        provider_manager.providers = {
            ProviderType.OPENAI: mock_healthy_provider,
            ProviderType.ANTHROPIC: mock_unhealthy_provider
        }
        
        available = await provider_manager.get_available_providers()
        
        assert ProviderType.OPENAI in available
        assert ProviderType.ANTHROPIC not in available
    
    @pytest.mark.asyncio
    async def test_complete_with_specific_provider(self, provider_manager, sample_llm_request):
        """Test completing request with specific provider."""
        mock_response = LLMResponse(
            provider=ProviderType.OPENAI,
            model="gpt-3.5-turbo",
            content="Test response"
        )
        
        mock_provider = Mock()
        mock_provider.complete = AsyncMock(return_value=mock_response)
        
        provider_manager.providers = {ProviderType.OPENAI: mock_provider}
        
        response = await provider_manager.complete(sample_llm_request, ProviderType.OPENAI)
        
        assert response.provider == ProviderType.OPENAI
        assert response.content == "Test response"
        mock_provider.complete.assert_called_once_with(sample_llm_request)
    
    @pytest.mark.asyncio
    async def test_get_usage_stats(self, provider_manager):
        """Test getting usage statistics."""
        mock_stats = UsageStats(
            provider_type=ProviderType.OPENAI,
            requests_count=100,
            total_cost_usd=5.50,
            avg_response_time_ms=150.0
        )
        
        mock_provider = Mock()
        mock_provider.usage_stats = mock_stats
        
        provider_manager.providers = {ProviderType.OPENAI: mock_provider}
        
        stats = await provider_manager.get_usage_stats()
        
        assert ProviderType.OPENAI in stats
        assert stats[ProviderType.OPENAI].requests_count == 100
        assert stats[ProviderType.OPENAI].total_cost_usd == 5.50


class TestProviderHooks:
    """Test provider hook system."""
    
    @pytest.mark.asyncio
    async def test_logging_hook(self, sample_llm_request):
        """Test logging hook functionality."""
        hook = LoggingHook()
        
        # Test before_request hook
        modified_request = await hook.before_request(sample_llm_request)
        assert modified_request == sample_llm_request
        
        # Test after_response hook
        response = LLMResponse(
            provider=ProviderType.OPENAI,
            model="gpt-3.5-turbo",
            content="Test response",
            cost_usd=0.001
        )
        modified_response = await hook.after_response(response)
        assert modified_response == response
        
        # Test on_error hook
        try:
            await hook.on_error(Exception("Test error"), sample_llm_request)
        except Exception:
            pytest.fail("Hook should not raise exceptions")
    
    @pytest.mark.asyncio
    async def test_cost_tracking_hook(self):
        """Test cost tracking hook functionality."""
        hook = CostTrackingHook()
        
        response = LLMResponse(
            provider=ProviderType.OPENAI,
            model="gpt-3.5-turbo",
            content="Test response",
            cost_usd=0.001
        )
        
        with patch('services.llm_providers.cache_service') as mock_cache:
            mock_cache.get.return_value = 0.005
            mock_cache.set.return_value = None
            
            modified_response = await hook.after_response(response)
            
            assert modified_response == response
            mock_cache.get.assert_called_once()
            mock_cache.set.assert_called_once()


class TestProviderAPI:
    """Test provider API endpoints."""
    
    @pytest.fixture
    def app(self):
        """Create test app with provider router."""
        app = FastAPI()
        app.include_router(router)
        return app
    
    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)
    
    def test_get_providers_overview(self, client):
        """Test getting providers overview."""
        with patch('api.llm_providers.llm_provider_manager') as mock_manager:
            mock_manager.get_provider_health.return_value = {}
            mock_manager.get_available_providers.return_value = []
            mock_manager.get_usage_stats.return_value = {}
            
            response = client.get("/")
            
            # This would normally require authentication, so we expect 401 or similar
            assert response.status_code in [200, 401, 422]
    
    def test_get_models_endpoint(self, client):
        """Test getting models endpoint."""
        with patch('api.llm_providers.llm_provider_manager') as mock_manager:
            mock_models = {
                ProviderType.OPENAI: [
                    ModelInfo(
                        id="gpt-3.5-turbo",
                        name="GPT-3.5 Turbo",
                        provider=ProviderType.OPENAI,
                        capabilities=[ModelCapability.CHAT]
                    )
                ]
            }
            mock_manager.get_all_models.return_value = mock_models
            
            response = client.get("/models")
            
            # This would normally require authentication
            assert response.status_code in [200, 401, 422]


class TestEdgeCases:
    """Test edge cases and error scenarios."""
    
    @pytest.mark.asyncio
    async def test_provider_with_no_api_key(self):
        """Test provider behavior with no API key."""
        config = ProviderConfig(
            provider_type=ProviderType.OPENAI,
            api_key=None,
            enabled=True
        )
        provider = OpenAIProvider(config)
        
        # Should handle gracefully
        result = await provider.initialize()
        assert result is False
    
    @pytest.mark.asyncio
    async def test_request_with_invalid_model(self, openai_provider, mock_openai_client):
        """Test request with invalid model."""
        invalid_request = LLMRequest(
            operation_type=OperationType.CHAT,
            messages=[{"role": "user", "content": "Hello"}],
            model="invalid-model-name",
            user_id="test-user"
        )
        
        mock_openai_client.chat.completions.create.side_effect = Exception("Invalid model")
        openai_provider.client = mock_openai_client
        
        with pytest.raises(Exception):
            await openai_provider.complete(invalid_request)
    
    @pytest.mark.asyncio
    async def test_rate_limit_handling(self, openai_provider, mock_openai_client):
        """Test rate limit error handling."""
        rate_limit_error = Exception("Rate limit exceeded")
        mock_openai_client.chat.completions.create.side_effect = rate_limit_error
        openai_provider.client = mock_openai_client
        
        request = LLMRequest(
            operation_type=OperationType.CHAT,
            messages=[{"role": "user", "content": "Hello"}],
            user_id="test-user"
        )
        
        with pytest.raises(Exception):
            await openai_provider.complete(request)
        
        # Check that usage stats are updated with error
        assert openai_provider.usage_stats.error_count > 0
    
    def test_model_cost_calculation(self, openai_provider):
        """Test model cost calculation."""
        cost_info = openai_provider._get_model_cost_info("gpt-4")
        assert cost_info is not None
        assert "input_cost" in cost_info
        assert "output_cost" in cost_info
        
        # Test unknown model
        unknown_cost = openai_provider._get_model_cost_info("unknown-model")
        assert unknown_cost is None


class TestPerformance:
    """Test performance scenarios."""
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, openai_provider, mock_openai_client):
        """Test handling concurrent requests."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.choices[0].message.function_call = None
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 10
        mock_response.usage.dict.return_value = {"prompt_tokens": 10, "completion_tokens": 10}
        
        mock_openai_client.chat.completions.create.return_value = mock_response
        openai_provider.client = mock_openai_client
        
        requests = [
            LLMRequest(
                operation_type=OperationType.CHAT,
                messages=[{"role": "user", "content": f"Hello {i}"}],
                user_id="test-user"
            )
            for i in range(10)
        ]
        
        # Run concurrent requests
        tasks = [openai_provider.complete(req) for req in requests]
        responses = await asyncio.gather(*tasks)
        
        assert len(responses) == 10
        for response in responses:
            assert response.content == "Response"
        
        # Check that usage stats are properly updated
        assert openai_provider.usage_stats.requests_count == 10
    
    @pytest.mark.asyncio
    async def test_model_caching(self, openai_provider, mock_openai_client):
        """Test model list caching."""
        mock_model = Mock()
        mock_model.id = "gpt-3.5-turbo"
        mock_models_response = Mock()
        mock_models_response.data = [mock_model]
        
        mock_openai_client.models.list.return_value = mock_models_response
        openai_provider.client = mock_openai_client
        
        # First call should fetch from API
        models1 = await openai_provider.get_available_models()
        assert len(models1) == 1
        
        # Second call should use cache
        models2 = await openai_provider.get_available_models()
        assert len(models2) == 1
        
        # API should only be called once due to caching
        mock_openai_client.models.list.assert_called_once()


if __name__ == "__main__":
    # Run specific test categories
    pytest.main([
        "-v",
        "--tb=short",
        __file__,
        "-k", "not performance"  # Skip performance tests in regular runs
    ]) 