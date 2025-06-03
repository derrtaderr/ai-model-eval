"""
Enhanced LLM Provider Integration System
Comprehensive abstraction layer for multiple LLM providers with hooks, health monitoring, and advanced features.

This service provides:
- Provider abstraction with standardized interfaces
- Health monitoring and capability detection
- Usage tracking and cost management
- Streaming and batch operations
- Model discovery and selection
- Provider-specific optimizations
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Callable, AsyncIterator
from uuid import UUID, uuid4

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func
from decouple import config
import httpx

from config.settings import get_settings
from utils.cache_service import cache_service


logger = logging.getLogger(__name__)


class ProviderType(str, Enum):
    """Supported LLM provider types."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"
    AZURE_OPENAI = "azure_openai"
    HUGGINGFACE = "huggingface"
    GOOGLE = "google"
    COHERE = "cohere"


class OperationType(str, Enum):
    """Types of LLM operations."""
    COMPLETION = "completion"
    CHAT = "chat"
    EMBEDDING = "embedding"
    EVALUATION = "evaluation"
    SUMMARIZATION = "summarization"
    CLASSIFICATION = "classification"
    GENERATION = "generation"


class ModelCapability(str, Enum):
    """Model capabilities."""
    TEXT_GENERATION = "text_generation"
    CHAT = "chat"
    FUNCTION_CALLING = "function_calling"
    EMBEDDINGS = "embeddings"
    VISION = "vision"
    CODE_GENERATION = "code_generation"
    REASONING = "reasoning"
    MULTIMODAL = "multimodal"


@dataclass
class ModelInfo:
    """Information about a specific model."""
    id: str
    name: str
    provider: ProviderType
    capabilities: List[ModelCapability]
    max_tokens: Optional[int] = None
    cost_per_token_input: Optional[float] = None
    cost_per_token_output: Optional[float] = None
    context_window: Optional[int] = None
    training_cutoff: Optional[str] = None
    description: Optional[str] = None
    supports_streaming: bool = False
    supports_functions: bool = False
    supports_vision: bool = False


@dataclass
class ProviderConfig:
    """Configuration for a provider."""
    provider_type: ProviderType
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    organization: Optional[str] = None
    project: Optional[str] = None
    api_version: Optional[str] = None
    timeout: int = 30
    max_retries: int = 3
    rate_limit_rpm: Optional[int] = None
    rate_limit_tpm: Optional[int] = None
    custom_headers: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True


@dataclass
class ProviderHealth:
    """Health status of a provider."""
    provider_type: ProviderType
    is_healthy: bool
    last_check: datetime
    response_time_ms: Optional[float] = None
    error_message: Optional[str] = None
    available_models: List[str] = field(default_factory=list)
    rate_limit_status: Optional[Dict[str, Any]] = None


@dataclass
class LLMRequest:
    """Standardized LLM request."""
    operation_type: OperationType
    messages: Optional[List[Dict[str, Any]]] = None
    prompt: Optional[str] = None
    model: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop: Optional[List[str]] = None
    stream: bool = False
    functions: Optional[List[Dict[str, Any]]] = None
    function_call: Optional[Union[str, Dict[str, Any]]] = None
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    """Standardized LLM response."""
    provider: ProviderType
    model: str
    content: str
    usage: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    response_time_ms: Optional[float] = None
    cost_usd: Optional[float] = None
    request_id: Optional[str] = None


@dataclass
class UsageStats:
    """Usage statistics for a provider."""
    provider_type: ProviderType
    requests_count: int = 0
    tokens_input: int = 0
    tokens_output: int = 0
    total_cost_usd: float = 0.0
    avg_response_time_ms: float = 0.0
    error_count: int = 0
    last_request: Optional[datetime] = None


class ProviderHook:
    """Base class for provider hooks."""
    
    def __init__(self, name: str):
        self.name = name
    
    async def before_request(self, request: LLMRequest) -> LLMRequest:
        """Hook called before making a request."""
        return request
    
    async def after_response(self, response: LLMResponse) -> LLMResponse:
        """Hook called after receiving a response."""
        return response
    
    async def on_error(self, error: Exception, request: LLMRequest) -> None:
        """Hook called when an error occurs."""
        pass


class BaseLLMProvider(ABC):
    """Base class for all LLM providers."""
    
    def __init__(self, config: ProviderConfig):
        self.config = config
        self.provider_type = config.provider_type
        self.client = None
        self._health_status = None
        self._last_health_check = None
        self.hooks: List[ProviderHook] = []
        self.usage_stats = UsageStats(provider_type=self.provider_type)
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the provider."""
        pass
    
    @abstractmethod
    async def get_available_models(self) -> List[ModelInfo]:
        """Get list of available models."""
        pass
    
    @abstractmethod
    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Complete a text generation request."""
        pass
    
    @abstractmethod
    async def stream_complete(self, request: LLMRequest) -> AsyncIterator[str]:
        """Stream a text generation request."""
        pass
    
    @abstractmethod
    async def health_check(self) -> ProviderHealth:
        """Check provider health."""
        pass
    
    def add_hook(self, hook: ProviderHook):
        """Add a hook to this provider."""
        self.hooks.append(hook)
    
    def remove_hook(self, hook_name: str):
        """Remove a hook by name."""
        self.hooks = [h for h in self.hooks if h.name != hook_name]
    
    async def _apply_before_hooks(self, request: LLMRequest) -> LLMRequest:
        """Apply all before-request hooks."""
        for hook in self.hooks:
            try:
                request = await hook.before_request(request)
            except Exception as e:
                logger.error(f"Error in before_request hook {hook.name}: {e}")
        return request
    
    async def _apply_after_hooks(self, response: LLMResponse) -> LLMResponse:
        """Apply all after-response hooks."""
        for hook in self.hooks:
            try:
                response = await hook.after_response(response)
            except Exception as e:
                logger.error(f"Error in after_response hook {hook.name}: {e}")
        return response
    
    async def _apply_error_hooks(self, error: Exception, request: LLMRequest):
        """Apply all error hooks."""
        for hook in self.hooks:
            try:
                await hook.on_error(error, request)
            except Exception as e:
                logger.error(f"Error in on_error hook {hook.name}: {e}")
    
    def _update_usage_stats(self, response: LLMResponse):
        """Update usage statistics."""
        self.usage_stats.requests_count += 1
        self.usage_stats.last_request = datetime.utcnow()
        
        if response.usage:
            self.usage_stats.tokens_input += response.usage.get('prompt_tokens', 0)
            self.usage_stats.tokens_output += response.usage.get('completion_tokens', 0)
        
        if response.cost_usd:
            self.usage_stats.total_cost_usd += response.cost_usd
        
        if response.response_time_ms:
            # Update average response time
            total_requests = self.usage_stats.requests_count
            current_avg = self.usage_stats.avg_response_time_ms
            new_avg = ((current_avg * (total_requests - 1)) + response.response_time_ms) / total_requests
            self.usage_stats.avg_response_time_ms = new_avg


class OpenAIProvider(BaseLLMProvider):
    """OpenAI provider implementation."""
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.models_cache = None
        self.models_cache_time = None
    
    async def initialize(self) -> bool:
        """Initialize OpenAI client."""
        try:
            # Import OpenAI
            try:
                from openai import AsyncOpenAI
                self.client = AsyncOpenAI(
                    api_key=self.config.api_key,
                    organization=self.config.organization,
                    timeout=self.config.timeout,
                    max_retries=self.config.max_retries
                )
                return True
            except ImportError:
                logger.error("OpenAI library not installed")
                return False
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI provider: {e}")
            return False
    
    async def get_available_models(self) -> List[ModelInfo]:
        """Get available OpenAI models."""
        # Cache models for 1 hour
        if (self.models_cache and self.models_cache_time and 
            datetime.utcnow() - self.models_cache_time < timedelta(hours=1)):
            return self.models_cache
        
        try:
            models_response = await self.client.models.list()
            models = []
            
            for model in models_response.data:
                model_info = self._parse_openai_model(model)
                if model_info:
                    models.append(model_info)
            
            self.models_cache = models
            self.models_cache_time = datetime.utcnow()
            return models
        
        except Exception as e:
            logger.error(f"Failed to get OpenAI models: {e}")
            return []
    
    def _parse_openai_model(self, model) -> Optional[ModelInfo]:
        """Parse OpenAI model into ModelInfo."""
        model_id = model.id
        
        # Define capabilities based on model name
        capabilities = [ModelCapability.TEXT_GENERATION]
        supports_functions = False
        supports_vision = False
        
        if any(x in model_id for x in ['gpt-4', 'gpt-3.5']):
            capabilities.append(ModelCapability.CHAT)
            if 'gpt-4' in model_id:
                capabilities.extend([ModelCapability.REASONING, ModelCapability.CODE_GENERATION])
                supports_functions = True
            if 'vision' in model_id:
                capabilities.append(ModelCapability.VISION)
                supports_vision = True
        
        # Determine context window and costs
        context_window = None
        cost_input = None
        cost_output = None
        
        if 'gpt-4' in model_id:
            context_window = 8192 if '32k' not in model_id else 32768
            cost_input = 0.03 / 1000  # $0.03 per 1K tokens
            cost_output = 0.06 / 1000  # $0.06 per 1K tokens
        elif 'gpt-3.5' in model_id:
            context_window = 4096 if '16k' not in model_id else 16384
            cost_input = 0.0015 / 1000  # $0.0015 per 1K tokens
            cost_output = 0.002 / 1000  # $0.002 per 1K tokens
        
        return ModelInfo(
            id=model_id,
            name=model_id,
            provider=ProviderType.OPENAI,
            capabilities=capabilities,
            context_window=context_window,
            cost_per_token_input=cost_input,
            cost_per_token_output=cost_output,
            supports_streaming=True,
            supports_functions=supports_functions,
            supports_vision=supports_vision
        )
    
    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Complete a request using OpenAI."""
        start_time = time.time()
        
        try:
            # Apply before hooks
            request = await self._apply_before_hooks(request)
            
            # Prepare OpenAI request
            openai_request = self._prepare_openai_request(request)
            
            # Make the request
            if request.stream:
                raise NotImplementedError("Use stream_complete for streaming requests")
            
            response = await self.client.chat.completions.create(**openai_request)
            
            # Parse response
            llm_response = self._parse_openai_response(response, request.model, start_time)
            
            # Apply after hooks
            llm_response = await self._apply_after_hooks(llm_response)
            
            # Update usage stats
            self._update_usage_stats(llm_response)
            
            return llm_response
        
        except Exception as e:
            await self._apply_error_hooks(e, request)
            self.usage_stats.error_count += 1
            raise
    
    async def stream_complete(self, request: LLMRequest) -> AsyncIterator[str]:
        """Stream a completion request."""
        try:
            request = await self._apply_before_hooks(request)
            openai_request = self._prepare_openai_request(request)
            openai_request['stream'] = True
            
            stream = await self.client.chat.completions.create(**openai_request)
            
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        
        except Exception as e:
            await self._apply_error_hooks(e, request)
            self.usage_stats.error_count += 1
            raise
    
    def _prepare_openai_request(self, request: LLMRequest) -> Dict[str, Any]:
        """Prepare request for OpenAI API."""
        openai_request = {
            'model': request.model or 'gpt-3.5-turbo',
            'messages': request.messages or [{'role': 'user', 'content': request.prompt}],
        }
        
        if request.max_tokens:
            openai_request['max_tokens'] = request.max_tokens
        if request.temperature is not None:
            openai_request['temperature'] = request.temperature
        if request.top_p is not None:
            openai_request['top_p'] = request.top_p
        if request.frequency_penalty is not None:
            openai_request['frequency_penalty'] = request.frequency_penalty
        if request.presence_penalty is not None:
            openai_request['presence_penalty'] = request.presence_penalty
        if request.stop:
            openai_request['stop'] = request.stop
        if request.functions:
            openai_request['functions'] = request.functions
        if request.function_call:
            openai_request['function_call'] = request.function_call
        if request.user_id:
            openai_request['user'] = request.user_id
        
        return openai_request
    
    def _parse_openai_response(self, response, model: str, start_time: float) -> LLMResponse:
        """Parse OpenAI response."""
        choice = response.choices[0]
        content = choice.message.content or ""
        
        # Calculate cost
        usage = response.usage
        cost_usd = None
        if usage:
            model_info = self._get_model_cost_info(model)
            if model_info:
                cost_usd = (
                    usage.prompt_tokens * model_info['input_cost'] +
                    usage.completion_tokens * model_info['output_cost']
                )
        
        return LLMResponse(
            provider=ProviderType.OPENAI,
            model=model,
            content=content,
            usage=usage.dict() if usage else None,
            finish_reason=choice.finish_reason,
            function_call=choice.message.function_call.dict() if choice.message.function_call else None,
            response_time_ms=(time.time() - start_time) * 1000,
            cost_usd=cost_usd,
            request_id=getattr(response, 'id', None)
        )
    
    def _get_model_cost_info(self, model: str) -> Optional[Dict[str, float]]:
        """Get cost information for a model."""
        cost_map = {
            'gpt-4': {'input_cost': 0.03/1000, 'output_cost': 0.06/1000},
            'gpt-4-32k': {'input_cost': 0.06/1000, 'output_cost': 0.12/1000},
            'gpt-3.5-turbo': {'input_cost': 0.0015/1000, 'output_cost': 0.002/1000},
            'gpt-3.5-turbo-16k': {'input_cost': 0.003/1000, 'output_cost': 0.004/1000},
        }
        
        for model_key, costs in cost_map.items():
            if model_key in model:
                return costs
        
        return None
    
    async def health_check(self) -> ProviderHealth:
        """Check OpenAI provider health."""
        start_time = time.time()
        
        try:
            # Simple health check: list models
            models_response = await self.client.models.list()
            available_models = [model.id for model in models_response.data[:5]]  # First 5 models
            
            response_time = (time.time() - start_time) * 1000
            
            return ProviderHealth(
                provider_type=ProviderType.OPENAI,
                is_healthy=True,
                last_check=datetime.utcnow(),
                response_time_ms=response_time,
                available_models=available_models
            )
        
        except Exception as e:
            return ProviderHealth(
                provider_type=ProviderType.OPENAI,
                is_healthy=False,
                last_check=datetime.utcnow(),
                error_message=str(e)
            )


class AnthropicProvider(BaseLLMProvider):
    """Anthropic provider implementation."""
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.models_cache = None
        self.models_cache_time = None
    
    async def initialize(self) -> bool:
        """Initialize Anthropic client."""
        try:
            # Import Anthropic
            try:
                import anthropic
                self.client = anthropic.AsyncAnthropic(
                    api_key=self.config.api_key,
                    timeout=self.config.timeout,
                    max_retries=self.config.max_retries
                )
                return True
            except ImportError:
                logger.error("Anthropic library not installed")
                return False
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic provider: {e}")
            return False
    
    async def get_available_models(self) -> List[ModelInfo]:
        """Get available Anthropic models."""
        # Cache models for 1 hour
        if (self.models_cache and self.models_cache_time and 
            datetime.utcnow() - self.models_cache_time < timedelta(hours=1)):
            return self.models_cache
        
        # Anthropic doesn't have a models endpoint, so we define known models
        models = [
            ModelInfo(
                id="claude-3-sonnet-20240229",
                name="Claude 3 Sonnet",
                provider=ProviderType.ANTHROPIC,
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.CHAT,
                    ModelCapability.REASONING,
                    ModelCapability.CODE_GENERATION
                ],
                context_window=200000,
                cost_per_token_input=3.0/1000000,  # $3 per million tokens
                cost_per_token_output=15.0/1000000,  # $15 per million tokens
                supports_streaming=True
            ),
            ModelInfo(
                id="claude-3-haiku-20240307",
                name="Claude 3 Haiku",
                provider=ProviderType.ANTHROPIC,
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.CHAT,
                    ModelCapability.REASONING
                ],
                context_window=200000,
                cost_per_token_input=0.25/1000000,  # $0.25 per million tokens
                cost_per_token_output=1.25/1000000,  # $1.25 per million tokens
                supports_streaming=True
            ),
            ModelInfo(
                id="claude-3-opus-20240229",
                name="Claude 3 Opus",
                provider=ProviderType.ANTHROPIC,
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.CHAT,
                    ModelCapability.REASONING,
                    ModelCapability.CODE_GENERATION,
                    ModelCapability.MULTIMODAL
                ],
                context_window=200000,
                cost_per_token_input=15.0/1000000,  # $15 per million tokens
                cost_per_token_output=75.0/1000000,  # $75 per million tokens
                supports_streaming=True
            )
        ]
        
        self.models_cache = models
        self.models_cache_time = datetime.utcnow()
        return models
    
    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Complete a request using Anthropic."""
        start_time = time.time()
        
        try:
            # Apply before hooks
            request = await self._apply_before_hooks(request)
            
            # Prepare Anthropic request
            anthropic_request = self._prepare_anthropic_request(request)
            
            # Make the request
            response = await self.client.messages.create(**anthropic_request)
            
            # Parse response
            llm_response = self._parse_anthropic_response(response, request.model, start_time)
            
            # Apply after hooks
            llm_response = await self._apply_after_hooks(llm_response)
            
            # Update usage stats
            self._update_usage_stats(llm_response)
            
            return llm_response
        
        except Exception as e:
            await self._apply_error_hooks(e, request)
            self.usage_stats.error_count += 1
            raise
    
    async def stream_complete(self, request: LLMRequest) -> AsyncIterator[str]:
        """Stream a completion request."""
        try:
            request = await self._apply_before_hooks(request)
            anthropic_request = self._prepare_anthropic_request(request)
            anthropic_request['stream'] = True
            
            async with self.client.messages.stream(**anthropic_request) as stream:
                async for text in stream.text_stream:
                    yield text
        
        except Exception as e:
            await self._apply_error_hooks(e, request)
            self.usage_stats.error_count += 1
            raise
    
    def _prepare_anthropic_request(self, request: LLMRequest) -> Dict[str, Any]:
        """Prepare request for Anthropic API."""
        # Convert OpenAI-style messages to Anthropic format
        messages = request.messages or []
        if not messages and request.prompt:
            messages = [{'role': 'user', 'content': request.prompt}]
        
        # Extract system message if present
        system_message = None
        user_messages = []
        
        for msg in messages:
            if msg['role'] == 'system':
                system_message = msg['content']
            else:
                user_messages.append(msg)
        
        anthropic_request = {
            'model': request.model or 'claude-3-haiku-20240307',
            'messages': user_messages,
            'max_tokens': request.max_tokens or 1000,
        }
        
        if system_message:
            anthropic_request['system'] = system_message
        if request.temperature is not None:
            anthropic_request['temperature'] = request.temperature
        if request.top_p is not None:
            anthropic_request['top_p'] = request.top_p
        if request.stop:
            anthropic_request['stop_sequences'] = request.stop
        
        return anthropic_request
    
    def _parse_anthropic_response(self, response, model: str, start_time: float) -> LLMResponse:
        """Parse Anthropic response."""
        content = ""
        if response.content:
            content = "".join([block.text for block in response.content if hasattr(block, 'text')])
        
        # Calculate cost
        usage = response.usage
        cost_usd = None
        if usage:
            model_costs = self._get_anthropic_model_costs(model)
            if model_costs:
                cost_usd = (
                    usage.input_tokens * model_costs['input_cost'] +
                    usage.output_tokens * model_costs['output_cost']
                )
        
        return LLMResponse(
            provider=ProviderType.ANTHROPIC,
            model=model,
            content=content,
            usage={'prompt_tokens': usage.input_tokens, 'completion_tokens': usage.output_tokens} if usage else None,
            finish_reason=response.stop_reason,
            response_time_ms=(time.time() - start_time) * 1000,
            cost_usd=cost_usd,
            request_id=response.id
        )
    
    def _get_anthropic_model_costs(self, model: str) -> Optional[Dict[str, float]]:
        """Get cost information for Anthropic models."""
        cost_map = {
            'claude-3-haiku': {'input_cost': 0.25/1000000, 'output_cost': 1.25/1000000},
            'claude-3-sonnet': {'input_cost': 3.0/1000000, 'output_cost': 15.0/1000000},
            'claude-3-opus': {'input_cost': 15.0/1000000, 'output_cost': 75.0/1000000},
        }
        
        for model_key, costs in cost_map.items():
            if model_key in model:
                return costs
        
        return None
    
    async def health_check(self) -> ProviderHealth:
        """Check Anthropic provider health."""
        start_time = time.time()
        
        try:
            # Simple health check: send a basic message
            test_message = await self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=10,
                messages=[{"role": "user", "content": "Hi"}]
            )
            
            response_time = (time.time() - start_time) * 1000
            available_models = [model.id for model in await self.get_available_models()]
            
            return ProviderHealth(
                provider_type=ProviderType.ANTHROPIC,
                is_healthy=True,
                last_check=datetime.utcnow(),
                response_time_ms=response_time,
                available_models=available_models[:5]
            )
        
        except Exception as e:
            return ProviderHealth(
                provider_type=ProviderType.ANTHROPIC,
                is_healthy=False,
                last_check=datetime.utcnow(),
                error_message=str(e)
            )


class LLMProviderManager:
    """Manager for all LLM providers."""
    
    def __init__(self):
        self.providers: Dict[ProviderType, BaseLLMProvider] = {}
        self.settings = get_settings()
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize all configured providers."""
        # OpenAI
        if self.settings.openai_api_key:
            openai_config = ProviderConfig(
                provider_type=ProviderType.OPENAI,
                api_key=self.settings.openai_api_key,
                enabled=True
            )
            self.providers[ProviderType.OPENAI] = OpenAIProvider(openai_config)
        
        # Anthropic
        if self.settings.anthropic_api_key:
            anthropic_config = ProviderConfig(
                provider_type=ProviderType.ANTHROPIC,
                api_key=self.settings.anthropic_api_key,
                enabled=True
            )
            self.providers[ProviderType.ANTHROPIC] = AnthropicProvider(anthropic_config)
    
    async def initialize_all(self) -> Dict[ProviderType, bool]:
        """Initialize all providers."""
        results = {}
        for provider_type, provider in self.providers.items():
            try:
                results[provider_type] = await provider.initialize()
            except Exception as e:
                logger.error(f"Failed to initialize {provider_type}: {e}")
                results[provider_type] = False
        return results
    
    async def get_available_providers(self) -> List[ProviderType]:
        """Get list of available providers."""
        available = []
        for provider_type, provider in self.providers.items():
            if provider.config.enabled:
                try:
                    health = await provider.health_check()
                    if health.is_healthy:
                        available.append(provider_type)
                except Exception:
                    pass
        return available
    
    async def get_all_models(self) -> Dict[ProviderType, List[ModelInfo]]:
        """Get all available models from all providers."""
        all_models = {}
        for provider_type, provider in self.providers.items():
            if provider.config.enabled:
                try:
                    models = await provider.get_available_models()
                    all_models[provider_type] = models
                except Exception as e:
                    logger.error(f"Failed to get models from {provider_type}: {e}")
                    all_models[provider_type] = []
        return all_models
    
    async def complete(
        self, 
        request: LLMRequest, 
        provider_type: Optional[ProviderType] = None
    ) -> LLMResponse:
        """Complete a request using specified or best available provider."""
        if provider_type and provider_type in self.providers:
            provider = self.providers[provider_type]
        else:
            # Auto-select best available provider
            available_providers = await self.get_available_providers()
            if not available_providers:
                raise RuntimeError("No LLM providers available")
            provider = self.providers[available_providers[0]]
        
        return await provider.complete(request)
    
    async def stream_complete(
        self, 
        request: LLMRequest, 
        provider_type: Optional[ProviderType] = None
    ) -> AsyncIterator[str]:
        """Stream complete a request."""
        if provider_type and provider_type in self.providers:
            provider = self.providers[provider_type]
        else:
            available_providers = await self.get_available_providers()
            if not available_providers:
                raise RuntimeError("No LLM providers available")
            provider = self.providers[available_providers[0]]
        
        async for chunk in provider.stream_complete(request):
            yield chunk
    
    async def get_provider_health(self) -> Dict[ProviderType, ProviderHealth]:
        """Get health status of all providers."""
        health_status = {}
        for provider_type, provider in self.providers.items():
            try:
                health = await provider.health_check()
                health_status[provider_type] = health
            except Exception as e:
                health_status[provider_type] = ProviderHealth(
                    provider_type=provider_type,
                    is_healthy=False,
                    last_check=datetime.utcnow(),
                    error_message=str(e)
                )
        return health_status
    
    async def get_usage_stats(self) -> Dict[ProviderType, UsageStats]:
        """Get usage statistics for all providers."""
        stats = {}
        for provider_type, provider in self.providers.items():
            stats[provider_type] = provider.usage_stats
        return stats
    
    def add_global_hook(self, hook: ProviderHook):
        """Add a hook to all providers."""
        for provider in self.providers.values():
            provider.add_hook(hook)
    
    def add_provider_hook(self, provider_type: ProviderType, hook: ProviderHook):
        """Add a hook to a specific provider."""
        if provider_type in self.providers:
            self.providers[provider_type].add_hook(hook)


# Global provider manager instance
llm_provider_manager = LLMProviderManager()


# Common provider hooks
class LoggingHook(ProviderHook):
    """Hook for logging requests and responses."""
    
    def __init__(self):
        super().__init__("logging")
    
    async def before_request(self, request: LLMRequest) -> LLMRequest:
        logger.info(f"LLM Request: {request.operation_type} - Model: {request.model}")
        return request
    
    async def after_response(self, response: LLMResponse) -> LLMResponse:
        logger.info(f"LLM Response: {response.provider} - Cost: ${response.cost_usd:.4f}")
        return response
    
    async def on_error(self, error: Exception, request: LLMRequest) -> None:
        logger.error(f"LLM Error: {error} for request {request.operation_type}")


class CostTrackingHook(ProviderHook):
    """Hook for tracking costs in cache."""
    
    def __init__(self):
        super().__init__("cost_tracking")
    
    async def after_response(self, response: LLMResponse) -> LLMResponse:
        if response.cost_usd:
            # Update cost tracking in cache
            cost_key = f"llm_cost:{response.provider}:{datetime.utcnow().strftime('%Y-%m-%d')}"
            current_cost = await cache_service.get(cost_key) or 0.0
            await cache_service.set(
                cost_key, 
                current_cost + response.cost_usd, 
                ttl=86400 * 7  # 7 days
            )
        return response 