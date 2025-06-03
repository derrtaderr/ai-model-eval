"""
LLM Provider Management API
Comprehensive API for managing OpenAI, Anthropic, and other LLM providers.

Provides endpoints for:
- Provider health monitoring
- Model discovery and selection
- Configuration management
- Usage analytics and cost tracking
- Provider testing and validation
"""

from datetime import datetime
from typing import Dict, Any, Optional, List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from auth.dependencies import get_current_user, require_permission
from database.models import User
from services.llm_providers import (
    llm_provider_manager,
    ProviderType,
    OperationType,
    ModelCapability,
    LLMRequest,
    ModelInfo,
    ProviderHealth,
    UsageStats,
    LoggingHook,
    CostTrackingHook
)


router = APIRouter(prefix="/llm-providers", tags=["LLM Providers"])


class LLMCompletionRequest(BaseModel):
    """Request model for LLM completion."""
    operation_type: OperationType = Field(OperationType.CHAT, description="Type of LLM operation")
    messages: Optional[List[Dict[str, Any]]] = Field(None, description="Chat messages")
    prompt: Optional[str] = Field(None, description="Simple prompt for completion")
    model: Optional[str] = Field(None, description="Specific model to use")
    provider: Optional[ProviderType] = Field(None, description="Specific provider to use")
    max_tokens: Optional[int] = Field(None, ge=1, le=4000, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    stream: bool = Field(False, description="Whether to stream the response")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class LLMCompletionResponse(BaseModel):
    """Response model for LLM completion."""
    provider: ProviderType
    model: str
    content: str
    usage: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None
    response_time_ms: Optional[float] = None
    cost_usd: Optional[float] = None
    request_id: Optional[str] = None


class ProviderTestRequest(BaseModel):
    """Request model for testing a provider."""
    provider: ProviderType
    test_prompt: str = Field("Hello! Please respond with a brief greeting.", description="Test prompt to send")


class ProviderConfigRequest(BaseModel):
    """Request model for updating provider configuration."""
    provider: ProviderType
    enabled: bool = Field(True, description="Whether the provider is enabled")
    timeout: Optional[int] = Field(30, ge=1, le=300, description="Request timeout in seconds")
    max_retries: Optional[int] = Field(3, ge=0, le=10, description="Maximum retry attempts")


@router.get("/", response_model=Dict[str, Any])
async def get_providers_overview(
    current_user: User = Depends(get_current_user)
):
    """
    Get an overview of all configured LLM providers.
    
    Returns provider status, health, and basic configuration information.
    """
    try:
        # Get provider health status
        health_status = await llm_provider_manager.get_provider_health()
        
        # Get available providers
        available_providers = await llm_provider_manager.get_available_providers()
        
        # Get usage statistics
        usage_stats = await llm_provider_manager.get_usage_stats()
        
        return {
            "providers": [
                {
                    "type": provider_type.value,
                    "name": provider_type.value.title(),
                    "is_available": provider_type in available_providers,
                    "is_healthy": health_status.get(provider_type, {}).is_healthy if provider_type in health_status else False,
                    "last_check": health_status.get(provider_type, {}).last_check.isoformat() if provider_type in health_status and health_status[provider_type].last_check else None,
                    "response_time_ms": health_status.get(provider_type, {}).response_time_ms if provider_type in health_status else None,
                    "error_message": health_status.get(provider_type, {}).error_message if provider_type in health_status else None,
                    "requests_count": usage_stats.get(provider_type, UsageStats(provider_type)).requests_count,
                    "total_cost_usd": usage_stats.get(provider_type, UsageStats(provider_type)).total_cost_usd,
                    "avg_response_time_ms": usage_stats.get(provider_type, UsageStats(provider_type)).avg_response_time_ms
                }
                for provider_type in ProviderType
                if provider_type in llm_provider_manager.providers
            ],
            "summary": {
                "total_providers": len(llm_provider_manager.providers),
                "available_providers": len(available_providers),
                "total_requests": sum(stats.requests_count for stats in usage_stats.values()),
                "total_cost_usd": sum(stats.total_cost_usd for stats in usage_stats.values()),
                "last_updated": datetime.utcnow().isoformat()
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting providers overview: {str(e)}"
        )


@router.get("/models", response_model=Dict[ProviderType, List[Dict[str, Any]]])
async def get_all_models(
    provider: Optional[ProviderType] = None,
    capability: Optional[ModelCapability] = None,
    current_user: User = Depends(get_current_user)
):
    """
    Get all available models from all providers or a specific provider.
    
    Optionally filter by model capabilities.
    """
    try:
        if provider:
            # Get models from specific provider
            if provider not in llm_provider_manager.providers:
                raise HTTPException(status_code=404, detail=f"Provider {provider} not found")
            
            models = await llm_provider_manager.providers[provider].get_available_models()
            
            # Filter by capability if specified
            if capability:
                models = [model for model in models if capability in model.capabilities]
            
            return {provider: [model.__dict__ for model in models]}
        else:
            # Get models from all providers
            all_models = await llm_provider_manager.get_all_models()
            
            # Filter by capability if specified
            if capability:
                filtered_models = {}
                for provider_type, models in all_models.items():
                    filtered_models[provider_type] = [
                        model for model in models if capability in model.capabilities
                    ]
                all_models = filtered_models
            
            # Convert ModelInfo objects to dictionaries
            return {
                provider_type: [model.__dict__ for model in models]
                for provider_type, models in all_models.items()
            }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting models: {str(e)}"
        )


@router.post("/complete", response_model=LLMCompletionResponse)
async def complete_llm_request(
    completion_request: LLMCompletionRequest,
    current_user: User = Depends(require_permission("llm:use"))
):
    """
    Complete an LLM request using the specified or best available provider.
    
    Supports both chat-style messages and simple prompt completion.
    """
    try:
        # Convert request to internal format
        llm_request = LLMRequest(
            operation_type=completion_request.operation_type,
            messages=completion_request.messages,
            prompt=completion_request.prompt,
            model=completion_request.model,
            max_tokens=completion_request.max_tokens,
            temperature=completion_request.temperature,
            top_p=completion_request.top_p,
            stream=completion_request.stream,
            user_id=str(current_user.id),
            metadata={
                **completion_request.metadata,
                "team_id": str(current_user.team_id) if current_user.team_id else None,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        # Handle streaming request
        if completion_request.stream:
            raise HTTPException(
                status_code=400,
                detail="Use /complete-stream endpoint for streaming requests"
            )
        
        # Complete the request
        response = await llm_provider_manager.complete(
            llm_request,
            provider_type=completion_request.provider
        )
        
        return LLMCompletionResponse(
            provider=response.provider,
            model=response.model,
            content=response.content,
            usage=response.usage,
            finish_reason=response.finish_reason,
            response_time_ms=response.response_time_ms,
            cost_usd=response.cost_usd,
            request_id=response.request_id
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error completing LLM request: {str(e)}"
        )


@router.post("/complete-stream")
async def stream_llm_request(
    completion_request: LLMCompletionRequest,
    current_user: User = Depends(require_permission("llm:use"))
):
    """
    Stream an LLM completion request.
    
    Returns a streaming response with Server-Sent Events.
    """
    try:
        # Convert request to internal format
        llm_request = LLMRequest(
            operation_type=completion_request.operation_type,
            messages=completion_request.messages,
            prompt=completion_request.prompt,
            model=completion_request.model,
            max_tokens=completion_request.max_tokens,
            temperature=completion_request.temperature,
            top_p=completion_request.top_p,
            stream=True,
            user_id=str(current_user.id),
            metadata={
                **completion_request.metadata,
                "team_id": str(current_user.team_id) if current_user.team_id else None,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        async def generate_stream():
            try:
                async for chunk in llm_provider_manager.stream_complete(
                    llm_request,
                    provider_type=completion_request.provider
                ):
                    yield f"data: {chunk}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                yield f"data: [ERROR] {str(e)}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive"
            }
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error streaming LLM request: {str(e)}"
        )


@router.get("/health", response_model=Dict[ProviderType, Dict[str, Any]])
async def get_provider_health(
    current_user: User = Depends(get_current_user)
):
    """
    Get detailed health status for all LLM providers.
    
    Includes response times, available models, and error information.
    """
    try:
        health_status = await llm_provider_manager.get_provider_health()
        
        return {
            provider_type: {
                "is_healthy": health.is_healthy,
                "last_check": health.last_check.isoformat(),
                "response_time_ms": health.response_time_ms,
                "error_message": health.error_message,
                "available_models": health.available_models[:5],  # First 5 models
                "rate_limit_status": health.rate_limit_status
            }
            for provider_type, health in health_status.items()
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting provider health: {str(e)}"
        )


@router.post("/test")
async def test_provider(
    test_request: ProviderTestRequest,
    current_user: User = Depends(require_permission("admin"))
):
    """
    Test a specific LLM provider with a simple request.
    
    Admin-only endpoint for validating provider configurations.
    """
    try:
        if test_request.provider not in llm_provider_manager.providers:
            raise HTTPException(
                status_code=404,
                detail=f"Provider {test_request.provider} not configured"
            )
        
        # Create test request
        llm_request = LLMRequest(
            operation_type=OperationType.CHAT,
            messages=[{"role": "user", "content": test_request.test_prompt}],
            model=None,  # Use default model
            max_tokens=50,
            temperature=0.7,
            user_id=str(current_user.id),
            metadata={"test": True}
        )
        
        # Test the provider
        start_time = datetime.utcnow()
        response = await llm_provider_manager.complete(
            llm_request,
            provider_type=test_request.provider
        )
        end_time = datetime.utcnow()
        
        return {
            "status": "success",
            "provider": test_request.provider.value,
            "test_prompt": test_request.test_prompt,
            "response_content": response.content,
            "response_time_ms": response.response_time_ms,
            "cost_usd": response.cost_usd,
            "model_used": response.model,
            "test_duration_ms": (end_time - start_time).total_seconds() * 1000,
            "timestamp": start_time.isoformat()
        }
    
    except Exception as e:
        return {
            "status": "error",
            "provider": test_request.provider.value,
            "error_message": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


@router.get("/usage-stats", response_model=Dict[ProviderType, Dict[str, Any]])
async def get_usage_statistics(
    current_user: User = Depends(get_current_user)
):
    """
    Get usage statistics for all LLM providers.
    
    Includes request counts, token usage, costs, and performance metrics.
    """
    try:
        usage_stats = await llm_provider_manager.get_usage_stats()
        
        return {
            provider_type: {
                "requests_count": stats.requests_count,
                "tokens_input": stats.tokens_input,
                "tokens_output": stats.tokens_output,
                "total_cost_usd": stats.total_cost_usd,
                "avg_response_time_ms": stats.avg_response_time_ms,
                "error_count": stats.error_count,
                "last_request": stats.last_request.isoformat() if stats.last_request else None,
                "cost_per_request": stats.total_cost_usd / stats.requests_count if stats.requests_count > 0 else 0,
                "success_rate": ((stats.requests_count - stats.error_count) / stats.requests_count * 100) if stats.requests_count > 0 else 0
            }
            for provider_type, stats in usage_stats.items()
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting usage statistics: {str(e)}"
        )


@router.post("/initialize")
async def initialize_providers(
    current_user: User = Depends(require_permission("admin"))
):
    """
    Initialize or re-initialize all LLM providers.
    
    Admin-only endpoint for forcing provider re-initialization.
    """
    try:
        initialization_results = await llm_provider_manager.initialize_all()
        
        return {
            "status": "completed",
            "results": {
                provider_type.value: {
                    "initialized": success,
                    "status": "success" if success else "failed"
                }
                for provider_type, success in initialization_results.items()
            },
            "summary": {
                "total_providers": len(initialization_results),
                "successful": sum(1 for success in initialization_results.values() if success),
                "failed": sum(1 for success in initialization_results.values() if not success)
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error initializing providers: {str(e)}"
        )


@router.post("/hooks/enable")
async def enable_provider_hooks(
    current_user: User = Depends(require_permission("admin"))
):
    """
    Enable common provider hooks for logging and cost tracking.
    
    Admin-only endpoint for enabling system-wide hooks.
    """
    try:
        # Add logging hook
        logging_hook = LoggingHook()
        llm_provider_manager.add_global_hook(logging_hook)
        
        # Add cost tracking hook
        cost_hook = CostTrackingHook()
        llm_provider_manager.add_global_hook(cost_hook)
        
        return {
            "status": "success",
            "message": "Provider hooks enabled successfully",
            "hooks_enabled": ["logging", "cost_tracking"],
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error enabling provider hooks: {str(e)}"
        )


@router.get("/capabilities", response_model=List[Dict[str, Any]])
async def get_model_capabilities(
    current_user: User = Depends(get_current_user)
):
    """
    Get a summary of model capabilities across all providers.
    
    Returns available capabilities and which models support them.
    """
    try:
        all_models = await llm_provider_manager.get_all_models()
        
        # Aggregate capabilities
        capability_summary = {}
        
        for provider_type, models in all_models.items():
            for model in models:
                for capability in model.capabilities:
                    if capability not in capability_summary:
                        capability_summary[capability] = {
                            "capability": capability.value,
                            "description": capability.value.replace("_", " ").title(),
                            "providers": [],
                            "models": []
                        }
                    
                    if provider_type.value not in capability_summary[capability]["providers"]:
                        capability_summary[capability]["providers"].append(provider_type.value)
                    
                    capability_summary[capability]["models"].append({
                        "id": model.id,
                        "name": model.name,
                        "provider": provider_type.value
                    })
        
        return list(capability_summary.values())
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting model capabilities: {str(e)}"
        )


@router.get("/cost-analysis", response_model=Dict[str, Any])
async def get_cost_analysis(
    days: int = 7,
    current_user: User = Depends(get_current_user)
):
    """
    Get cost analysis for LLM providers over the specified time period.
    
    Includes cost breakdown by provider, trends, and recommendations.
    """
    try:
        from utils.cache_service import cache_service
        
        # Get cost data from cache
        cost_data = {}
        total_cost = 0.0
        
        for provider_type in ProviderType:
            provider_cost = 0.0
            for day_offset in range(days):
                date = (datetime.utcnow() - timedelta(days=day_offset)).strftime('%Y-%m-%d')
                cost_key = f"llm_cost:{provider_type.value}:{date}"
                daily_cost = await cache_service.get(cost_key) or 0.0
                provider_cost += daily_cost
            
            if provider_cost > 0:
                cost_data[provider_type.value] = provider_cost
                total_cost += provider_cost
        
        # Calculate percentages and recommendations
        cost_breakdown = []
        for provider, cost in cost_data.items():
            percentage = (cost / total_cost * 100) if total_cost > 0 else 0
            cost_breakdown.append({
                "provider": provider,
                "cost_usd": cost,
                "percentage": percentage
            })
        
        # Sort by cost descending
        cost_breakdown.sort(key=lambda x: x["cost_usd"], reverse=True)
        
        return {
            "period_days": days,
            "total_cost_usd": total_cost,
            "cost_breakdown": cost_breakdown,
            "average_daily_cost": total_cost / days if days > 0 else 0,
            "projected_monthly_cost": (total_cost / days) * 30 if days > 0 else 0,
            "recommendations": [
                "Consider using lower-cost models for simple tasks",
                "Monitor usage patterns to optimize provider selection",
                "Enable request caching to reduce duplicate calls"
            ] if total_cost > 10 else [
                "Current usage is within normal range",
                "Consider testing more advanced models for complex tasks"
            ],
            "last_updated": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting cost analysis: {str(e)}"
        ) 