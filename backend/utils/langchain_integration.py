"""
LangChain integration utilities for the LLM Evaluation Platform.
"""

import asyncio
from typing import Dict, Any, Optional, List
from uuid import uuid4
from datetime import datetime

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import BaseMessage
from langsmith import Client
from decouple import config

from services.trace_logger import trace_logger


class EvalPlatformCallbackHandler(BaseCallbackHandler):
    """
    Custom callback handler that captures LangChain traces and logs them to our platform.
    """
    
    def __init__(self, session_id: Optional[str] = None, user_id: Optional[str] = None):
        super().__init__()
        self.session_id = session_id or str(uuid4())
        self.user_id = user_id
        self.run_data = {}
        self.start_times = {}
    
    def on_llm_start(
        self, 
        serialized: Dict[str, Any], 
        prompts: List[str], 
        **kwargs: Any
    ) -> Any:
        """Called when LLM starts running."""
        run_id = kwargs.get("run_id")
        if run_id:
            self.start_times[str(run_id)] = datetime.utcnow()
            self.run_data[str(run_id)] = {
                "model_name": serialized.get("_type", "unknown"),
                "prompts": prompts,
                "metadata": {
                    "serialized": serialized,
                    "kwargs": kwargs
                }
            }
    
    def on_llm_end(self, response, **kwargs: Any) -> Any:
        """Called when LLM ends running."""
        run_id = kwargs.get("run_id")
        if run_id and str(run_id) in self.run_data:
            run_id_str = str(run_id)
            start_time = self.start_times.get(run_id_str)
            latency_ms = None
            
            if start_time:
                latency_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            # Extract response information
            output_text = ""
            token_usage = None
            
            if hasattr(response, 'generations') and response.generations:
                # Handle LLMResult
                generation = response.generations[0][0]
                output_text = generation.text
                
                if hasattr(response, 'llm_output') and response.llm_output:
                    token_usage = response.llm_output.get('token_usage')
            
            # Log trace asynchronously
            asyncio.create_task(self._log_trace_async(
                run_id_str=run_id_str,
                output_text=output_text,
                latency_ms=latency_ms,
                token_usage=token_usage
            ))
    
    def on_llm_error(self, error: Exception, **kwargs: Any) -> Any:
        """Called when LLM errors."""
        run_id = kwargs.get("run_id")
        if run_id and str(run_id) in self.run_data:
            # Log error trace
            asyncio.create_task(self._log_error_trace_async(str(run_id), str(error)))
    
    async def _log_trace_async(
        self, 
        run_id_str: str, 
        output_text: str, 
        latency_ms: Optional[int],
        token_usage: Optional[Dict[str, int]]
    ):
        """Asynchronously log a successful trace."""
        try:
            run_data = self.run_data.get(run_id_str, {})
            prompts = run_data.get("prompts", [])
            
            # Combine prompts into user input
            user_input = "\n".join(prompts) if prompts else ""
            
            # Calculate cost estimate (rough)
            cost_usd = None
            if token_usage:
                # Very rough cost estimation - would need actual model pricing
                input_tokens = token_usage.get("prompt_tokens", 0)
                output_tokens = token_usage.get("completion_tokens", 0)
                # Assume ~$0.02 per 1K tokens (varies by model)
                cost_usd = (input_tokens + output_tokens) * 0.00002
            
            await trace_logger.log_trace(
                user_input=user_input,
                model_output=output_text,
                model_name=run_data.get("model_name", "unknown"),
                session_id=self.session_id,
                user_id=self.user_id,
                metadata={
                    "langchain_run_id": run_id_str,
                    **run_data.get("metadata", {})
                },
                latency_ms=latency_ms,
                token_count=token_usage,
                cost_usd=cost_usd
            )
            
            # Clean up
            if run_id_str in self.run_data:
                del self.run_data[run_id_str]
            if run_id_str in self.start_times:
                del self.start_times[run_id_str]
                
        except Exception as e:
            print(f"Error logging trace: {e}")
    
    async def _log_error_trace_async(self, run_id_str: str, error_msg: str):
        """Asynchronously log a failed trace."""
        try:
            run_data = self.run_data.get(run_id_str, {})
            prompts = run_data.get("prompts", [])
            user_input = "\n".join(prompts) if prompts else ""
            
            await trace_logger.log_trace(
                user_input=user_input,
                model_output=f"ERROR: {error_msg}",
                model_name=run_data.get("model_name", "unknown"),
                session_id=self.session_id,
                user_id=self.user_id,
                metadata={
                    "langchain_run_id": run_id_str,
                    "error": error_msg,
                    **run_data.get("metadata", {})
                },
                latency_ms=None
            )
            
            # Clean up
            if run_id_str in self.run_data:
                del self.run_data[run_id_str]
            if run_id_str in self.start_times:
                del self.start_times[run_id_str]
                
        except Exception as e:
            print(f"Error logging error trace: {e}")


def create_eval_callback_handler(session_id: Optional[str] = None, user_id: Optional[str] = None) -> EvalPlatformCallbackHandler:
    """
    Create a callback handler for capturing traces in the evaluation platform.
    
    Args:
        session_id: Optional session identifier
        user_id: Optional user identifier
    
    Returns:
        Configured callback handler
    """
    return EvalPlatformCallbackHandler(session_id=session_id, user_id=user_id)


async def setup_langsmith_integration() -> Optional[Client]:
    """
    Set up LangSmith integration if configured.
    
    Returns:
        LangSmith client if configured, None otherwise
    """
    api_key = config("LANGCHAIN_API_KEY", default=None)
    if not api_key:
        print("LangSmith API key not found. Skipping LangSmith integration.")
        return None
    
    try:
        client = Client(api_key=api_key)
        
        # Test the connection
        try:
            list(client.list_datasets(limit=1))
            print("LangSmith integration configured successfully")
            return client
        except Exception as e:
            print(f"LangSmith connection test failed: {e}")
            return None
            
    except Exception as e:
        print(f"Failed to initialize LangSmith client: {e}")
        return None


def get_langchain_config_for_tracing(session_id: Optional[str] = None, user_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Get LangChain configuration that includes our custom tracing.
    
    Args:
        session_id: Optional session identifier
        user_id: Optional user identifier
    
    Returns:
        Configuration dict for LangChain
    """
    callbacks = [create_eval_callback_handler(session_id=session_id, user_id=user_id)]
    
    # Add LangSmith tracing if configured
    if config("LANGCHAIN_TRACING_V2", default="false").lower() == "true":
        from langchain.callbacks.tracers import LangChainTracer
        
        tracer = LangChainTracer(
            project_name=config("LANGCHAIN_PROJECT", default="llm-eval-platform")
        )
        callbacks.append(tracer)
    
    return {
        "callbacks": callbacks,
        "metadata": {
            "session_id": session_id,
            "user_id": user_id,
            "platform": "llm-eval-platform"
        }
    } 