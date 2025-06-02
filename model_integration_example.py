"""
Example: How to integrate your LLM models with the Evaluation Platform
This shows how to log traces to the platform for analysis.
"""

import openai
import httpx
import time
from typing import Dict, Any, Optional

class LLMEvaluationLogger:
    """Logger that sends traces to the evaluation platform."""
    
    def __init__(self, platform_url: str = "http://localhost:8000", api_key: str = "your-api-key"):
        self.platform_url = platform_url
        self.api_key = api_key
        self.client = httpx.AsyncClient()
    
    async def log_trace(
        self,
        user_input: str,
        model_output: str,
        model_name: str,
        tool: str,
        scenario: str,
        system_prompt: Optional[str] = None,
        session_id: Optional[str] = None,
        latency_ms: Optional[int] = None,
        token_count: Optional[Dict[str, int]] = None,
        cost_usd: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Send trace to evaluation platform."""
        
        trace_data = {
            "user_input": user_input,
            "model_output": model_output,
            "model_name": model_name,
            "tool": tool,
            "scenario": scenario,
            "system_prompt": system_prompt,
            "session_id": session_id,
            "latency_ms": latency_ms,
            "token_count": token_count,
            "cost_usd": cost_usd,
            "metadata": metadata or {}
        }
        
        try:
            response = await self.client.post(
                f"{self.platform_url}/api/traces",
                json=trace_data,
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Failed to log trace: {e}")
            return None

# Example 1: OpenAI GPT Integration
class GPTWithEvaluation:
    def __init__(self):
        self.openai_client = openai.AsyncOpenAI()
        self.eval_logger = LLMEvaluationLogger()
    
    async def generate_content(self, prompt: str, session_id: str = None) -> str:
        """Generate content and log to evaluation platform."""
        
        start_time = time.time()
        
        # Call OpenAI API
        response = await self.openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        end_time = time.time()
        latency_ms = int((end_time - start_time) * 1000)
        
        # Extract response details
        ai_response = response.choices[0].message.content
        usage = response.usage
        
        # Log to evaluation platform
        await self.eval_logger.log_trace(
            user_input=prompt,
            model_output=ai_response,
            model_name="gpt-4",
            tool="Content-Generator",  # Your use case
            scenario="Blog-Writing",   # Specific scenario
            system_prompt="You are a helpful assistant.",
            session_id=session_id,
            latency_ms=latency_ms,
            token_count={
                "input": usage.prompt_tokens,
                "output": usage.completion_tokens
            },
            cost_usd=self.calculate_cost(usage),
            metadata={
                "temperature": 0.7,
                "max_tokens": 500,
                "model_version": "gpt-4"
            }
        )
        
        return ai_response
    
    def calculate_cost(self, usage) -> float:
        """Calculate cost based on token usage."""
        # GPT-4 pricing example (update with current rates)
        input_cost = usage.prompt_tokens * 0.00003  # $0.03 per 1K tokens
        output_cost = usage.completion_tokens * 0.00006  # $0.06 per 1K tokens
        return input_cost + output_cost

# Example 2: Custom Model Integration
class CustomModelWithEvaluation:
    def __init__(self, model_endpoint: str):
        self.model_endpoint = model_endpoint
        self.eval_logger = LLMEvaluationLogger()
    
    async def process_customer_query(self, query: str, customer_id: str) -> str:
        """Process customer query and log to evaluation platform."""
        
        start_time = time.time()
        
        # Call your custom model
        response = await self.call_custom_model(query)
        
        end_time = time.time()
        latency_ms = int((end_time - start_time) * 1000)
        
        # Log to evaluation platform
        await self.eval_logger.log_trace(
            user_input=query,
            model_output=response["text"],
            model_name="custom-support-model-v2",
            tool="Customer-Support",      # Your tool/service
            scenario="Query-Resolution",  # Specific use case
            session_id=customer_id,
            latency_ms=latency_ms,
            token_count=response.get("token_count"),
            cost_usd=response.get("cost"),
            metadata={
                "customer_tier": "premium",
                "model_version": "v2.1",
                "confidence_score": response.get("confidence", 0.0)
            }
        )
        
        return response["text"]
    
    async def call_custom_model(self, query: str) -> Dict[str, Any]:
        """Call your custom model endpoint."""
        # Your model implementation here
        return {
            "text": "This is a sample response",
            "token_count": {"input": 25, "output": 15},
            "cost": 0.002,
            "confidence": 0.95
        }

# Example 3: Multi-Model A/B Testing
class ABTestingSetup:
    def __init__(self):
        self.eval_logger = LLMEvaluationLogger()
        self.gpt_client = openai.AsyncOpenAI()
    
    async def compare_models(self, prompt: str, user_id: str):
        """Run A/B test between different models."""
        
        # Determine which variant to use (50/50 split)
        variant = "gpt-4" if hash(user_id) % 2 == 0 else "gpt-3.5-turbo"
        
        start_time = time.time()
        
        if variant == "gpt-4":
            response = await self.gpt_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
        else:
            response = await self.gpt_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
        
        end_time = time.time()
        ai_response = response.choices[0].message.content
        
        # Log with experiment metadata
        await self.eval_logger.log_trace(
            user_input=prompt,
            model_output=ai_response,
            model_name=variant,
            tool="Chat-Assistant",
            scenario="General-Query",
            session_id=f"experiment_{user_id}",
            latency_ms=int((end_time - start_time) * 1000),
            token_count={
                "input": response.usage.prompt_tokens,
                "output": response.usage.completion_tokens
            },
            cost_usd=self.calculate_cost(response.usage, variant),
            metadata={
                "experiment": "gpt4_vs_gpt35_chat",
                "variant": variant,
                "user_segment": "beta_users"
            }
        )
        
        return ai_response
    
    def calculate_cost(self, usage, model: str) -> float:
        """Calculate cost based on model and usage."""
        if model == "gpt-4":
            return usage.prompt_tokens * 0.00003 + usage.completion_tokens * 0.00006
        else:  # gpt-3.5-turbo
            return usage.prompt_tokens * 0.0000015 + usage.completion_tokens * 0.000002

# Usage Examples
async def main():
    # Example 1: Content generation
    content_generator = GPTWithEvaluation()
    blog_post = await content_generator.generate_content(
        "Write about AI in healthcare",
        session_id="content_session_123"
    )
    
    # Example 2: Customer support
    support_bot = CustomModelWithEvaluation("https://api.yourmodel.com/chat")
    response = await support_bot.process_customer_query(
        "My order is delayed, what should I do?",
        customer_id="customer_456"
    )
    
    # Example 3: A/B testing
    ab_tester = ABTestingSetup()
    chat_response = await ab_tester.compare_models(
        "Explain quantum computing simply",
        user_id="user_789"
    )

if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 