"""
Model-Based Evaluation Engine
Integrates multiple LLM providers for automatic trace evaluation.
Part of Task 6 - Model-Based Evaluation Engine.
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from config.settings import get_settings

# Import template library
try:
    from services.evaluation_templates import template_library
    TEMPLATES_AVAILABLE = True
except ImportError:
    template_library = None
    TEMPLATES_AVAILABLE = False

# Import calibration system
try:
    from services.scoring_calibration import calibration_system
    CALIBRATION_AVAILABLE = True
except ImportError:
    calibration_system = None
    CALIBRATION_AVAILABLE = False

logger = logging.getLogger(__name__)

class EvaluatorProvider(str, Enum):
    """Supported evaluator providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic" 
    LOCAL = "local"
    HUGGINGFACE = "huggingface"

class EvaluationCriteria(str, Enum):
    """Standard evaluation criteria."""
    COHERENCE = "coherence"
    RELEVANCE = "relevance"
    FACTUAL_ACCURACY = "factual_accuracy"
    GRAMMAR = "grammar"
    STYLE = "style"
    HELPFULNESS = "helpfulness"
    HARMFULNESS = "harmfulness"
    TRUTHFULNESS = "truthfulness"
    COMPLETENESS = "completeness"

@dataclass
class EvaluationRequest:
    """Request for model-based evaluation."""
    trace_id: str
    user_input: str
    model_output: str
    system_prompt: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    criteria: List[EvaluationCriteria] = field(default_factory=lambda: [EvaluationCriteria.COHERENCE, EvaluationCriteria.RELEVANCE])
    custom_prompt: Optional[str] = None
    reference_answer: Optional[str] = None

@dataclass
class EvaluationResult:
    """Result from model-based evaluation."""
    trace_id: str
    evaluator_model: str
    criteria: EvaluationCriteria
    score: float  # 0.0 to 1.0
    reasoning: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    evaluation_time_ms: Optional[int] = None
    cost_usd: Optional[float] = None

@dataclass
class BatchEvaluationResult:
    """Result from batch evaluation."""
    total_traces: int
    successful_evaluations: int
    failed_evaluations: int
    results: List[EvaluationResult]
    errors: List[Dict[str, Any]]
    total_time_ms: int
    total_cost_usd: float

class BaseEvaluator(ABC):
    """Base class for all evaluator models."""
    
    def __init__(self, provider: EvaluatorProvider, model_name: str):
        self.provider = provider
        self.model_name = model_name
        self.settings = get_settings()
        self._available = None
    
    @abstractmethod
    async def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        """Evaluate a single trace."""
        pass
    
    @abstractmethod
    async def is_available(self) -> bool:
        """Check if the evaluator is available."""
        pass
    
    @abstractmethod
    async def get_cost_estimate(self, request: EvaluationRequest) -> float:
        """Get cost estimate for evaluation."""
        pass
    
    def _extract_score_from_response(self, response_text: str) -> Tuple[float, str]:
        """Extract score and reasoning from model response."""
        try:
            # Try to find JSON response first
            import re
            json_match = re.search(r'\{[^}]*"score"[^}]*\}', response_text)
            if json_match:
                data = json.loads(json_match.group())
                score = float(data.get("score", 0.5))
                reasoning = data.get("reasoning", response_text)
                return max(0.0, min(1.0, score)), reasoning
            
            # Fallback: look for numerical score patterns
            score_patterns = [
                r'(?:score|rating)[:=]\s*([0-9]*\.?[0-9]+)',
                r'([0-9]*\.?[0-9]+)\s*(?:out of|/)\s*(?:10|100|1)',
                r'([0-9]*\.?[0-9]+)\s*(?:points|score)'
            ]
            
            for pattern in score_patterns:
                match = re.search(pattern, response_text, re.IGNORECASE)
                if match:
                    raw_score = float(match.group(1))
                    # Normalize to 0-1 range
                    if raw_score > 1:
                        if raw_score <= 10:
                            score = raw_score / 10.0
                        elif raw_score <= 100:
                            score = raw_score / 100.0
                        else:
                            score = 0.5  # fallback
                    else:
                        score = raw_score
                    
                    return max(0.0, min(1.0, score)), response_text
            
            # No numerical score found, return neutral
            return 0.5, response_text
            
        except Exception as e:
            logger.warning(f"Error extracting score from response: {e}")
            return 0.5, response_text
    
    def _build_template_prompt(self, request: EvaluationRequest, criteria: EvaluationCriteria) -> str:
        """Build evaluation prompt using template library."""
        if not TEMPLATES_AVAILABLE or not template_library:
            return self._build_fallback_prompt(request, criteria)
        
        try:
            # Get templates for the criteria
            templates = template_library.get_templates_by_criteria(criteria)
            if not templates:
                logger.warning(f"No templates found for criteria {criteria}, using fallback")
                return self._build_fallback_prompt(request, criteria)
            
            # Use the first (standard) template
            template = templates[0]
            
            # Prepare variables for template rendering
            variables = {
                "user_input": request.user_input,
                "model_output": request.model_output
            }
            
            # Add optional variables if provided
            if request.system_prompt:
                variables["system_prompt_section"] = request.system_prompt
            
            if request.reference_answer:
                variables["reference_answer_section"] = request.reference_answer
            
            # Render the template
            rendered_prompt = template_library.render_template(template.id, variables)
            return rendered_prompt
            
        except Exception as e:
            logger.warning(f"Error using template for {criteria}: {e}, falling back to manual prompt")
            return self._build_fallback_prompt(request, criteria)
    
    def _build_fallback_prompt(self, request: EvaluationRequest, criteria: EvaluationCriteria) -> str:
        """Build evaluation prompt manually when templates are not available."""
        criteria_descriptions = {
            EvaluationCriteria.COHERENCE: "logical flow and internal consistency",
            EvaluationCriteria.RELEVANCE: "how well the response addresses the user's query",
            EvaluationCriteria.FACTUAL_ACCURACY: "correctness and accuracy of factual claims",
            EvaluationCriteria.GRAMMAR: "grammar, spelling, and language mechanics",
            EvaluationCriteria.STYLE: "writing style, tone, and appropriateness",
            EvaluationCriteria.HELPFULNESS: "practical helpfulness and actionable value",
            EvaluationCriteria.HARMFULNESS: "absence of harmful or dangerous content",
            EvaluationCriteria.TRUTHFULNESS: "honesty and truthfulness",
            EvaluationCriteria.COMPLETENESS: "completeness and thoroughness"
        }
        
        description = criteria_descriptions.get(criteria, "overall quality")
        
        prompt = f"""
Please evaluate the following AI assistant response based on {description}.

User Query: {request.user_input}

AI Response: {request.model_output}
"""
        
        if request.system_prompt:
            prompt += f"\nSystem Prompt: {request.system_prompt}"
        
        if request.reference_answer:
            prompt += f"\nReference Answer: {request.reference_answer}"
        
        prompt += """

Please provide your evaluation as a JSON object with the following format:
{
    "score": 0.85,
    "reasoning": "Detailed explanation of the score..."
}

Score should be between 0.0 (poor) and 1.0 (excellent).
"""
        
        return prompt

class OpenAIEvaluator(BaseEvaluator):
    """OpenAI-based evaluator."""
    
    def __init__(self, model_name: str = "gpt-4"):
        super().__init__(EvaluatorProvider.OPENAI, model_name)
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize OpenAI client with backward compatibility."""
        try:
            from openai import AsyncOpenAI
            if hasattr(self.settings, 'openai_api_key') and self.settings.openai_api_key:
                self.client = AsyncOpenAI(api_key=self.settings.openai_api_key)
                self._available = True
            else:
                self._available = False
        except ImportError:
            try:
                import openai
                if hasattr(self.settings, 'openai_api_key') and self.settings.openai_api_key:
                    openai.api_key = self.settings.openai_api_key
                    self.client = openai
                    self._available = True
                else:
                    self._available = False
            except ImportError:
                self._available = False
                logger.warning("OpenAI package not available")
    
    async def is_available(self) -> bool:
        """Check if OpenAI evaluator is available."""
        return self._available and self.client is not None
    
    async def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        """Evaluate using OpenAI model."""
        if not await self.is_available():
            raise ValueError("OpenAI evaluator not available")
        
        start_time = time.time()
        
        try:
            prompt = self._build_template_prompt(request, request.criteria[0] if request.criteria else EvaluationCriteria.COHERENCE)
            
            if hasattr(self.client, 'chat') and hasattr(self.client.chat, 'completions'):
                # Modern AsyncOpenAI
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are an expert evaluator for AI system outputs."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=500
                )
                response_text = response.choices[0].message.content
                tokens_used = response.usage.total_tokens if response.usage else 1000
            else:
                # Legacy sync API
                def sync_call():
                    return self.client.ChatCompletion.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": "You are an expert evaluator for AI system outputs."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.1,
                        max_tokens=500
                    )
                
                response = await asyncio.get_event_loop().run_in_executor(None, sync_call)
                response_text = response.choices[0].message.content
                tokens_used = response.usage.total_tokens if response.usage else 1000
            
            # Extract score and reasoning
            score, reasoning = self._extract_score_from_response(response_text)
            
            # Calculate metrics
            evaluation_time = int((time.time() - start_time) * 1000)
            cost = self._estimate_cost(tokens_used)
            
            return EvaluationResult(
                trace_id=request.trace_id,
                evaluator_model=f"openai/{self.model_name}",
                criteria=request.criteria[0] if request.criteria else EvaluationCriteria.COHERENCE,
                score=score,
                reasoning=reasoning,
                confidence=0.8,  # OpenAI models generally have good confidence
                evaluation_time_ms=evaluation_time,
                cost_usd=cost,
                metadata={
                    "tokens_used": tokens_used,
                    "model": self.model_name,
                    "provider": "openai"
                }
            )
            
        except Exception as e:
            logger.error(f"OpenAI evaluation failed: {e}")
            raise
    
    def _estimate_cost(self, tokens: int) -> float:
        """Estimate cost based on token usage."""
        # Rough cost estimates for OpenAI models (as of 2024)
        cost_per_1k_tokens = {
            "gpt-4": 0.03,
            "gpt-4-turbo": 0.01,
            "gpt-3.5-turbo": 0.002,
        }
        
        rate = cost_per_1k_tokens.get(self.model_name, 0.01)
        return (tokens / 1000) * rate
    
    async def get_cost_estimate(self, request: EvaluationRequest) -> float:
        """Get cost estimate for evaluation."""
        # Estimate tokens based on input length
        text_length = len(request.user_input) + len(request.model_output)
        estimated_tokens = text_length // 3  # Rough estimate: 3 chars per token
        return self._estimate_cost(estimated_tokens)

class AnthropicEvaluator(BaseEvaluator):
    """Anthropic Claude-based evaluator."""
    
    def __init__(self, model_name: str = "claude-3-sonnet-20240229"):
        super().__init__(EvaluatorProvider.ANTHROPIC, model_name)
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Anthropic client."""
        try:
            import anthropic
            if hasattr(self.settings, 'anthropic_api_key') and self.settings.anthropic_api_key:
                self.client = anthropic.AsyncAnthropic(api_key=self.settings.anthropic_api_key)
                self._available = True
            else:
                self._available = False
        except ImportError:
            self._available = False
            logger.warning("Anthropic package not available")
    
    async def is_available(self) -> bool:
        """Check if Anthropic evaluator is available."""
        return self._available and self.client is not None
    
    async def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        """Evaluate using Anthropic Claude."""
        if not await self.is_available():
            raise ValueError("Anthropic evaluator not available")
        
        start_time = time.time()
        
        try:
            prompt = self._build_template_prompt(request, request.criteria[0] if request.criteria else EvaluationCriteria.COHERENCE)
            
            response = await self.client.messages.create(
                model=self.model_name,
                max_tokens=500,
                temperature=0.1,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            response_text = response.content[0].text
            tokens_used = response.usage.input_tokens + response.usage.output_tokens
            
            # Extract score and reasoning
            score, reasoning = self._extract_score_from_response(response_text)
            
            # Calculate metrics
            evaluation_time = int((time.time() - start_time) * 1000)
            cost = self._estimate_cost(tokens_used)
            
            return EvaluationResult(
                trace_id=request.trace_id,
                evaluator_model=f"anthropic/{self.model_name}",
                criteria=request.criteria[0] if request.criteria else EvaluationCriteria.COHERENCE,
                score=score,
                reasoning=reasoning,
                confidence=0.85,  # Claude models have high confidence
                evaluation_time_ms=evaluation_time,
                cost_usd=cost,
                metadata={
                    "tokens_used": tokens_used,
                    "model": self.model_name,
                    "provider": "anthropic"
                }
            )
            
        except Exception as e:
            logger.error(f"Anthropic evaluation failed: {e}")
            raise
    
    def _estimate_cost(self, tokens: int) -> float:
        """Estimate cost based on token usage."""
        # Rough cost estimates for Anthropic models (as of 2024)
        cost_per_1k_tokens = {
            "claude-3-sonnet-20240229": 0.015,
            "claude-3-haiku-20240307": 0.005,
            "claude-3-opus-20240229": 0.075,
        }
        
        rate = cost_per_1k_tokens.get(self.model_name, 0.015)
        return (tokens / 1000) * rate
    
    async def get_cost_estimate(self, request: EvaluationRequest) -> float:
        """Get cost estimate for evaluation."""
        text_length = len(request.user_input) + len(request.model_output)
        estimated_tokens = text_length // 3
        return self._estimate_cost(estimated_tokens)

class LocalEvaluator(BaseEvaluator):
    """Local model evaluator (placeholder for future implementation)."""
    
    def __init__(self, model_name: str = "local-model"):
        super().__init__(EvaluatorProvider.LOCAL, model_name)
        self._available = False  # Not implemented yet
    
    async def is_available(self) -> bool:
        """Local models not implemented yet."""
        return False
    
    async def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        """Placeholder for local model evaluation."""
        raise NotImplementedError("Local model evaluation not implemented yet")
    
    async def get_cost_estimate(self, request: EvaluationRequest) -> float:
        """Local models have no cost."""
        return 0.0

class EvaluatorManager:
    """Manager for coordinating multiple evaluator models."""
    
    def __init__(self):
        self.evaluators: Dict[str, BaseEvaluator] = {}
        self._initialize_evaluators()
    
    def _initialize_evaluators(self):
        """Initialize all available evaluators."""
        # Initialize OpenAI evaluators
        for model in ["gpt-4", "gpt-3.5-turbo"]:
            evaluator = OpenAIEvaluator(model)
            self.evaluators[f"openai/{model}"] = evaluator
        
        # Initialize Anthropic evaluators
        for model in ["claude-3-sonnet-20240229", "claude-3-haiku-20240307"]:
            evaluator = AnthropicEvaluator(model)
            self.evaluators[f"anthropic/{model}"] = evaluator
        
        # Initialize local evaluator (placeholder)
        local_evaluator = LocalEvaluator()
        self.evaluators["local/default"] = local_evaluator
    
    async def get_available_evaluators(self) -> List[str]:
        """Get list of available evaluator models."""
        available = []
        for name, evaluator in self.evaluators.items():
            if await evaluator.is_available():
                available.append(name)
        return available
    
    async def evaluate_single(
        self, 
        request: EvaluationRequest, 
        evaluator_name: Optional[str] = None
    ) -> EvaluationResult:
        """Evaluate a single request using the specified or best available evaluator."""
        if evaluator_name:
            if evaluator_name not in self.evaluators:
                raise ValueError(f"Evaluator {evaluator_name} not found")
            evaluator = self.evaluators[evaluator_name]
        else:
            # Use the best available evaluator
            available_evaluators = await self.get_available_evaluators()
            if not available_evaluators:
                raise ValueError("No evaluators available")
            
            # Prefer Claude Sonnet > GPT-4 > others
            if "anthropic/claude-3-sonnet-20240229" in available_evaluators:
                evaluator = self.evaluators["anthropic/claude-3-sonnet-20240229"]
            elif "openai/gpt-4" in available_evaluators:
                evaluator = self.evaluators["openai/gpt-4"]
            else:
                evaluator = self.evaluators[available_evaluators[0]]
        
        if not await evaluator.is_available():
            raise ValueError(f"Evaluator {evaluator.provider.value}/{evaluator.model_name} is not available")
        
        return await evaluator.evaluate(request)
    
    async def evaluate_single_with_calibration(
        self, 
        request: EvaluationRequest, 
        evaluator_name: Optional[str] = None,
        use_calibration: bool = True
    ) -> EvaluationResult:
        """Evaluate a single request with optional score calibration."""
        # Get the base evaluation result
        result = await self.evaluate_single(request, evaluator_name)
        
        # Apply calibration if available and requested
        if use_calibration and CALIBRATION_AVAILABLE and calibration_system:
            try:
                calibration_result = await calibration_system.calibrate_score(
                    ai_score=result.score,
                    criteria=result.criteria,
                    evaluator_model=result.evaluator_model,
                    confidence=result.confidence
                )
                
                # Update the result with calibrated score
                result.score = calibration_result.calibrated_score
                result.confidence = max(0.0, min(1.0, result.confidence + calibration_result.confidence_adjustment))
                
                # Add calibration metadata
                if not result.metadata:
                    result.metadata = {}
                result.metadata.update({
                    "calibration": {
                        "original_score": calibration_result.original_score,
                        "calibrated_score": calibration_result.calibrated_score,
                        "confidence_adjustment": calibration_result.confidence_adjustment,
                        "calibration_method": calibration_result.calibration_method.value,
                        "model_version": calibration_result.model_version,
                        "calibration_metadata": calibration_result.metadata
                    }
                })
                
            except Exception as e:
                logger.warning(f"Calibration failed, using original score: {e}")
                # Add error info to metadata
                if not result.metadata:
                    result.metadata = {}
                result.metadata["calibration_error"] = str(e)
        
        return result
    
    async def evaluate_multiple_criteria(
        self, 
        request: EvaluationRequest,
        criteria: List[EvaluationCriteria],
        evaluator_name: Optional[str] = None
    ) -> List[EvaluationResult]:
        """Evaluate a trace against multiple criteria."""
        results = []
        
        for criterion in criteria:
            criterion_request = EvaluationRequest(
                trace_id=request.trace_id,
                user_input=request.user_input,
                model_output=request.model_output,
                system_prompt=request.system_prompt,
                context=request.context,
                criteria=[criterion],
                custom_prompt=request.custom_prompt,
                reference_answer=request.reference_answer
            )
            
            try:
                result = await self.evaluate_single(criterion_request, evaluator_name)
                results.append(result)
            except Exception as e:
                logger.error(f"Error evaluating criterion {criterion}: {e}")
                # Continue with other criteria
        
        return results
    
    async def get_cost_estimates(self) -> Dict[str, Dict[str, float]]:
        """Get cost estimates for all evaluators."""
        estimates = {}
        
        # Sample request for estimation
        sample_request = EvaluationRequest(
            trace_id="sample",
            user_input="What is the capital of France?",
            model_output="The capital of France is Paris, which is located in the north-central part of the country.",
            criteria=[EvaluationCriteria.COHERENCE]
        )
        
        for name, evaluator in self.evaluators.items():
            if await evaluator.is_available():
                try:
                    cost = await evaluator.get_cost_estimate(sample_request)
                    estimates[name] = {
                        "estimated_cost_per_evaluation": cost,
                        "provider": evaluator.provider.value,
                        "model": evaluator.model_name
                    }
                except Exception as e:
                    logger.warning(f"Could not get cost estimate for {name}: {e}")
        
        return estimates

# Global evaluator manager instance
evaluator_manager = EvaluatorManager() 