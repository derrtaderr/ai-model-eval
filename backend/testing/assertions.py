"""
Assertion engine for LLM output validation.
Supports multiple assertion types for comprehensive testing.
"""

import re
import json
import asyncio
from typing import Any, Dict, List, Optional, Union, Callable
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

import jsonschema
from textblob import TextBlob
import httpx


class AssertionResult(Enum):
    """Result of an assertion."""
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"


@dataclass
class AssertionOutcome:
    """Outcome of running an assertion."""
    result: AssertionResult
    message: str
    expected: Any = None
    actual: Any = None
    details: Dict[str, Any] = None
    execution_time_ms: Optional[int] = None


class BaseAssertion(ABC):
    """Base class for all assertion types."""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
    
    @abstractmethod
    async def evaluate(self, model_output: str, expected: Any = None, context: Dict[str, Any] = None) -> AssertionOutcome:
        """Evaluate the assertion against the model output."""
        pass


class ContainsAssertion(BaseAssertion):
    """Assert that output contains specific text."""
    
    async def evaluate(self, model_output: str, expected: Any = None, context: Dict[str, Any] = None) -> AssertionOutcome:
        start_time = datetime.now()
        
        try:
            text_to_find = expected or self.config.get("text", "")
            case_sensitive = self.config.get("case_sensitive", False)
            
            if not case_sensitive:
                contains = text_to_find.lower() in model_output.lower()
            else:
                contains = text_to_find in model_output
            
            execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
            
            if contains:
                return AssertionOutcome(
                    result=AssertionResult.PASSED,
                    message=f"Output contains expected text: '{text_to_find}'",
                    expected=text_to_find,
                    actual=model_output,
                    execution_time_ms=execution_time
                )
            else:
                return AssertionOutcome(
                    result=AssertionResult.FAILED,
                    message=f"Output does not contain expected text: '{text_to_find}'",
                    expected=text_to_find,
                    actual=model_output,
                    execution_time_ms=execution_time
                )
        
        except Exception as e:
            execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
            return AssertionOutcome(
                result=AssertionResult.ERROR,
                message=f"Error in contains assertion: {str(e)}",
                execution_time_ms=execution_time
            )


class NotContainsAssertion(BaseAssertion):
    """Assert that output does not contain specific text."""
    
    async def evaluate(self, model_output: str, expected: Any = None, context: Dict[str, Any] = None) -> AssertionOutcome:
        start_time = datetime.now()
        
        try:
            text_to_avoid = expected or self.config.get("text", "")
            case_sensitive = self.config.get("case_sensitive", False)
            
            if not case_sensitive:
                contains = text_to_avoid.lower() in model_output.lower()
            else:
                contains = text_to_avoid in model_output
            
            execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
            
            if not contains:
                return AssertionOutcome(
                    result=AssertionResult.PASSED,
                    message=f"Output correctly does not contain: '{text_to_avoid}'",
                    expected=f"Not containing: {text_to_avoid}",
                    actual=model_output,
                    execution_time_ms=execution_time
                )
            else:
                return AssertionOutcome(
                    result=AssertionResult.FAILED,
                    message=f"Output unexpectedly contains: '{text_to_avoid}'",
                    expected=f"Not containing: {text_to_avoid}",
                    actual=model_output,
                    execution_time_ms=execution_time
                )
        
        except Exception as e:
            execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
            return AssertionOutcome(
                result=AssertionResult.ERROR,
                message=f"Error in not-contains assertion: {str(e)}",
                execution_time_ms=execution_time
            )


class RegexAssertion(BaseAssertion):
    """Assert that output matches a regular expression."""
    
    async def evaluate(self, model_output: str, expected: Any = None, context: Dict[str, Any] = None) -> AssertionOutcome:
        start_time = datetime.now()
        
        try:
            pattern = expected or self.config.get("pattern", "")
            flags = self.config.get("flags", 0)
            
            match = re.search(pattern, model_output, flags)
            execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
            
            if match:
                return AssertionOutcome(
                    result=AssertionResult.PASSED,
                    message=f"Output matches regex pattern: '{pattern}'",
                    expected=pattern,
                    actual=model_output,
                    details={"match_groups": match.groups(), "match_span": match.span()},
                    execution_time_ms=execution_time
                )
            else:
                return AssertionOutcome(
                    result=AssertionResult.FAILED,
                    message=f"Output does not match regex pattern: '{pattern}'",
                    expected=pattern,
                    actual=model_output,
                    execution_time_ms=execution_time
                )
        
        except Exception as e:
            execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
            return AssertionOutcome(
                result=AssertionResult.ERROR,
                message=f"Error in regex assertion: {str(e)}",
                execution_time_ms=execution_time
            )


class SentimentAssertion(BaseAssertion):
    """Assert sentiment of the output (positive, negative, neutral)."""
    
    async def evaluate(self, model_output: str, expected: Any = None, context: Dict[str, Any] = None) -> AssertionOutcome:
        start_time = datetime.now()
        
        try:
            expected_sentiment = expected or self.config.get("sentiment", "positive")
            threshold = self.config.get("threshold", 0.1)
            
            # Use TextBlob for sentiment analysis
            blob = TextBlob(model_output)
            polarity = blob.sentiment.polarity
            
            # Determine actual sentiment
            if polarity > threshold:
                actual_sentiment = "positive"
            elif polarity < -threshold:
                actual_sentiment = "negative"
            else:
                actual_sentiment = "neutral"
            
            execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
            
            if actual_sentiment == expected_sentiment:
                return AssertionOutcome(
                    result=AssertionResult.PASSED,
                    message=f"Sentiment is {actual_sentiment} as expected",
                    expected=expected_sentiment,
                    actual=actual_sentiment,
                    details={"polarity_score": polarity, "threshold": threshold},
                    execution_time_ms=execution_time
                )
            else:
                return AssertionOutcome(
                    result=AssertionResult.FAILED,
                    message=f"Expected sentiment {expected_sentiment}, got {actual_sentiment}",
                    expected=expected_sentiment,
                    actual=actual_sentiment,
                    details={"polarity_score": polarity, "threshold": threshold},
                    execution_time_ms=execution_time
                )
        
        except Exception as e:
            execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
            return AssertionOutcome(
                result=AssertionResult.ERROR,
                message=f"Error in sentiment assertion: {str(e)}",
                execution_time_ms=execution_time
            )


class JSONSchemaAssertion(BaseAssertion):
    """Assert that output is valid JSON matching a schema."""
    
    async def evaluate(self, model_output: str, expected: Any = None, context: Dict[str, Any] = None) -> AssertionOutcome:
        start_time = datetime.now()
        
        try:
            schema = expected or self.config.get("schema", {})
            
            # Try to parse JSON
            try:
                parsed_json = json.loads(model_output)
            except json.JSONDecodeError as e:
                execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
                return AssertionOutcome(
                    result=AssertionResult.FAILED,
                    message=f"Output is not valid JSON: {str(e)}",
                    expected="Valid JSON",
                    actual=model_output,
                    execution_time_ms=execution_time
                )
            
            # Validate against schema
            try:
                jsonschema.validate(instance=parsed_json, schema=schema)
                execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
                return AssertionOutcome(
                    result=AssertionResult.PASSED,
                    message="JSON output matches schema",
                    expected="Valid JSON matching schema",
                    actual=parsed_json,
                    execution_time_ms=execution_time
                )
            except jsonschema.ValidationError as e:
                execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
                return AssertionOutcome(
                    result=AssertionResult.FAILED,
                    message=f"JSON does not match schema: {str(e)}",
                    expected="JSON matching schema",
                    actual=parsed_json,
                    details={"validation_error": str(e)},
                    execution_time_ms=execution_time
                )
        
        except Exception as e:
            execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
            return AssertionOutcome(
                result=AssertionResult.ERROR,
                message=f"Error in JSON schema assertion: {str(e)}",
                execution_time_ms=execution_time
            )


class LengthAssertion(BaseAssertion):
    """Assert output length (character count, word count, etc.)."""
    
    async def evaluate(self, model_output: str, expected: Any = None, context: Dict[str, Any] = None) -> AssertionOutcome:
        start_time = datetime.now()
        
        try:
            length_type = self.config.get("type", "characters")  # characters, words, lines
            min_length = self.config.get("min_length")
            max_length = self.config.get("max_length")
            exact_length = expected or self.config.get("exact_length")
            
            # Calculate actual length
            if length_type == "characters":
                actual_length = len(model_output)
            elif length_type == "words":
                actual_length = len(model_output.split())
            elif length_type == "lines":
                actual_length = len(model_output.split('\n'))
            else:
                raise ValueError(f"Unsupported length type: {length_type}")
            
            execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
            
            # Check constraints
            if exact_length is not None:
                if actual_length == exact_length:
                    return AssertionOutcome(
                        result=AssertionResult.PASSED,
                        message=f"Output has exact length of {actual_length} {length_type}",
                        expected=exact_length,
                        actual=actual_length,
                        execution_time_ms=execution_time
                    )
                else:
                    return AssertionOutcome(
                        result=AssertionResult.FAILED,
                        message=f"Expected {exact_length} {length_type}, got {actual_length}",
                        expected=exact_length,
                        actual=actual_length,
                        execution_time_ms=execution_time
                    )
            
            # Check min/max constraints
            if min_length is not None and actual_length < min_length:
                return AssertionOutcome(
                    result=AssertionResult.FAILED,
                    message=f"Output too short: {actual_length} {length_type} (min: {min_length})",
                    expected=f"At least {min_length} {length_type}",
                    actual=actual_length,
                    execution_time_ms=execution_time
                )
            
            if max_length is not None and actual_length > max_length:
                return AssertionOutcome(
                    result=AssertionResult.FAILED,
                    message=f"Output too long: {actual_length} {length_type} (max: {max_length})",
                    expected=f"At most {max_length} {length_type}",
                    actual=actual_length,
                    execution_time_ms=execution_time
                )
            
            return AssertionOutcome(
                result=AssertionResult.PASSED,
                message=f"Output length ({actual_length} {length_type}) meets constraints",
                expected="Length within bounds",
                actual=actual_length,
                execution_time_ms=execution_time
            )
        
        except Exception as e:
            execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
            return AssertionOutcome(
                result=AssertionResult.ERROR,
                message=f"Error in length assertion: {str(e)}",
                execution_time_ms=execution_time
            )


class CustomFunctionAssertion(BaseAssertion):
    """Assert using a custom evaluation function."""
    
    async def evaluate(self, model_output: str, expected: Any = None, context: Dict[str, Any] = None) -> AssertionOutcome:
        start_time = datetime.now()
        
        try:
            function_code = self.config.get("function_code", "")
            function_name = self.config.get("function_name", "evaluate_output")
            
            if not function_code:
                raise ValueError("No function code provided for custom assertion")
            
            # Create a safe execution environment
            safe_globals = {
                "__builtins__": {},
                "len": len,
                "str": str,
                "int": int,
                "float": float,
                "bool": bool,
                "list": list,
                "dict": dict,
                "re": re,
                "json": json,
            }
            
            local_vars = {}
            
            # Execute the function definition
            exec(function_code, safe_globals, local_vars)
            
            if function_name not in local_vars:
                raise ValueError(f"Function '{function_name}' not found in code")
            
            # Call the function
            evaluation_function = local_vars[function_name]
            result = evaluation_function(model_output, expected, context or {})
            
            execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
            
            # Handle different return types
            if isinstance(result, bool):
                if result:
                    return AssertionOutcome(
                        result=AssertionResult.PASSED,
                        message="Custom function evaluation passed",
                        expected="Custom evaluation: True",
                        actual=result,
                        execution_time_ms=execution_time
                    )
                else:
                    return AssertionOutcome(
                        result=AssertionResult.FAILED,
                        message="Custom function evaluation failed",
                        expected="Custom evaluation: True",
                        actual=result,
                        execution_time_ms=execution_time
                    )
            elif isinstance(result, dict) and "passed" in result:
                return AssertionOutcome(
                    result=AssertionResult.PASSED if result["passed"] else AssertionResult.FAILED,
                    message=result.get("message", "Custom function evaluation completed"),
                    expected=result.get("expected"),
                    actual=result.get("actual"),
                    details=result.get("details"),
                    execution_time_ms=execution_time
                )
            else:
                raise ValueError("Custom function must return bool or dict with 'passed' key")
        
        except Exception as e:
            execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
            return AssertionOutcome(
                result=AssertionResult.ERROR,
                message=f"Error in custom function assertion: {str(e)}",
                execution_time_ms=execution_time
            )


# Registry of available assertion types
ASSERTION_REGISTRY = {
    "contains": ContainsAssertion,
    "not_contains": NotContainsAssertion,
    "regex": RegexAssertion,
    "sentiment": SentimentAssertion,
    "json_schema": JSONSchemaAssertion,
    "length": LengthAssertion,
    "custom_function": CustomFunctionAssertion,
}


def create_assertion(assertion_type: str, name: str, config: Dict[str, Any] = None) -> BaseAssertion:
    """Factory function to create assertion instances."""
    if assertion_type not in ASSERTION_REGISTRY:
        raise ValueError(f"Unknown assertion type: {assertion_type}")
    
    assertion_class = ASSERTION_REGISTRY[assertion_type]
    return assertion_class(name=name, config=config) 