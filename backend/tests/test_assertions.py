"""
Tests for the assertion framework.
"""

import pytest
import asyncio
from backend.testing.assertions import (
    create_assertion,
    AssertionResult,
    ContainsAssertion,
    NotContainsAssertion,
    RegexAssertion,
    SentimentAssertion,
    JSONSchemaAssertion,
    LengthAssertion,
    CustomFunctionAssertion,
    ASSERTION_REGISTRY
)


class TestContainsAssertion:
    """Test the ContainsAssertion class."""
    
    @pytest.mark.asyncio
    async def test_contains_case_sensitive(self):
        """Test case-sensitive contains assertion."""
        assertion = ContainsAssertion("test", {"case_sensitive": True})
        
        # Should pass
        result = await assertion.evaluate("Hello World", "Hello")
        assert result.result == AssertionResult.PASSED
        
        # Should fail
        result = await assertion.evaluate("Hello World", "hello")
        assert result.result == AssertionResult.FAILED
    
    @pytest.mark.asyncio
    async def test_contains_case_insensitive(self):
        """Test case-insensitive contains assertion."""
        assertion = ContainsAssertion("test", {"case_sensitive": False})
        
        # Should pass
        result = await assertion.evaluate("Hello World", "hello")
        assert result.result == AssertionResult.PASSED
        
        result = await assertion.evaluate("Hello World", "WORLD")
        assert result.result == AssertionResult.PASSED


class TestNotContainsAssertion:
    """Test the NotContainsAssertion class."""
    
    @pytest.mark.asyncio
    async def test_not_contains_success(self):
        """Test successful not contains assertion."""
        assertion = NotContainsAssertion("test", {"case_sensitive": False})
        
        result = await assertion.evaluate("Hello World", "goodbye")
        assert result.result == AssertionResult.PASSED
    
    @pytest.mark.asyncio
    async def test_not_contains_failure(self):
        """Test failed not contains assertion."""
        assertion = NotContainsAssertion("test", {"case_sensitive": False})
        
        result = await assertion.evaluate("Hello World", "hello")
        assert result.result == AssertionResult.FAILED


class TestRegexAssertion:
    """Test the RegexAssertion class."""
    
    @pytest.mark.asyncio
    async def test_regex_match(self):
        """Test successful regex match."""
        assertion = RegexAssertion("test", {"pattern": r"\d{3}-\d{2}-\d{4}"})
        
        result = await assertion.evaluate("SSN: 123-45-6789", r"\d{3}-\d{2}-\d{4}")
        assert result.result == AssertionResult.PASSED
        assert result.details["match_groups"] == ()
    
    @pytest.mark.asyncio
    async def test_regex_no_match(self):
        """Test failed regex match."""
        assertion = RegexAssertion("test", {"pattern": r"\d{3}-\d{2}-\d{4}"})
        
        result = await assertion.evaluate("No SSN here", r"\d{3}-\d{2}-\d{4}")
        assert result.result == AssertionResult.FAILED


class TestSentimentAssertion:
    """Test the SentimentAssertion class."""
    
    @pytest.mark.asyncio
    async def test_positive_sentiment(self):
        """Test positive sentiment detection."""
        assertion = SentimentAssertion("test", {"sentiment": "positive", "threshold": 0.1})
        
        result = await assertion.evaluate("I love this! It's amazing and wonderful!", "positive")
        assert result.result == AssertionResult.PASSED
        assert result.details["polarity_score"] > 0.1
    
    @pytest.mark.asyncio
    async def test_negative_sentiment(self):
        """Test negative sentiment detection."""
        assertion = SentimentAssertion("test", {"sentiment": "negative", "threshold": 0.1})
        
        result = await assertion.evaluate("I hate this! It's terrible and awful!", "negative")
        assert result.result == AssertionResult.PASSED
        assert result.details["polarity_score"] < -0.1
    
    @pytest.mark.asyncio
    async def test_neutral_sentiment(self):
        """Test neutral sentiment detection."""
        assertion = SentimentAssertion("test", {"sentiment": "neutral", "threshold": 0.1})
        
        result = await assertion.evaluate("This is a factual statement.", "neutral")
        assert result.result == AssertionResult.PASSED
        assert abs(result.details["polarity_score"]) <= 0.1


class TestJSONSchemaAssertion:
    """Test the JSONSchemaAssertion class."""
    
    @pytest.mark.asyncio
    async def test_valid_json_schema(self):
        """Test valid JSON schema validation."""
        schema = {
            "type": "object",
            "required": ["name", "age"],
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"}
            }
        }
        assertion = JSONSchemaAssertion("test", {"schema": schema})
        
        json_data = '{"name": "John", "age": 30}'
        result = await assertion.evaluate(json_data, schema)
        assert result.result == AssertionResult.PASSED
    
    @pytest.mark.asyncio
    async def test_invalid_json_schema(self):
        """Test invalid JSON schema validation."""
        schema = {
            "type": "object",
            "required": ["name", "age"],
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"}
            }
        }
        assertion = JSONSchemaAssertion("test", {"schema": schema})
        
        json_data = '{"name": "John"}'  # Missing required "age"
        result = await assertion.evaluate(json_data, schema)
        assert result.result == AssertionResult.FAILED
    
    @pytest.mark.asyncio
    async def test_invalid_json_format(self):
        """Test invalid JSON format."""
        schema = {"type": "object"}
        assertion = JSONSchemaAssertion("test", {"schema": schema})
        
        invalid_json = '{"name": "John", "age":}'  # Invalid JSON
        result = await assertion.evaluate(invalid_json, schema)
        assert result.result == AssertionResult.FAILED
        assert "not valid JSON" in result.message


class TestLengthAssertion:
    """Test the LengthAssertion class."""
    
    @pytest.mark.asyncio
    async def test_character_length_exact(self):
        """Test exact character length."""
        assertion = LengthAssertion("test", {"type": "characters", "exact_length": 11})
        
        result = await assertion.evaluate("Hello World", 11)
        assert result.result == AssertionResult.PASSED
        assert result.actual == 11
    
    @pytest.mark.asyncio
    async def test_word_length_range(self):
        """Test word length within range."""
        assertion = LengthAssertion("test", {
            "type": "words",
            "min_length": 2,
            "max_length": 5
        })
        
        result = await assertion.evaluate("Hello beautiful world")  # 3 words
        assert result.result == AssertionResult.PASSED
        assert result.actual == 3
    
    @pytest.mark.asyncio
    async def test_length_too_short(self):
        """Test length too short."""
        assertion = LengthAssertion("test", {
            "type": "words",
            "min_length": 5
        })
        
        result = await assertion.evaluate("Hello world")  # 2 words
        assert result.result == AssertionResult.FAILED
        assert "too short" in result.message
    
    @pytest.mark.asyncio
    async def test_length_too_long(self):
        """Test length too long."""
        assertion = LengthAssertion("test", {
            "type": "words",
            "max_length": 2
        })
        
        result = await assertion.evaluate("Hello beautiful wonderful world")  # 4 words
        assert result.result == AssertionResult.FAILED
        assert "too long" in result.message


class TestCustomFunctionAssertion:
    """Test the CustomFunctionAssertion class."""
    
    @pytest.mark.asyncio
    async def test_custom_function_boolean_return(self):
        """Test custom function returning boolean."""
        function_code = '''
def check_length(output, expected, context):
    return len(output) > 5
'''
        assertion = CustomFunctionAssertion("test", {
            "function_code": function_code,
            "function_name": "check_length"
        })
        
        result = await assertion.evaluate("Hello World!")
        assert result.result == AssertionResult.PASSED
        
        result = await assertion.evaluate("Hi")
        assert result.result == AssertionResult.FAILED
    
    @pytest.mark.asyncio
    async def test_custom_function_dict_return(self):
        """Test custom function returning detailed dict."""
        function_code = '''
def validate_email(output, expected, context):
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if re.match(pattern, output.strip()):
        return {
            "passed": True,
            "message": "Valid email format",
            "expected": "Email format",
            "actual": output.strip()
        }
    else:
        return {
            "passed": False,
            "message": "Invalid email format",
            "expected": "Email format",
            "actual": output.strip()
        }
'''
        assertion = CustomFunctionAssertion("test", {
            "function_code": function_code,
            "function_name": "validate_email"
        })
        
        result = await assertion.evaluate("user@example.com")
        assert result.result == AssertionResult.PASSED
        assert "Valid email format" in result.message
        
        result = await assertion.evaluate("invalid-email")
        assert result.result == AssertionResult.FAILED
        assert "Invalid email format" in result.message
    
    @pytest.mark.asyncio
    async def test_custom_function_error(self):
        """Test custom function with runtime error."""
        function_code = '''
def broken_function(output, expected, context):
    raise ValueError("This function is broken")
'''
        assertion = CustomFunctionAssertion("test", {
            "function_code": function_code,
            "function_name": "broken_function"
        })
        
        result = await assertion.evaluate("test input")
        assert result.result == AssertionResult.ERROR
        assert "This function is broken" in result.message


class TestAssertionFactory:
    """Test the assertion factory function."""
    
    def test_create_assertion_valid_types(self):
        """Test creating assertions for all valid types."""
        for assertion_type in ASSERTION_REGISTRY.keys():
            assertion = create_assertion(assertion_type, "test", {})
            assert assertion is not None
            assert assertion.name == "test"
    
    def test_create_assertion_invalid_type(self):
        """Test creating assertion with invalid type."""
        with pytest.raises(ValueError, match="Unknown assertion type"):
            create_assertion("invalid_type", "test", {})
    
    def test_assertion_registry_completeness(self):
        """Test that assertion registry contains expected types."""
        expected_types = {
            "contains", "not_contains", "regex", "sentiment", 
            "json_schema", "length", "custom_function"
        }
        assert set(ASSERTION_REGISTRY.keys()) == expected_types


class TestAsyncBehavior:
    """Test async behavior and concurrency."""
    
    @pytest.mark.asyncio
    async def test_concurrent_assertions(self):
        """Test running multiple assertions concurrently."""
        assertions = []
        for i in range(10):
            assertion = create_assertion("contains", f"test_{i}", {"case_sensitive": False})
            assertions.append(assertion)
        
        # Run all assertions concurrently
        tasks = []
        for assertion in assertions:
            task = assertion.evaluate("Hello World", "hello")
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # All should pass
        assert len(results) == 10
        for result in results:
            assert result.result == AssertionResult.PASSED
            assert result.execution_time_ms is not None
            assert result.execution_time_ms >= 0 