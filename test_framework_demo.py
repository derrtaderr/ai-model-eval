#!/usr/bin/env python3
"""
Demo script for the LLM Evaluation Platform Testing Framework.
Run this to see the testing system in action.
"""

import asyncio
import sys
import os

# Add backend to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from backend.testing.assertions import (
    create_assertion, 
    AssertionResult, 
    ASSERTION_REGISTRY
)


async def demo_assertions():
    """Demonstrate different assertion types."""
    print("🧪 LLM Evaluation Platform - Testing Framework Demo")
    print("=" * 60)
    
    # Demo test outputs
    test_outputs = {
        "greeting": "Hello! I'm doing great today, thank you for asking. How can I help you?",
        "email": "Subject: Meeting Follow-up\n\nDear John,\n\nThank you for the productive meeting today.\n\nBest regards,\nAI Assistant",
        "json": '{"name": "John Doe", "email": "john@example.com", "id": "user123", "created_at": "2025-01-27"}',
        "positive": "I'm so happy to help you today! This is going to be wonderful.",
        "negative": "I'm really sorry to hear about your problem. This situation is quite frustrating.",
        "code": """```python
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
```""",
        "discount": "As a premium customer, you're eligible for a 15% discount on your order!"
    }
    
    # Test 1: Contains Assertion
    print("\n1. 📝 Contains Assertion")
    print("   Testing if greeting contains 'Hello'")
    contains_assertion = create_assertion("contains", "greeting_test", {"case_sensitive": False})
    result = await contains_assertion.evaluate(test_outputs["greeting"], "Hello")
    print(f"   Result: {result.result.value} - {result.message}")
    
    # Test 2: Sentiment Analysis
    print("\n2. 😊 Sentiment Assertion")
    print("   Testing positive sentiment")
    sentiment_assertion = create_assertion("sentiment", "positive_test", {"threshold": 0.1})
    result = await sentiment_assertion.evaluate(test_outputs["positive"], "positive")
    print(f"   Result: {result.result.value} - {result.message}")
    print(f"   Details: Polarity score = {result.details['polarity_score']:.3f}")
    
    # Test 3: JSON Schema Validation
    print("\n3. 📋 JSON Schema Assertion")
    print("   Testing JSON schema compliance")
    schema = {
        "type": "object",
        "required": ["name", "email", "id"],
        "properties": {
            "name": {"type": "string"},
            "email": {"type": "string"},
            "id": {"type": "string"}
        }
    }
    json_assertion = create_assertion("json_schema", "json_test", {"schema": schema})
    result = await json_assertion.evaluate(test_outputs["json"])
    print(f"   Result: {result.result.value} - {result.message}")
    
    # Test 4: Regular Expression
    print("\n4. 🔍 Regex Assertion")
    print("   Testing email format pattern")
    regex_assertion = create_assertion("regex", "email_test", {"pattern": r"Subject:.*Best regards"})
    result = await regex_assertion.evaluate(test_outputs["email"])
    print(f"   Result: {result.result.value} - {result.message}")
    
    # Test 5: Length Validation
    print("\n5. 📏 Length Assertion")
    print("   Testing response length (word count)")
    length_assertion = create_assertion("length", "length_test", {
        "type": "words",
        "min_length": 5,
        "max_length": 20
    })
    result = await length_assertion.evaluate(test_outputs["greeting"])
    print(f"   Result: {result.result.value} - {result.message}")
    print(f"   Word count: {result.actual}")
    
    # Test 6: Custom Function
    print("\n6. ⚙️  Custom Function Assertion")
    print("   Testing business logic (discount validation)")
    
    custom_code = '''
def validate_discount(output, expected, context):
    """Validate that premium customers get appropriate discount."""
    import re
    
    # Look for percentage in the output
    percentage_match = re.search(r'(\\d+(?:\\.\\d+)?)%', output)
    if not percentage_match:
        return {
            "passed": False,
            "message": "No discount percentage found in output",
            "expected": "Discount percentage",
            "actual": output
        }
    
    discount = float(percentage_match.group(1))
    
    # Premium customers should get at least 10% discount
    if discount >= 10:
        return {
            "passed": True,
            "message": f"Premium discount of {discount}% is appropriate",
            "expected": ">=10% discount",
            "actual": f"{discount}% discount"
        }
    else:
        return {
            "passed": False,
            "message": f"Premium discount of {discount}% is too low",
            "expected": ">=10% discount",
            "actual": f"{discount}% discount"
        }
'''
    
    custom_assertion = create_assertion("custom_function", "discount_test", {
        "function_code": custom_code,
        "function_name": "validate_discount"
    })
    result = await custom_assertion.evaluate(test_outputs["discount"])
    print(f"   Result: {result.result.value} - {result.message}")
    
    # Test 7: Not Contains (Safety Check)
    print("\n7. 🛡️  Not Contains Assertion (Safety)")
    print("   Testing that positive response doesn't contain negative words")
    not_contains_assertion = create_assertion("not_contains", "safety_test", {"case_sensitive": False})
    result = await not_contains_assertion.evaluate(test_outputs["positive"], "terrible")
    print(f"   Result: {result.result.value} - {result.message}")
    
    print("\n" + "=" * 60)
    print("🎉 Demo completed! All assertion types demonstrated.")
    print(f"📊 Available assertion types: {list(ASSERTION_REGISTRY.keys())}")
    print("\n💡 Next steps:")
    print("   • Start your FastAPI server: cd backend && python main.py")
    print("   • Visit http://localhost:8000/docs for API documentation")
    print("   • Create test cases via API: POST /api/test-cases")
    print("   • Run tests against traces: POST /api/test-runs")


async def demo_performance():
    """Demonstrate performance characteristics."""
    print("\n⚡ Performance Demo")
    print("-" * 30)
    
    import time
    
    # Create multiple assertions
    assertions = []
    for i in range(100):
        assertion = create_assertion("contains", f"perf_test_{i}", {"case_sensitive": False})
        assertions.append(assertion)
    
    # Test concurrent execution
    test_text = "This is a performance test for the LLM evaluation platform testing framework."
    
    start_time = time.time()
    
    # Run assertions concurrently
    tasks = []
    for assertion in assertions[:10]:  # Test with 10 assertions
        task = assertion.evaluate(test_text, "test")
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    
    end_time = time.time()
    execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
    
    passed_count = sum(1 for r in results if r.result == AssertionResult.PASSED)
    
    print(f"   🏃 Executed 10 assertions concurrently")
    print(f"   ⏱️  Total time: {execution_time:.2f}ms")
    print(f"   📈 Average per assertion: {execution_time/10:.2f}ms")
    print(f"   ✅ Passed: {passed_count}/10")


if __name__ == "__main__":
    print("Starting LLM Testing Framework Demo...")
    asyncio.run(demo_assertions())
    asyncio.run(demo_performance()) 