"""
Custom Filter Configuration Examples
Customize the evaluation platform for different use cases and industries.
"""

# Example 1: E-commerce Customer Support
ECOMMERCE_FILTERS = {
    "tools": [
        "Order-Management",
        "Product-Recommendations", 
        "Customer-Support",
        "Returns-Assistant",
        "Shipping-Tracker"
    ],
    "scenarios": [
        "Order-Status-Inquiry",
        "Product-Questions", 
        "Return-Request",
        "Complaint-Resolution",
        "Payment-Issues",
        "Shipping-Problems"
    ],
    "custom_statuses": [
        "needs-escalation",
        "customer-satisfied",
        "requires-followup"
    ]
}

# Example 2: Healthcare AI Assistant
HEALTHCARE_FILTERS = {
    "tools": [
        "Symptom-Checker",
        "Medication-Info",
        "Appointment-Scheduler",
        "Health-Educator",
        "Emergency-Triage"
    ],
    "scenarios": [
        "Symptom-Assessment",
        "Drug-Interactions",
        "Appointment-Booking",
        "Health-Questions",
        "Emergency-Situations",
        "Wellness-Tips"
    ],
    "custom_metadata": [
        "medical_accuracy_score",
        "safety_compliance",
        "patient_satisfaction"
    ]
}

# Example 3: Financial Services
FINTECH_FILTERS = {
    "tools": [
        "Account-Management",
        "Investment-Advisor", 
        "Fraud-Detection",
        "Loan-Assistant",
        "Trading-Bot"
    ],
    "scenarios": [
        "Account-Balance",
        "Investment-Advice",
        "Fraud-Alert",
        "Loan-Application",
        "Market-Analysis",
        "Risk-Assessment"
    ],
    "compliance_tags": [
        "regulatory_compliant",
        "risk_warning_included",
        "accuracy_verified"
    ]
}

# Example 4: Education Platform
EDUCATION_FILTERS = {
    "tools": [
        "Tutor-Assistant",
        "Essay-Reviewer",
        "Quiz-Generator", 
        "Study-Planner",
        "Code-Mentor"
    ],
    "scenarios": [
        "Math-Help",
        "Writing-Assistance",
        "Programming-Help",
        "Study-Planning",
        "Exam-Prep",
        "Concept-Explanation"
    ],
    "educational_metrics": [
        "learning_effectiveness",
        "age_appropriateness", 
        "curriculum_alignment"
    ]
}

def generate_trace_with_custom_filters(
    user_input: str,
    ai_response: str,
    filter_config: dict,
    tool: str,
    scenario: str,
    **kwargs
) -> dict:
    """Generate a trace with custom filter categories."""
    
    # Validate that tool and scenario are in the allowed filters
    if tool not in filter_config["tools"]:
        raise ValueError(f"Tool '{tool}' not in allowed tools: {filter_config['tools']}")
    
    if scenario not in filter_config["scenarios"]:
        raise ValueError(f"Scenario '{scenario}' not in allowed scenarios: {filter_config['scenarios']}")
    
    trace = {
        "id": f"trace-{int(time.time())}",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tool": tool,
        "scenario": scenario,
        "status": "pending",
        "modelScore": "pass",
        "humanScore": None,
        "dataSource": "human",
        "conversation": {
            "userInput": user_input,
            "aiResponse": ai_response,
            "systemPrompt": kwargs.get("system_prompt", "")
        },
        "functions": kwargs.get("functions", []),
        "metadata": {
            "modelName": kwargs.get("model_name", "gpt-4"),
            "latencyMs": kwargs.get("latency_ms", 1000),
            "tokenCount": kwargs.get("token_count", {"input": 50, "output": 100}),
            "costUsd": kwargs.get("cost_usd", 0.005),
            "temperature": kwargs.get("temperature", 0.7),
            "maxTokens": kwargs.get("max_tokens", 500)
        }
    }
    
    # Add custom metadata fields
    if "custom_metadata" in filter_config:
        for field in filter_config["custom_metadata"]:
            trace["metadata"][field] = kwargs.get(field, 0.0)
    
    if "compliance_tags" in filter_config:
        trace["compliance"] = {}
        for tag in filter_config["compliance_tags"]:
            trace["compliance"][tag] = kwargs.get(tag, False)
    
    if "educational_metrics" in filter_config:
        trace["educational"] = {}
        for metric in filter_config["educational_metrics"]:
            trace["educational"][metric] = kwargs.get(metric, 0.0)
    
    return trace

# Usage Examples
import time

def create_ecommerce_traces():
    """Create sample traces for e-commerce use case."""
    traces = []
    
    # Order status inquiry
    traces.append(generate_trace_with_custom_filters(
        user_input="Where is my order #12345?",
        ai_response="Your order #12345 is currently in transit and will arrive tomorrow by 6 PM.",
        filter_config=ECOMMERCE_FILTERS,
        tool="Order-Management",
        scenario="Order-Status-Inquiry",
        model_name="gpt-4-ecommerce",
        latency_ms=800
    ))
    
    # Product recommendation
    traces.append(generate_trace_with_custom_filters(
        user_input="I need a laptop for gaming under $1500",
        ai_response="Based on your budget, I recommend the ASUS ROG Strix with RTX 4060, currently $1,299.",
        filter_config=ECOMMERCE_FILTERS,
        tool="Product-Recommendations",
        scenario="Product-Questions",
        model_name="recommendation-engine-v2",
        latency_ms=1200
    ))
    
    return traces

def create_healthcare_traces():
    """Create sample traces for healthcare use case."""
    traces = []
    
    # Symptom assessment
    traces.append(generate_trace_with_custom_filters(
        user_input="I have a headache and fever for 2 days",
        ai_response="Based on your symptoms, you may have a viral infection. Please consult a healthcare provider if symptoms worsen.",
        filter_config=HEALTHCARE_FILTERS,
        tool="Symptom-Checker",
        scenario="Symptom-Assessment",
        model_name="medical-ai-v3",
        medical_accuracy_score=0.92,
        safety_compliance=True
    ))
    
    return traces

def create_custom_frontend_config(filter_config: dict, industry: str):
    """Generate frontend configuration for custom filters."""
    
    config = {
        "industry": industry,
        "filterOptions": {
            "tools": ["All Tools"] + filter_config["tools"],
            "scenarios": ["All Scenarios"] + filter_config["scenarios"],
            "statuses": ["All Status", "pending", "accepted", "rejected"]
        }
    }
    
    # Add custom status options
    if "custom_statuses" in filter_config:
        config["filterOptions"]["statuses"].extend(filter_config["custom_statuses"])
    
    # Add custom filter categories
    if "compliance_tags" in filter_config:
        config["filterOptions"]["compliance"] = filter_config["compliance_tags"]
    
    if "educational_metrics" in filter_config:
        config["filterOptions"]["educational"] = filter_config["educational_metrics"]
    
    return config

# Generate configuration files for different industries
def setup_industry_configs():
    """Generate configuration for different industries."""
    
    configs = {
        "ecommerce": create_custom_frontend_config(ECOMMERCE_FILTERS, "E-commerce"),
        "healthcare": create_custom_frontend_config(HEALTHCARE_FILTERS, "Healthcare"), 
        "fintech": create_custom_frontend_config(FINTECH_FILTERS, "Financial Services"),
        "education": create_custom_frontend_config(EDUCATION_FILTERS, "Education")
    }
    
    return configs

if __name__ == "__main__":
    # Example: Create sample data for e-commerce
    ecommerce_traces = create_ecommerce_traces()
    print("E-commerce traces created:", len(ecommerce_traces))
    
    # Example: Create healthcare traces  
    healthcare_traces = create_healthcare_traces()
    print("Healthcare traces created:", len(healthcare_traces))
    
    # Example: Generate frontend configs
    industry_configs = setup_industry_configs()
    print("Industry configurations:", list(industry_configs.keys())) 