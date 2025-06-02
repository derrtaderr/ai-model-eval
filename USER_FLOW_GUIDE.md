# Complete User Flow Guide: LLM Evaluation Platform

## Overview: How Model Analysis Works

The platform operates on a **trace-based evaluation system**. Instead of directly calling your models, it analyzes traces (logs) of your model interactions to provide insights and enable human evaluation.

```
Your App → LLM Model → Response → Log Trace → Platform Analysis
```

## Step-by-Step User Flow

### Phase 1: Model Integration & Data Collection

#### Option A: Quick Start with Sample Data
1. **Start the platform:**
   ```bash
   cd backend && python run_backend.py
   cd frontend && npm run dev
   ```

2. **Upload sample data:**
   - Click "Upload Data" in dashboard
   - Upload the provided `sample_traces.json`
   - See 3 sample traces appear immediately

#### Option B: Production Integration (Recommended)
1. **Integrate logging into your app:**
   ```python
   from model_integration_example import LLMEvaluationLogger
   
   logger = LLMEvaluationLogger("http://localhost:8000")
   
   # After each model call:
   await logger.log_trace(
       user_input=prompt,
       model_output=response,
       model_name="gpt-4",
       tool="Customer-Support",    # Your use case
       scenario="Order-Inquiry",   # Specific scenario
       latency_ms=1200,
       cost_usd=0.005
   )
   ```

2. **Real-time data flow:**
   - Your app makes model calls as usual
   - Each interaction gets logged to the platform
   - Dashboard updates in real-time

### Phase 2: Three-Tier Analysis

#### Tier 1: Automated Unit Tests
- **What it analyzes:** Response quality, format compliance, safety
- **When it runs:** Automatically on every trace
- **Results:** Pass/fail scores visible in trace list

#### Tier 2: Human/Model Evaluation  
- **What you do:** Click traces → Review conversations → Accept/Reject
- **Interface:** Side-by-side user input and AI response
- **Output:** Human evaluation scores, agreement rates

#### Tier 3: A/B Testing
- **Setup:** Log traces with different `model_name` values
- **Analysis:** Compare performance between model variants
- **Results:** Statistical significance testing

### Phase 3: Insights & Optimization

#### Main Dashboard Analysis
- **Filter traces** by tool, scenario, status
- **View agreement rates** between human and automated scores
- **Identify problematic** traces for improvement
- **Export labeled data** for model retraining

#### Analytics Dashboard (Advanced)
- **System health** monitoring
- **Cost analysis** across models and time periods
- **Performance trends** and latency tracking
- **User engagement** metrics

## Customizing for Your Use Case

### 1. Define Your Categories

Edit your integration to use specific categories:

```python
# E-commerce example
await logger.log_trace(
    tool="Product-Recommendations",    # Your service/feature
    scenario="Gift-Suggestions",       # Specific use case
    # ... other fields
)
```

**Common Tools by Industry:**
- **E-commerce:** Product-Recommendations, Order-Management, Customer-Support
- **Healthcare:** Symptom-Checker, Medication-Info, Emergency-Triage  
- **Finance:** Account-Management, Investment-Advisor, Fraud-Detection
- **Education:** Tutor-Assistant, Essay-Reviewer, Study-Planner

### 2. Custom Filter Configuration

The dashboard filters automatically populate from your trace data:

```python
# Your logged traces determine available filters
tools = ["Product-Recommendations", "Customer-Support", "Order-Management"]
scenarios = ["Gift-Suggestions", "Size-Questions", "Order-Status"]

# These become filter options in the dashboard automatically
```

### 3. Industry-Specific Metrics

Add custom metadata for your domain:

```python
# Healthcare example
await logger.log_trace(
    # ... standard fields ...
    metadata={
        "medical_accuracy_score": 0.95,
        "safety_compliance": True,
        "patient_risk_level": "low"
    }
)
```

## Model Comparison & A/B Testing

### Setup A/B Tests

```python
# Route users to different models
variant = "gpt-4" if user_id % 2 == 0 else "claude-3"

response = await call_model(prompt, model=variant)

await logger.log_trace(
    user_input=prompt,
    model_output=response,
    model_name=variant,           # This creates the comparison
    tool="Chat-Assistant",
    scenario="General-Query",
    metadata={
        "experiment": "gpt4_vs_claude3",
        "user_segment": "premium_users"
    }
)
```

### Analyze Results

1. **Filter by model:** Use dashboard filters to view each variant
2. **Compare metrics:** Human acceptance rates, costs, latency
3. **Statistical significance:** Export data for statistical analysis
4. **Make decisions:** Switch to better-performing model

## Practical Examples by Industry

### E-commerce Customer Support

```python
# Product recommendation
await logger.log_trace(
    user_input="I need a laptop for gaming under $1500",
    model_output="I recommend the ASUS ROG Strix...",
    tool="Product-Recommendations",
    scenario="Budget-Gaming-Laptop",
    model_name="recommendation-engine-v2"
)

# Order status  
await logger.log_trace(
    user_input="Where is my order #12345?",
    model_output="Your order is in transit...",
    tool="Order-Management", 
    scenario="Order-Status-Inquiry",
    model_name="gpt-4-customer-service"
)
```

### Healthcare Assistant

```python
# Symptom assessment
await logger.log_trace(
    user_input="I have a headache and fever for 2 days",
    model_output="Based on your symptoms, you may have a viral infection...",
    tool="Symptom-Checker",
    scenario="Symptom-Assessment",
    model_name="medical-ai-v3",
    metadata={
        "medical_accuracy_score": 0.92,
        "requires_medical_review": True
    }
)
```

### Financial Services

```python
# Investment advice
await logger.log_trace(
    user_input="Should I invest in tech stocks right now?",
    model_output="Consider diversifying with both growth and value stocks...",
    tool="Investment-Advisor",
    scenario="Portfolio-Advice", 
    model_name="fintech-advisor-v1",
    metadata={
        "regulatory_compliant": True,
        "risk_disclosure_included": True
    }
)
```

## Dashboard Navigation

### Main Dashboard (`/`)
- **Purpose:** Daily evaluation workflow
- **Use for:** Reviewing traces, human evaluation, filtering data
- **Key features:** Upload data, trace list, evaluation interface

### Analytics Dashboard (`/analytics`)  
- **Purpose:** Strategic insights and monitoring
- **Use for:** Performance trends, cost analysis, system health
- **Key features:** Charts, metrics, real-time monitoring

## Common Workflows

### Daily Evaluation Workflow
1. Open main dashboard (`/`)
2. Filter to pending traces
3. Review conversations
4. Accept/reject responses
5. Add evaluation notes
6. Export results

### Model Performance Review
1. Go to Analytics Dashboard (`/analytics`)
2. Set time range (last week/month)
3. Compare model variants
4. Analyze cost vs. quality metrics
5. Make optimization decisions

### A/B Test Analysis
1. Filter traces by experiment metadata
2. Compare human acceptance rates
3. Analyze latency and cost differences
4. Check statistical significance
5. Deploy winning variant

## Best Practices

### 1. Consistent Categorization
- Use standardized tool/scenario names
- Create a taxonomy for your domain
- Document your categories for team alignment

### 2. Meaningful Metadata
- Include relevant business metrics
- Add experiment identifiers
- Track user segments and contexts

### 3. Regular Evaluation
- Set up daily evaluation routines
- Focus on high-risk scenarios first
- Track evaluation agreement over time

### 4. Continuous Improvement
- Use evaluation results for model fine-tuning
- Export labeled data for training
- Monitor performance trends over time

This platform turns your LLM interactions into a structured evaluation and improvement process, helping you build better AI products through data-driven insights. 