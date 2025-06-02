# Interactive Onboarding Guide

## Welcome to the LLM Evaluation Platform! 🎉

This guide will help you get started with the platform quickly and effectively. Choose your path based on your role and experience level.

## Quick Start Paths

### 🎯 I'm New to LLM Evaluation
**Perfect for**: Product managers, domain experts, first-time users
- **Time**: 15 minutes
- **Goal**: Understand the platform and start evaluating AI responses
- [Start Here →](#new-user-onboarding)

### 🔧 I'm a Developer/Technical User  
**Perfect for**: Engineers, data scientists, technical integrators
- **Time**: 30 minutes
- **Goal**: Integrate APIs and set up automated evaluation workflows
- [Start Here →](#developer-onboarding)

### 📊 I'm an AI/ML Researcher
**Perfect for**: Researchers, AI engineers, evaluation specialists
- **Time**: 45 minutes
- **Goal**: Set up experiments, A/B tests, and advanced analytics
- [Start Here →](#researcher-onboarding)

---

## New User Onboarding

### Step 1: Platform Overview (5 minutes)

**What You'll Learn**: Understanding the three-tier evaluation system

#### Interactive Demo
```
🎬 Video: "LLM Evaluation Platform Overview" (3 min)
📍 Location: https://platform.demo/videos/overview
```

**Key Concepts**:
- **Level 1**: Automated unit tests (runs in background)
- **Level 2**: Human evaluation (your expertise matters here!)
- **Level 3**: A/B testing (measure real-world impact)

#### Quick Quiz
Test your understanding:
1. Which level involves human judgment? → Level 2 ✓
2. What runs automatically in the background? → Level 1 ✓
3. How do we measure business impact? → Level 3 ✓

### Step 2: Your First Evaluation (5 minutes)

**What You'll Do**: Rate your first AI response

#### Guided Walkthrough
1. **Navigate to Traces**
   ```
   Dashboard → Traces → View Sample Data
   ```

2. **Select a Trace**
   - Look for traces marked "Ready for Evaluation"
   - Click on any trace to see the full conversation

3. **Evaluate the Response**
   - Rate on a 1-5 scale for:
     - **Accuracy**: Is the information correct?
     - **Helpfulness**: Does it answer the question?
     - **Clarity**: Is it easy to understand?
   - Add written feedback (optional but valuable!)

4. **Submit Your Evaluation**
   - Click "Submit Evaluation"
   - See your contribution to the platform's knowledge base

#### Practice Exercise
```
🎯 Try This: Evaluate 3 different AI responses
📊 Goal: Get comfortable with the evaluation interface
⏱️ Time: 3 minutes
```

### Step 3: Understanding Your Impact (5 minutes)

**What You'll Learn**: How your evaluations help improve AI systems

#### View Your Contributions
1. **Personal Dashboard**
   ```
   Profile → My Evaluations → View Statistics
   ```

2. **Impact Metrics**
   - Number of evaluations completed
   - Average scores you've given
   - Most common feedback themes

3. **System Improvements**
   - See how your feedback influences model updates
   - Track improvements in AI response quality over time

#### Next Steps
- **Join the Community**: Connect with other evaluators
- **Set Goals**: Aim for 10 evaluations this week
- **Explore Advanced Features**: Ready for A/B testing?

---

## Developer Onboarding

### Step 1: API Setup (10 minutes)

**What You'll Do**: Get your API key and make your first API call

#### Get Your API Key
1. **Navigate to Settings**
   ```
   Dashboard → Settings → API Keys → Generate New Key
   ```

2. **Save Your Key Securely**
   ```bash
   # Add to your .env file
   EVAL_PLATFORM_API_KEY=your_api_key_here
   ```

#### First API Call
```python
import requests

# Test connection
response = requests.get(
    "http://localhost:8000/api/traces",
    headers={"Authorization": "Bearer YOUR_API_KEY"}
)

print(f"Status: {response.status_code}")
print(f"Data: {response.json()}")
```

#### Verify Setup
```bash
# Quick health check
curl -H "Authorization: Bearer YOUR_API_KEY" \
     http://localhost:8000/api/health
```

### Step 2: Log Your First Trace (10 minutes)

**What You'll Do**: Send AI interaction data to the platform

#### Basic Integration
```python
import requests
from datetime import datetime

def log_llm_interaction(user_input, ai_output, model_name):
    """Log an LLM interaction to the evaluation platform"""
    
    trace_data = {
        "user_input": user_input,
        "model_output": ai_output,
        "model_name": model_name,
        "timestamp": datetime.utcnow().isoformat(),
        "session_id": "demo_session_001"
    }
    
    response = requests.post(
        "http://localhost:8000/api/traces",
        headers={
            "Authorization": "Bearer YOUR_API_KEY",
            "Content-Type": "application/json"
        },
        json=trace_data
    )
    
    if response.status_code == 200:
        trace_id = response.json()["trace_id"]
        print(f"✅ Logged trace: {trace_id}")
        return trace_id
    else:
        print(f"❌ Error: {response.text}")
        return None

# Example usage
trace_id = log_llm_interaction(
    user_input="What is machine learning?",
    ai_output="Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
    model_name="gpt-4"
)
```

#### Verify Your Trace
1. **Check the Dashboard**
   ```
   Dashboard → Traces → Filter by Session ID: demo_session_001
   ```

2. **View Trace Details**
   - Click on your trace to see all logged information
   - Verify all fields are populated correctly

### Step 3: Set Up Automated Testing (10 minutes)

**What You'll Do**: Create your first automated test suite

#### Create a Test Suite
```python
import pytest
import requests

class TestLLMQuality:
    def setup_method(self):
        self.api_key = "YOUR_API_KEY"
        self.base_url = "http://localhost:8000"
        
    def test_response_length(self):
        """Test that responses are within acceptable length"""
        # Your LLM call here
        response = your_llm_function("Explain quantum computing")
        
        # Log to evaluation platform
        trace_id = self.log_trace("Explain quantum computing", response, "test-model")
        
        # Assertions
        assert len(response) > 50, "Response too short"
        assert len(response) < 1000, "Response too long"
        
    def test_response_relevance(self):
        """Test that responses contain relevant keywords"""
        response = your_llm_function("What is Python programming?")
        
        # Log to evaluation platform
        trace_id = self.log_trace("What is Python programming?", response, "test-model")
        
        # Check for relevant keywords
        keywords = ["python", "programming", "language", "code"]
        assert any(keyword.lower() in response.lower() for keyword in keywords)
        
    def log_trace(self, user_input, model_output, model_name):
        """Helper method to log traces"""
        # Implementation from Step 2
        pass
```

#### Run Your Tests
```bash
# Install pytest if needed
pip install pytest

# Run your test suite
pytest test_llm_quality.py -v

# View results in the platform
# Dashboard → Tests → View Recent Runs
```

---

## Researcher Onboarding

### Step 1: Advanced Analytics Setup (15 minutes)

**What You'll Do**: Configure comprehensive evaluation metrics and analytics

#### Set Up Evaluation Criteria
1. **Define Custom Metrics**
   ```python
   evaluation_criteria = {
       "accuracy": {
           "description": "Factual correctness of the response",
           "scale": "1-5",
           "guidelines": {
               "5": "Completely accurate, no errors",
               "4": "Mostly accurate, minor errors",
               "3": "Somewhat accurate, some errors",
               "2": "Mostly inaccurate, major errors", 
               "1": "Completely inaccurate"
           }
       },
       "coherence": {
           "description": "Logical flow and consistency",
           "scale": "1-5"
       },
       "safety": {
           "description": "Absence of harmful content",
           "scale": "binary",
           "options": ["safe", "unsafe"]
       }
   }
   ```

2. **Configure Evaluation Templates**
   ```
   Dashboard → Settings → Evaluation Templates → Create New
   ```

#### Set Up Analytics Dashboard
1. **Custom Metrics Tracking**
   - Model performance over time
   - Evaluation score distributions
   - Inter-rater reliability metrics
   - Cost and latency analysis

2. **Export Configuration**
   ```python
   # Configure automated data exports
   export_config = {
       "frequency": "daily",
       "format": "csv",
       "include_fields": [
           "trace_id", "timestamp", "model_name",
           "user_input", "model_output", "evaluation_scores",
           "latency_ms", "cost_usd"
       ],
       "filters": {
           "date_range": "last_30_days",
           "min_evaluation_count": 3
       }
   }
   ```

### Step 2: Design Your First Experiment (15 minutes)

**What You'll Do**: Set up a rigorous A/B test comparing different models or configurations

#### Experiment Design
```python
experiment_config = {
    "name": "GPT-4 vs Claude-3 Comparison",
    "description": "Compare response quality between GPT-4 and Claude-3 for customer support queries",
    "hypothesis": "Claude-3 will show higher customer satisfaction scores",
    
    "variants": [
        {
            "name": "control",
            "description": "GPT-4 with standard prompt",
            "config": {
                "model": "gpt-4",
                "temperature": 0.7,
                "max_tokens": 500
            },
            "traffic_percentage": 50
        },
        {
            "name": "treatment", 
            "description": "Claude-3 with optimized prompt",
            "config": {
                "model": "claude-3-sonnet",
                "temperature": 0.7,
                "max_tokens": 500
            },
            "traffic_percentage": 50
        }
    ],
    
    "metrics": [
        {
            "name": "response_quality",
            "type": "numeric",
            "description": "1-5 rating of response quality"
        },
        {
            "name": "customer_satisfaction",
            "type": "numeric", 
            "description": "1-10 customer satisfaction score"
        },
        {
            "name": "resolution_rate",
            "type": "binary",
            "description": "Whether the query was resolved"
        }
    ],
    
    "sample_size": {
        "target": 1000,
        "minimum_per_variant": 400,
        "power": 0.8,
        "significance_level": 0.05
    },
    
    "duration": {
        "max_days": 14,
        "early_stopping": True
    }
}
```

#### Create the Experiment
```python
import requests

def create_experiment(config):
    """Create a new A/B test experiment"""
    
    response = requests.post(
        "http://localhost:8000/api/experiments",
        headers={
            "Authorization": "Bearer YOUR_API_KEY",
            "Content-Type": "application/json"
        },
        json=config
    )
    
    if response.status_code == 200:
        experiment_id = response.json()["experiment_id"]
        print(f"✅ Created experiment: {experiment_id}")
        return experiment_id
    else:
        print(f"❌ Error: {response.text}")
        return None

# Create your experiment
experiment_id = create_experiment(experiment_config)
```

### Step 3: Advanced Analysis and Reporting (15 minutes)

**What You'll Do**: Set up sophisticated analysis pipelines and automated reporting

#### Statistical Analysis Pipeline
```python
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

class ExperimentAnalyzer:
    def __init__(self, experiment_id, api_key):
        self.experiment_id = experiment_id
        self.api_key = api_key
        
    def fetch_experiment_data(self):
        """Fetch experiment results from the platform"""
        response = requests.get(
            f"http://localhost:8000/api/experiments/{self.experiment_id}/results",
            headers={"Authorization": f"Bearer {self.api_key}"}
        )
        return response.json()
    
    def analyze_results(self):
        """Perform comprehensive statistical analysis"""
        data = self.fetch_experiment_data()
        
        # Extract metrics for each variant
        control_scores = data["variants"]["control"]["metrics"]["response_quality"]["values"]
        treatment_scores = data["variants"]["treatment"]["metrics"]["response_quality"]["values"]
        
        # Statistical tests
        t_stat, p_value = stats.ttest_ind(control_scores, treatment_scores)
        effect_size = self.calculate_cohens_d(control_scores, treatment_scores)
        
        # Confidence intervals
        control_ci = stats.t.interval(0.95, len(control_scores)-1, 
                                    loc=np.mean(control_scores), 
                                    scale=stats.sem(control_scores))
        treatment_ci = stats.t.interval(0.95, len(treatment_scores)-1,
                                      loc=np.mean(treatment_scores),
                                      scale=stats.sem(treatment_scores))
        
        return {
            "statistical_significance": p_value < 0.05,
            "p_value": p_value,
            "effect_size": effect_size,
            "control_mean": np.mean(control_scores),
            "treatment_mean": np.mean(treatment_scores),
            "control_ci": control_ci,
            "treatment_ci": treatment_ci,
            "sample_sizes": {
                "control": len(control_scores),
                "treatment": len(treatment_scores)
            }
        }
    
    def calculate_cohens_d(self, group1, group2):
        """Calculate Cohen's d effect size"""
        n1, n2 = len(group1), len(group2)
        pooled_std = np.sqrt(((n1-1)*np.var(group1) + (n2-1)*np.var(group2)) / (n1+n2-2))
        return (np.mean(group1) - np.mean(group2)) / pooled_std
    
    def generate_report(self):
        """Generate comprehensive experiment report"""
        results = self.analyze_results()
        
        report = f"""
        # Experiment Analysis Report
        
        ## Experiment: {self.experiment_id}
        
        ### Results Summary
        - **Statistical Significance**: {'Yes' if results['statistical_significance'] else 'No'}
        - **P-value**: {results['p_value']:.4f}
        - **Effect Size (Cohen's d)**: {results['effect_size']:.3f}
        
        ### Variant Performance
        - **Control Mean**: {results['control_mean']:.2f} (95% CI: {results['control_ci'][0]:.2f} - {results['control_ci'][1]:.2f})
        - **Treatment Mean**: {results['treatment_mean']:.2f} (95% CI: {results['treatment_ci'][0]:.2f} - {results['treatment_ci'][1]:.2f})
        
        ### Sample Sizes
        - **Control**: {results['sample_sizes']['control']} participants
        - **Treatment**: {results['sample_sizes']['treatment']} participants
        
        ### Recommendations
        {self.generate_recommendations(results)}
        """
        
        return report
    
    def generate_recommendations(self, results):
        """Generate actionable recommendations based on results"""
        if results['statistical_significance']:
            if results['treatment_mean'] > results['control_mean']:
                return "✅ **Recommendation**: Deploy the treatment variant. It shows statistically significant improvement."
            else:
                return "❌ **Recommendation**: Keep the control variant. Treatment shows significant decrease in performance."
        else:
            return "⚠️ **Recommendation**: No significant difference detected. Consider running longer or with larger sample size."

# Usage
analyzer = ExperimentAnalyzer(experiment_id, "YOUR_API_KEY")
report = analyzer.generate_report()
print(report)
```

#### Automated Reporting Setup
```python
# Set up automated daily reports
def setup_automated_reporting():
    """Configure automated experiment monitoring and reporting"""
    
    monitoring_config = {
        "experiment_id": experiment_id,
        "alerts": {
            "significance_threshold": 0.05,
            "minimum_sample_size": 100,
            "effect_size_threshold": 0.2
        },
        "reporting": {
            "frequency": "daily",
            "recipients": ["researcher@company.com"],
            "include_visualizations": True
        }
    }
    
    # Register monitoring
    response = requests.post(
        "http://localhost:8000/api/experiments/monitoring",
        headers={
            "Authorization": "Bearer YOUR_API_KEY",
            "Content-Type": "application/json"
        },
        json=monitoring_config
    )
    
    return response.json()
```

---

## Interactive Tutorials

### 🎮 Hands-On Exercises

#### Exercise 1: Evaluation Consistency Challenge
**Goal**: Learn to evaluate consistently across different types of content

```
📝 Task: Evaluate 10 AI responses about different topics
🎯 Focus: Maintain consistent scoring criteria
⏱️ Time: 15 minutes
🏆 Success: Achieve <0.5 standard deviation in your scores
```

**Topics Include**:
- Technical explanations
- Creative writing
- Customer support responses
- Educational content
- News summaries

#### Exercise 2: API Integration Challenge
**Goal**: Build a complete integration from scratch

```
🔧 Task: Create a mini-application that logs traces and retrieves evaluations
🎯 Focus: Error handling and best practices
⏱️ Time: 30 minutes
🏆 Success: Successfully log 5 traces and retrieve their evaluations
```

**Requirements**:
- Handle API errors gracefully
- Implement retry logic
- Add proper logging
- Include unit tests

#### Exercise 3: Experiment Design Challenge
**Goal**: Design and launch a statistically rigorous experiment

```
🧪 Task: Design an A/B test for a specific use case
🎯 Focus: Statistical power and practical significance
⏱️ Time: 45 minutes
🏆 Success: Launch experiment with proper power analysis
```

**Scenarios**:
- Customer support chatbot optimization
- Content generation quality improvement
- Response time vs. quality trade-off

### 🎬 Video Tutorials

#### Getting Started Series
1. **Platform Overview** (3 min)
   - Three-tier evaluation system
   - Navigation and key features
   - User roles and permissions

2. **Your First Evaluation** (5 min)
   - Finding traces to evaluate
   - Using the evaluation interface
   - Best practices for consistent scoring

3. **Understanding Analytics** (7 min)
   - Reading performance dashboards
   - Interpreting evaluation trends
   - Identifying improvement opportunities

#### Developer Series
1. **API Quickstart** (8 min)
   - Authentication setup
   - Making your first API calls
   - Error handling and debugging

2. **Integration Patterns** (12 min)
   - LangChain integration
   - OpenAI wrapper implementation
   - Batch processing strategies

3. **Testing and CI/CD** (15 min)
   - Automated test suites
   - GitHub Actions integration
   - Continuous evaluation workflows

#### Advanced Series
1. **Experiment Design** (20 min)
   - Statistical considerations
   - Sample size calculations
   - Avoiding common pitfalls

2. **Advanced Analytics** (18 min)
   - Custom metrics creation
   - Statistical analysis techniques
   - Reporting and visualization

3. **Production Deployment** (25 min)
   - Scaling considerations
   - Monitoring and alerting
   - Performance optimization

### 📊 Progress Tracking

#### Onboarding Checklist

**New User Path**:
- [ ] Complete platform overview
- [ ] Submit first evaluation
- [ ] View personal dashboard
- [ ] Join community forum
- [ ] Complete 10 evaluations

**Developer Path**:
- [ ] Get API key
- [ ] Make first API call
- [ ] Log first trace
- [ ] Set up automated testing
- [ ] Integrate with existing system

**Researcher Path**:
- [ ] Configure custom metrics
- [ ] Design first experiment
- [ ] Set up analytics pipeline
- [ ] Generate first report
- [ ] Configure automated monitoring

#### Skill Badges

Earn badges as you master different aspects of the platform:

🥉 **Evaluator Bronze**: Complete 25 evaluations
🥈 **Evaluator Silver**: Complete 100 evaluations with high consistency
🥇 **Evaluator Gold**: Become a top contributor with expert-level evaluations

🥉 **Developer Bronze**: Successfully integrate API
🥈 **Developer Silver**: Set up automated testing pipeline
🥇 **Developer Gold**: Contribute to platform improvements

🥉 **Researcher Bronze**: Launch first experiment
🥈 **Researcher Silver**: Complete statistically significant study
🥇 **Researcher Gold**: Publish research using platform data

### 🤝 Community and Support

#### Getting Help
- **Live Chat**: Available 9 AM - 5 PM PST
- **Community Forum**: 24/7 peer support
- **Office Hours**: Weekly Q&A sessions with experts
- **Documentation**: Comprehensive guides and tutorials

#### Contributing Back
- **Feedback**: Help improve the platform
- **Content**: Create tutorials and guides
- **Code**: Contribute to open-source components
- **Research**: Share findings and best practices

---

## Next Steps

### After Onboarding

1. **Set Your Goals**
   - Define what success looks like for your use case
   - Set measurable targets (evaluations, integrations, experiments)
   - Create a timeline for achieving your objectives

2. **Join the Community**
   - Introduce yourself in the forum
   - Follow relevant discussion topics
   - Share your experiences and learn from others

3. **Explore Advanced Features**
   - Custom evaluation criteria
   - Advanced analytics and reporting
   - Integration with your existing tools

4. **Stay Updated**
   - Subscribe to platform updates
   - Join monthly webinars
   - Follow our blog for best practices

### Continuous Learning

- **Monthly Challenges**: Participate in community challenges
- **Certification Program**: Earn official platform certifications
- **Research Partnerships**: Collaborate on evaluation research
- **Conference Presentations**: Share your success stories

---

*Welcome to the LLM Evaluation Platform community! We're excited to see what you'll build and discover. If you have any questions during onboarding, don't hesitate to reach out to our support team.* 