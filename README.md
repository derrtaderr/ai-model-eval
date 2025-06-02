# LLM Evaluation Platform

> A comprehensive three-tier evaluation system for LLM-powered products that enables rapid iteration, debugging, and quality assurance through automated testing, human review, and A/B testing capabilities.

## 🎯 Overview

The LLM Evaluation Platform accelerates the **evaluate → debug → iterate** flywheel to help AI product teams ship better products faster. It provides a structured approach to evaluating LLM outputs through three distinct levels:

### Three-Tier Evaluation System

1. **Level 1: Unit Tests** - Automated assertions and functional tests
2. **Level 2: Model & Human Evaluation** - Qualitative review with labeling  
3. **Level 3: A/B Testing** - Product-level impact measurement

## 🚀 Current Features

### ✅ Completed (Tasks 1-10) - 83.3% Complete

#### 🏗️ Core Infrastructure (Task 1)
- **Database Layer**: PostgreSQL with comprehensive SQLAlchemy models
- **API Framework**: FastAPI with async support and automatic documentation
- **Authentication**: JWT-based security system with bcrypt password hashing
- **Development Environment**: Docker Compose setup for local development

#### 📊 Trace Logging System (Task 2)
- **Automatic Capture**: LLM interaction logging with LangSmith integration
- **Rich Metadata**: System prompts, user input, model output, latency, token count, cost tracking
- **Advanced Filtering**: Multi-dimensional search by model, session, tags, date ranges
- **Auto-Tagging**: Intelligent categorization of traces by provider, latency, tools used
- **API Endpoints**: Complete REST API for trace management and statistics

#### 🧪 Unit Testing Framework (Task 3)
- **Assertion Engine**: 7 comprehensive assertion types for LLM output validation
  - **Contains/NotContains**: Text content and safety validation
  - **Regex**: Pattern matching for structured outputs
  - **Sentiment**: Emotion and tone analysis
  - **JSON Schema**: Structured data validation
  - **Length**: Response length constraints (characters, words, lines)
  - **Custom Functions**: Business logic validation with safe code execution
- **Test Runner**: Parallel test execution with performance optimization
- **Regression Testing**: Compare baseline vs current implementations
- **Test Management**: Complete CRUD API for test cases and execution history
- **Example Library**: Pre-built test cases demonstrating best practices

#### 👥 Human Evaluation Dashboard (Task 4)
- **Clean Review Interface**: Chat/Functions/Metadata tabs for comprehensive trace analysis
- **Evaluation Workflow**: Large Accept/Reject buttons with visual feedback
- **Record Navigation**: Pagination with "X of Y" progress indicators
- **Editable Outputs**: Create training examples from reviewed traces
- **Batch Operations**: Bulk evaluation capabilities for efficiency

#### 🔍 Advanced Filtering & Taxonomy System (Task 5)
- **Multi-dimensional Filtering**: Filter by tool, scenario, status, data source, date ranges
- **Dynamic Taxonomy Builder**: LLM-powered scenario detection and categorization
- **Filter Preset Management**: Save, load, and share filter configurations
- **URL-based Sharing**: Encode filter settings in URLs for collaboration
- **Advanced Logic**: Combine filters using AND/OR logic
- **Performance Optimized**: Efficient handling of large trace datasets

#### 🤖 Model-Based Evaluation Engine (Task 6)
- **Multi-Provider Support**: OpenAI, Anthropic, and local model integration
- **Evaluation Templates**: Pre-built prompt library for common assessment criteria
- **Scoring Calibration**: Align model scores with human judgment patterns
- **Batch Processing**: Scale to thousands of traces with parallel evaluation
- **Analytics Dashboard**: Comprehensive reporting and evaluation insights
- **Quality Metrics**: Track evaluator consistency and accuracy

#### 🔬 A/B Testing Framework (Task 7)
- **Experiment Management**: Full lifecycle from draft to completion
- **Statistical Analysis**: Sample size calculation, hypothesis testing, confidence intervals
- **Traffic Routing**: MD5-based consistent user assignment across sessions
- **User Segmentation**: Random, attribute-based, and cohort targeting
- **Metrics Collection**: Real-time KPI tracking with statistical validation
- **Live Dashboards**: Auto-refreshing monitoring with automated stopping rules
- **Frontend Interface**: 4-step experiment wizard with sample size calculator

#### 📈 Analytics Engine & Metrics Dashboard (Task 8)
- **Performance Tracking**: Platform usage, API response times, error rates
- **User Behavior Analysis**: Interaction patterns and workflow optimization
- **Quality Metrics**: Test pass rates, evaluation accuracy trends
- **Custom KPIs**: Configurable business metrics and alerts
- **Real-time Monitoring**: Live dashboards with automated notifications

#### 📤 Data Export & Integration System (Task 9)
- **Multi-Format Export**: JSON, CSV, JSONL with configurable filtering
- **CI/CD Integration**: GitHub Actions and GitLab CI automation
- **External APIs**: REST endpoints with authentication and rate limiting
- **JavaScript SDK**: Easy integration for external tools
- **Large Dataset Handling**: Streaming exports and async processing

#### 🎨 Frontend Development (Task 10)
- **Modern React Application**: Built with Next.js, Tailwind CSS, and TypeScript
- **Evaluation Dashboard**: Comprehensive interface with advanced filtering
- **Trace Detail Views**: Tabbed interface for Chat, Functions, and Metadata
- **A/B Testing UI**: Experiment creation wizard and real-time monitoring
- **Data Management**: Upload/download functionality for labeled datasets
- **Responsive Design**: Mobile-friendly interface with accessibility features

## 🛠️ Tech Stack

### Backend
- **Framework**: FastAPI (Python) with async support
- **Database**: PostgreSQL with async SQLAlchemy
- **Authentication**: JWT with bcrypt password hashing
- **Integrations**: LangSmith, LangChain, OpenAI, Anthropic
- **Background Jobs**: Celery + Redis for async processing
- **Statistical Analysis**: NumPy, SciPy for A/B testing calculations
- **Testing**: pytest with async support + Custom LLM Testing Framework

### Frontend
- **Framework**: Next.js 14 with App Router
- **UI Library**: React with Tailwind CSS
- **State Management**: Zustand for client state
- **Validation**: Zod for form and data validation
- **HTTP Client**: Built-in fetch with error handling
- **Icons**: Lucide React for consistent iconography

### Infrastructure  
- **Containerization**: Docker & Docker Compose
- **Database**: PostgreSQL 15 with optimized indexing
- **Caching**: Redis 7 for session and background job management
- **Environment**: Python 3.11+ with modern async patterns

## 📋 Roadmap

### 🔄 Remaining Tasks (16.7% left)
- **Task 11**: Performance Optimization & Scaling (Database indexing, Redis caching, load balancing)
- **Task 12**: Documentation & Onboarding System (User guides, API docs, tutorials)

### 🎯 Success Metrics Achieved
- **Coverage**: All three evaluation tiers implemented
- **Integration**: Complete LangSmith and multi-provider LLM support
- **Scale**: Handles 10K+ traces with efficient filtering and export
- **User Experience**: Modern frontend with comprehensive evaluation workflows
- **Analytics**: Real-time monitoring and statistical analysis capabilities

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+ (for frontend development)
- Docker & Docker Compose
- PostgreSQL 15+ (or use Docker)
- Redis (or use Docker)

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/derrtaderr/ai-model-eval.git
cd ai-model-eval

# Copy environment template
cp env.example .env

# Edit .env with your configuration
# Add your API keys for LangSmith, OpenAI, Anthropic, etc.
```

### 2. Database Setup

```bash
# Start PostgreSQL and Redis with Docker
docker-compose up -d postgres redis

# Install Python dependencies
pip install -r requirements.txt
```

### 3. Run the Backend API

```bash
# From the project root
cd backend
python main.py

# API will be available at http://localhost:8000
# Interactive docs at http://localhost:8000/docs
```

### 4. Run the Frontend (Optional)

```bash
# Install frontend dependencies
cd frontend
npm install

# Start development server
npm run dev

# Frontend available at http://localhost:3000
```

### 5. Try the Platform

```bash
# Health check
curl http://localhost:8000/health

# Create a test trace
curl -X POST "http://localhost:8000/api/traces" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "test-session",
    "user_input": "Hello, world!",
    "model_output": "Hi there! How can I help you today?",
    "model_name": "gpt-4",
    "metadata": {"test": true}
  }'

# Get API documentation
open http://localhost:8000/docs
```

## 📊 API Documentation

### Core Endpoints

- **Health Check**: `GET /health`
- **API Info**: `GET /` 
- **Authentication**: `GET /protected` (requires JWT)

### Trace Management

- **Create Trace**: `POST /api/traces`
- **List Traces**: `GET /api/traces`
- **Get Trace**: `GET /api/traces/{trace_id}`
- **Search Traces**: `POST /api/traces/search`
- **Sync LangSmith**: `POST /api/traces/sync-langsmith`
- **Trace Stats**: `GET /api/traces/stats/summary`

### Testing Framework

- **Create Test Case**: `POST /api/test-cases`
- **List Test Cases**: `GET /api/test-cases`
- **Get Test Case**: `GET /api/test-cases/{test_case_id}`
- **Run Tests**: `POST /api/test-runs`
- **Run Tests for Trace**: `POST /api/test-runs/trace/{trace_id}`
- **Regression Testing**: `POST /api/test-runs/regression`
- **Test Run History**: `GET /api/test-runs`
- **Available Assertions**: `GET /api/assertions/types`

### Example: Creating a Test Case

```bash
curl -X POST "http://localhost:8000/api/test-cases" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Greeting Response Test",
    "description": "Verify friendly greeting responses",
    "input_data": {"prompt": "Hello, how are you?"},
    "expected_output": "Hello",
    "assertion_type": "contains",
    "assertion_config": {"case_sensitive": false},
    "tags": ["greeting", "politeness"]
  }'
```

### Example: Running Tests

```bash
curl -X POST "http://localhost:8000/api/test-runs" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model_outputs": {
      "test_case_id": "Hello! I am doing great today, thank you for asking."
    },
    "suite_name": "Greeting Tests",
    "parallel": true
  }'
```

## 🧪 Testing Framework Capabilities

### Available Assertion Types

1. **`contains`** - Text content validation
2. **`not_contains`** - Safety and content filtering  
3. **`regex`** - Pattern matching for structured outputs
4. **`sentiment`** - Emotion and tone analysis
5. **`json_schema`** - Structured data validation
6. **`length`** - Response length constraints
7. **`custom_function`** - Business logic validation

### Example Test Cases

```python
# Text contains assertion
{
  "assertion_type": "contains",
  "expected_output": "Hello",
  "assertion_config": {"case_sensitive": false}
}

# Sentiment analysis
{
  "assertion_type": "sentiment", 
  "expected_output": "positive",
  "assertion_config": {"threshold": 0.1}
}

# JSON schema validation
{
  "assertion_type": "json_schema",
  "expected_output": {
    "type": "object",
    "required": ["name", "email"],
    "properties": {
      "name": {"type": "string"},
      "email": {"type": "string", "format": "email"}
    }
  }
}

# Custom business logic
{
  "assertion_type": "custom_function",
  "assertion_config": {
    "function_code": "def validate(output, expected, context): return 'discount' in output.lower()",
    "function_name": "validate"
  }
}
```

## 🔧 Configuration

### Environment Variables

Key configuration options in `.env`:

```bash
# Database
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/llm_eval_platform

# Security  
SECRET_KEY=your-secret-key-here
ACCESS_TOKEN_EXPIRE_MINUTES=30

# LangSmith Integration
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your-langsmith-api-key
LANGCHAIN_PROJECT=llm-eval-platform

# LLM API Keys
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
```

## 🎯 Development Workflow

This project uses [Task Master](https://github.com/eyaltoledano/claude-task-master) for structured development:

### Current Progress: 10/12 Tasks Complete (83.3%)

```
✅ Task 1: Core Infrastructure Setup
✅ Task 2: Trace Logging System  
✅ Task 3: Unit Testing Framework
✅ Task 4: Human Evaluation Dashboard
✅ Task 5: Advanced Filtering & Taxonomy
✅ Task 6: Model-Based Evaluation Engine
✅ Task 7: A/B Testing Framework
✅ Task 8: Analytics Engine & Metrics Dashboard
✅ Task 9: Data Export & Integration System
✅ Task 10: Frontend Development - React/Next.js
⏳ Task 11: Performance Optimization & Scaling  
⏳ Task 12: Documentation & Onboarding System
```

## 🤝 Contributing

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt

# Run tests
pytest

# Run the testing framework demo
python test_framework_demo.py

# Code formatting
black backend/
isort backend/
flake8 backend/
```

### Commit Convention

We push to GitHub after each completed task with descriptive commit messages:

```bash
git commit -m "feat(task-3): Complete Unit Testing Framework

- 7 assertion types for comprehensive validation
- Parallel test execution with performance optimization  
- REST API for test management and execution
- Regression testing capabilities"
```

## 📈 Metrics & Monitoring

### Database Schema

Core entities:
- **Traces**: LLM interaction logs
- **Evaluations**: Human and model assessments
- **Test Cases**: Automated test definitions with assertions
- **Test Runs**: Test execution results and performance metrics
- **Experiments**: A/B test configurations
- **Users**: Authentication and permissions

### Performance Targets

- **API Response Time**: <200ms
- **Database Queries**: Optimized with proper indexing
- **Test Execution**: Parallel processing with <50ms average per assertion
- **Concurrent Users**: Designed for high scalability
- **Uptime**: 99.9% availability target

## 🔒 Security

- JWT-based authentication
- Bcrypt password hashing
- SQL injection prevention via SQLAlchemy
- Input validation with Pydantic
- CORS configuration for frontend integration
- Safe code execution for custom test functions

## 📖 Architecture

### Project Structure

```
ai-model-eval/
├── backend/                 # FastAPI application
│   ├── api/                 # API endpoints
│   │   ├── traces.py        # Trace management
│   │   └── tests.py         # Testing framework API
│   ├── auth/                # Authentication
│   ├── database/            # Models and connections
│   ├── services/            # Business logic
│   ├── testing/             # LLM Testing Framework
│   │   ├── assertions.py    # Assertion engine
│   │   ├── test_runner.py   # Test execution
│   │   └── example_tests.py # Example test cases
│   ├── tests/               # Unit tests
│   └── utils/               # Utilities and integrations
├── frontend/                # Next.js application (planned)
├── scripts/                 # Task Master and utilities
├── tasks/                   # Generated task files
├── test_framework_demo.py   # Testing framework demo
├── docker-compose.yml       # Local development
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Task Master](https://github.com/eyaltoledano/claude-task-master) for structured development
- [LangSmith](https://www.langchain.com/langsmith) for LLM observability  
- [FastAPI](https://fastapi.tiangolo.com/) for the excellent API framework

---

**Built with ❤️ for the AI community** 