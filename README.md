# LLM Evaluation Platform

> A comprehensive three-tier evaluation system for LLM-powered products that enables rapid iteration, debugging, and quality assurance through automated testing, human review, and A/B testing capabilities.

## 🎯 Overview

The LLM Evaluation Platform accelerates the **evaluate → debug → iterate** flywheel to help AI product teams ship better products faster. It provides a structured approach to evaluating LLM outputs through three distinct levels:

### Three-Tier Evaluation System

1. **Level 1: Unit Tests** - Automated assertions and functional tests
2. **Level 2: Model & Human Evaluation** - Qualitative review with labeling  
3. **Level 3: A/B Testing** - Product-level impact measurement

## 🚀 Current Features

### ✅ Completed (Tasks 1-2)

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

## 🛠️ Tech Stack

### Backend
- **Framework**: FastAPI (Python)
- **Database**: PostgreSQL with async SQLAlchemy
- **Authentication**: JWT with bcrypt
- **Integrations**: LangSmith, LangChain
- **Background Jobs**: Celery + Redis
- **Testing**: pytest with async support

### Infrastructure  
- **Containerization**: Docker & Docker Compose
- **Database**: PostgreSQL 15
- **Caching**: Redis 7
- **Environment**: Python 3.11+

### Planned Frontend
- **Framework**: Next.js 14 with App Router
- **UI Library**: React with Tailwind CSS
- **State Management**: Zustand
- **Validation**: Zod
- **HTTP Client**: Built-in fetch with React Query

## 📋 Roadmap

### 🔄 In Progress
- **Task 3**: Unit Testing Framework (Next)
- **Task 10**: Frontend Development (High Priority)

### 📅 Upcoming Features
- **Task 4**: Human Evaluation Dashboard  
- **Task 5**: Advanced Filtering & Taxonomy System
- **Task 6**: Model-Based Evaluation Engine
- **Task 7**: A/B Testing Framework
- **Task 8**: Analytics Engine & Metrics Dashboard
- **Task 9**: Data Export & Integration System
- **Task 11**: Performance Optimization & Scaling
- **Task 12**: Documentation & Onboarding System

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
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

### 3. Run the API

```bash
# From the project root
cd backend
python main.py

# API will be available at http://localhost:8000
# Interactive docs at http://localhost:8000/docs
```

### 4. Verify Installation

```bash
# Health check
curl http://localhost:8000/health

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

### Example: Creating a Trace

```bash
curl -X POST "http://localhost:8000/api/traces" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "user_input": "What is the capital of France?",
    "model_output": "The capital of France is Paris.",
    "model_name": "gpt-4",
    "latency_ms": 1250,
    "cost_usd": 0.002
  }'
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

### Current Progress: 2/12 Tasks Complete (16.7%)

```
✅ Task 1: Core Infrastructure Setup
✅ Task 2: Trace Logging System  
⏳ Task 3: Unit Testing Framework (Next)
⏳ Task 4: Human Evaluation Dashboard
⏳ Task 5: Advanced Filtering & Taxonomy
⏳ Task 6: Model-Based Evaluation Engine
⏳ Task 7: A/B Testing Framework
⏳ Task 8: Analytics Engine & Metrics Dashboard
⏳ Task 9: Data Export & Integration System
⏳ Task 10: Frontend Development - React/Next.js
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

# Code formatting
black backend/
isort backend/
flake8 backend/
```

### Commit Convention

We push to GitHub after each completed task with descriptive commit messages:

```bash
git commit -m "feat(task-1): Core infrastructure setup

- Database models and connections
- FastAPI application structure  
- Authentication system
- Docker Compose configuration"
```

## 📈 Metrics & Monitoring

### Database Schema

Core entities:
- **Traces**: LLM interaction logs
- **Evaluations**: Human and model assessments
- **Test Cases**: Automated test definitions
- **Test Runs**: Test execution results
- **Experiments**: A/B test configurations
- **Users**: Authentication and permissions

### Performance Targets

- **API Response Time**: <200ms
- **Database Queries**: Optimized with proper indexing
- **Concurrent Users**: Designed for high scalability
- **Uptime**: 99.9% availability target

## 🔒 Security

- JWT-based authentication
- Bcrypt password hashing
- SQL injection prevention via SQLAlchemy
- Input validation with Pydantic
- CORS configuration for frontend integration

## 📖 Architecture

### Project Structure

```
ai-model-eval/
├── backend/                 # FastAPI application
│   ├── api/                 # API endpoints
│   ├── auth/                # Authentication
│   ├── database/            # Models and connections
│   ├── services/            # Business logic
│   └── utils/               # Utilities and integrations
├── frontend/                # Next.js application (planned)
├── scripts/                 # Task Master and utilities
├── tasks/                   # Generated task files
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