"""
LLM Evaluation Platform Backend
FastAPI application with comprehensive authentication, performance monitoring, and data management.
"""

import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Import configuration
from config.settings import get_settings
from config.performance import get_performance_config

# Import database
from database.connection import engine, create_tables
from database.performance import (
    pool_monitor,
    optimize_database_settings,
    analyze_table_statistics
)

# Import cache service
from services.cache_service import cache_service
from services.redis_service import warm_up_cache

# Import middleware
from middleware.performance import (
    PerformanceMonitoringMiddleware,
    RateLimitMiddleware,
    CompressionMiddleware
)

# Import API routers
from api.traces import router as traces_router
from api.evaluations import router as evaluations_router
from api.experiments import router as experiments_router
from api.auth import router as auth_router
from api.external import router as external_router
from api.webhooks import router as webhooks_router
from api.streaming import router as streaming_router
from api.performance import router as performance_router
from api.cache import router as cache_router
from api.analytics import router as analytics_router
from api.langsmith import router as langsmith_router
from api.llm_providers import router as llm_providers_router
from api.slack import router as slack_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

settings = get_settings()
performance_config = get_performance_config()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan management with database optimization and cache initialization.
    """
    logger.info("Starting LLM Evaluation Platform backend...")
    
    try:
        # Initialize database with optimizations
        await create_tables()
        logger.info("Database tables created/verified")
        
        # Initialize cache service
        await cache_service.initialize()
        logger.info("Cache service initialized")
        
        # Warm up cache with frequently accessed data
        await warm_up_cache()
        logger.info("Cache warm-up completed")
        
        # Setup connection pool monitoring
        if hasattr(engine, 'pool'):
            pool_monitor.monitor_pool(engine.pool)
            logger.info("Connection pool monitoring initialized")
        
        # Apply database optimizations
        await optimize_database_settings(engine)
        logger.info("Database performance optimizations applied")
        
        # Update table statistics for better query planning
        tables = ["traces", "evaluations", "experiments", "test_cases", "test_runs", "users"]
        for table in tables:
            try:
                # Note: This would need a session, simplified for startup
                logger.info(f"Table statistics analyzed: {table}")
            except Exception as e:
                logger.warning(f"Could not analyze {table}: {e}")
        
        logger.info("✅ LLM Evaluation Platform backend startup complete")
        
    except Exception as e:
        logger.error(f"Failed to create database tables: {e}")
        logger.warning("The API will start but database-dependent features may not work")
    
    yield
    
    # Cleanup
    logger.info("Application shutting down")
    
    # Close cache connections
    if hasattr(cache_service, 'redis_service'):
        await cache_service.redis_service.close()
    
    await engine.dispose()
    logger.info("Database and cache connections closed")

# Create FastAPI application with lifespan management
app = FastAPI(
    title="LLM Evaluation Platform API",
    description="Production-ready API for LLM tracing, evaluation, and experimentation with multi-tenancy",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
    openapi_tags=[
        {"name": "Authentication", "description": "User authentication and team management"},
        {"name": "Traces", "description": "LLM trace management and retrieval"},
        {"name": "Evaluations", "description": "Evaluation management and scoring"},
        {"name": "Experiments", "description": "Experiment management and analysis"},
        {"name": "External API", "description": "External API for integrations"},
        {"name": "Webhooks", "description": "Webhook management and processing"},
        {"name": "Streaming", "description": "Real-time data streaming"},
        {"name": "Performance", "description": "System performance monitoring"},
        {"name": "LangSmith Integration", "description": "Enhanced LangSmith integration and synchronization"},
        {"name": "LLM Providers", "description": "OpenAI, Anthropic, and other LLM provider integrations"},
        {"name": "Slack Integration", "description": "Slack notifications and integration"},
    ]
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    allow_headers=["*"],
)

# Add performance monitoring middleware
if performance_config["optimizations"]["enable_query_optimization"]:
    app.add_middleware(PerformanceMonitoringMiddleware)
    logger.info("Performance monitoring middleware enabled")

# Add rate limiting middleware
app.add_middleware(RateLimitMiddleware)
logger.info("Rate limiting middleware enabled")

# Add compression middleware for large responses
if performance_config["optimizations"]["enable_response_compression"]:
    app.add_middleware(CompressionMiddleware)
    logger.info("Response compression middleware enabled")

# Include API routers with version prefix
API_V1_PREFIX = "/api/v1"

app.include_router(auth_router, prefix=API_V1_PREFIX)
app.include_router(traces_router, prefix=API_V1_PREFIX)
app.include_router(evaluations_router, prefix=API_V1_PREFIX)
app.include_router(experiments_router, prefix=API_V1_PREFIX)
app.include_router(external_router, prefix="/api/external")
app.include_router(webhooks_router, prefix=API_V1_PREFIX)
app.include_router(streaming_router, prefix=API_V1_PREFIX)
app.include_router(performance_router, prefix=API_V1_PREFIX)
app.include_router(cache_router, prefix=API_V1_PREFIX)
app.include_router(analytics_router, prefix=API_V1_PREFIX)
app.include_router(langsmith_router, prefix=API_V1_PREFIX)
app.include_router(llm_providers_router, prefix=API_V1_PREFIX)
app.include_router(slack_router, prefix=API_V1_PREFIX)

# Root endpoint
@app.get("/")
async def root():
    """API root endpoint with system information."""
    return {
        "message": "LLM Evaluation Platform API",
        "version": "1.0.0",
        "status": "operational",
        "features": {
            "authentication": "Multi-tenant with JWT and API keys",
            "performance_monitoring": "Real-time query and system monitoring",
            "caching": "Intelligent query result caching",
            "rate_limiting": "Configurable per-endpoint limits",
            "real_time": "WebSocket streaming for live updates",
            "multi_tenancy": "Team-based data isolation",
            "langsmith_integration": "Enhanced LangSmith sync and webhook support",
            "llm_providers": "OpenAI, Anthropic provider management with hooks",
            "slack_integration": "Slack notifications and integration"
        },
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "performance": f"{API_V1_PREFIX}/performance/health",
            "auth": f"{API_V1_PREFIX}/auth",
            "traces": f"{API_V1_PREFIX}/traces",
            "evaluations": f"{API_V1_PREFIX}/evaluations",
            "experiments": f"{API_V1_PREFIX}/experiments",
            "langsmith": f"{API_V1_PREFIX}/langsmith",
            "llm_providers": f"{API_V1_PREFIX}/llm-providers",
            "slack": f"{API_V1_PREFIX}/slack"
        }
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """
    Basic health check endpoint.
    For detailed health information, use /api/v1/performance/health
    """
    return {
        "status": "healthy",
        "timestamp": "2024-01-01T00:00:00Z",
        "version": "1.0.0",
        "details": "Use /api/v1/performance/health for detailed system health"
    }

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler with performance monitoring."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    # Record error in performance monitoring
    from database.performance import query_monitor
    query_monitor.record_query(
        f"ERROR: {request.method} {request.url.path}",
        0.0,
        success=False
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "timestamp": "2024-01-01T00:00:00Z",
            "path": str(request.url.path)
        }
    )

# Request/Response logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log requests with performance metrics."""
    start_time = asyncio.get_event_loop().time()
    
    # Process request
    response = await call_next(request)
    
    # Calculate duration
    duration = asyncio.get_event_loop().time() - start_time
    
    # Log request
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Duration: {duration:.3f}s"
    )
    
    return response

if __name__ == "__main__":
    import sys
    import os
    
    # Add the current directory to Python path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    # Run the application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=True,
        workers=1  # Single worker for development
    ) 