"""
Main FastAPI application for LLM Evaluation Platform.
"""

import uvicorn
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

# Use absolute imports that work when running directly
from database.connection import create_tables
from auth.security import get_current_user_email
from api.traces import router as traces_router
from api.tests import router as tests_router
from api.evaluations import router as evaluations_router
from api.integrations import router as integrations_router
from api.external import router as external_router
from api.large_dataset_handler import router as large_dataset_router
from api.experiments import router as experiments_router

# Performance middleware
from middleware.performance import (
    PerformanceMonitoringMiddleware,
    RateLimitMiddleware,
    CompressionMiddleware
)
from services.cache_service import cache_manager
from config.performance import get_performance_config


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    try:
        await create_tables()
        print("Database tables created successfully")
        
        # Warm up cache
        cache_manager.warm_up_cache()
        print("Cache service initialized")
        
    except Exception as e:
        print(f"Warning: Database initialization failed: {e}")
        print("The API will start but database-dependent features may not work")
    yield
    # Shutdown
    print("Application shutting down")


# Create FastAPI app
app = FastAPI(
    title="LLM Evaluation Platform",
    description="A comprehensive three-tier evaluation system for LLM-powered products",
    version="1.0.0",
    lifespan=lifespan,
)

# Add performance middleware (order matters - compression should be last)
app.add_middleware(PerformanceMonitoringMiddleware)
app.add_middleware(RateLimitMiddleware) 
app.add_middleware(CompressionMiddleware)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "LLM Evaluation Platform API",
        "version": "1.0.0",
        "status": "running",
        "features": [
            "Trace Logging System",
            "LangSmith Integration", 
            "Authentication",
            "Database Management",
            "Unit Testing Framework",
            "Human Evaluation Dashboard",
            "A/B Testing Framework",
            "Analytics Engine",
            "Data Export System",
            "Performance Optimization"
        ]
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "LLM Evaluation Platform API",
        "timestamp": "2025-01-27T00:00:00Z",
        "cache_status": cache_manager.get_cache_stats()["status"],
        "performance_config": get_performance_config()["optimizations"]
    }


@app.get("/metrics")
async def get_metrics(current_user_email: str = Depends(get_current_user_email)):
    """Get system performance metrics."""
    try:
        # Get cache statistics
        cache_stats = cache_manager.get_cache_stats()
        
        # Get performance configuration
        perf_config = get_performance_config()
        
        return {
            "cache": cache_stats,
            "performance": {
                "targets": {
                    "api_response_time_ms": perf_config["monitoring"]["api_response_time_target_ms"],
                    "error_rate_threshold": perf_config["monitoring"]["alert_on_error_rate_threshold"]
                }
            },
            "rate_limits": perf_config["rate_limits"],
            "optimizations": perf_config["optimizations"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")


@app.post("/admin/cache/clear")
async def clear_cache(current_user_email: str = Depends(get_current_user_email)):
    """Clear all cache data (admin only)."""
    try:
        success = cache_manager.clear_all_cache()
        return {
            "message": "Cache cleared successfully" if success else "Failed to clear cache",
            "success": success
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")


@app.post("/admin/cache/invalidate-stale")
async def invalidate_stale_cache(current_user_email: str = Depends(get_current_user_email)):
    """Remove stale cache entries."""
    try:
        cleared_count = cache_manager.invalidate_stale_data()
        return {
            "message": f"Cleared {cleared_count} stale cache entries",
            "cleared_count": cleared_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to invalidate stale cache: {str(e)}")


@app.get("/protected")
async def protected_endpoint(current_user_email: str = Depends(get_current_user_email)):
    """Protected endpoint to test authentication."""
    return {
        "message": f"Hello, {current_user_email}!",
        "authenticated": True
    }


# Include routers
app.include_router(traces_router, prefix="/api", tags=["Traces"])
app.include_router(tests_router, prefix="/api", tags=["Testing"])
app.include_router(evaluations_router, prefix="/api", tags=["Evaluations"])
app.include_router(integrations_router, prefix="/api/integrations", tags=["Integrations"])
app.include_router(external_router, prefix="/api/external", tags=["External Integration API"])
app.include_router(large_dataset_router, prefix="/api/large_dataset_handler", tags=["Large Dataset Handler"])
app.include_router(experiments_router, prefix="/api/experiments", tags=["Experiments"])

# Future routers (will be added as we build more features)
# app.include_router(auth_router, prefix="/auth", tags=["Authentication"])
# app.include_router(experiments_router, prefix="/api", tags=["Experiments"])


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 