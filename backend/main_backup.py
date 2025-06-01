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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    try:
        await create_tables()
        print("Database tables created successfully")
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
            "Human Evaluation Dashboard"
        ]
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "LLM Evaluation Platform API",
        "timestamp": "2025-01-27T00:00:00Z"
    }


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