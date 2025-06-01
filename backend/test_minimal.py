"""
Minimal test FastAPI application.
"""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Create simple FastAPI app without database or complex imports
app = FastAPI(
    title="Test API",
    description="Minimal test API",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Test API is running", "status": "ok"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(
        "test_minimal:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_level="info"
    ) 