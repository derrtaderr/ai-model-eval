#!/usr/bin/env python3
"""
Script to run the backend with proper import path handling.
"""

import sys
import os

# Add the current directory to Python path so backend imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Now run the backend
if __name__ == "__main__":
    os.chdir("backend")
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0", 
        port=8000,
        reload=True,
        log_level="info"
    ) 