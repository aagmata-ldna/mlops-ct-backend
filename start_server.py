#!/usr/bin/env python3
"""
Start the MLOps Control Tower server with proper initialization
"""

import uvicorn
import logging
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    """Start the server"""
    try:
        logger.info("🚀 Starting LifeDNA MLOps Control Tower...")
        logger.info("📊 Dashboard will be available at: http://localhost:8000")
        logger.info("📚 API docs will be available at: http://localhost:8000/docs")
        logger.info("⏳ MLflow sync will start in background after server startup...")
        
        # Start the FastAPI server
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=False,  # Disable reload for production
            log_level="info"
        )
        
    except KeyboardInterrupt:
        logger.info("👋 Server stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()