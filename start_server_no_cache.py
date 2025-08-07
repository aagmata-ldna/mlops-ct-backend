#!/usr/bin/env python3
"""
Quick startup script for development - disables cache for faster startup
"""

import os
import sys

# Disable cache for fast startup
os.environ["ENABLE_CACHE"] = "false"
os.environ["LOG_LEVEL"] = "INFO"

print("🚀 Starting server with cache disabled (fast startup mode)")
print("📝 This is useful for development and debugging")

# Import and run the main app
from main import app
import uvicorn
from utils import Config

if __name__ == "__main__":
    api_config = Config.get_api_config()
    uvicorn.run(
        app,
        host=api_config["host"], 
        port=api_config["port"],
        log_level="info"
    )