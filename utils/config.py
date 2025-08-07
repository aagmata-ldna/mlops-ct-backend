"""
Configuration utilities for MLOps Control Tower Backend

This module provides configuration management for:
- MLflow connection settings
- API settings  
- Environment variables
- Default values
"""

import os
from typing import Optional, Dict, Any
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for the MLOps Control Tower Backend"""
    
    # MLflow Configuration
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "")
    MLFLOW_TRACKING_USERNAME: Optional[str] = os.getenv("MLFLOW_TRACKING_USERNAME")
    MLFLOW_TRACKING_PASSWORD: Optional[str] = os.getenv("MLFLOW_TRACKING_PASSWORD")
    
    # API Configuration
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    API_TITLE: str = "LifeDNA MLOps Control Tower API"
    API_VERSION: str = "1.0.0"
    
    # CORS Configuration
    CORS_ORIGINS: list = [
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ]
    
    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Model Tracking Configuration
    DEFAULT_MODEL_TYPE: str = "classifier"
    MAX_MODELS_RETURNED: int = int(os.getenv("MAX_MODELS_RETURNED", "1000"))
    CACHE_REFRESH_INTERVAL_MINUTES: int = int(os.getenv("CACHE_REFRESH_INTERVAL_MINUTES", "30"))
    ENABLE_CACHE: bool = os.getenv("ENABLE_CACHE", "true").lower() == "true"
    CACHE_TIMEOUT_SECONDS: int = int(os.getenv("CACHE_TIMEOUT_SECONDS", "300"))
    
    # Feature Flags
    ENABLE_MOCK_DATA: bool = os.getenv("ENABLE_MOCK_DATA", "false").lower() == "false"
    ENABLE_DEBUG_ENDPOINTS: bool = os.getenv("ENABLE_DEBUG_ENDPOINTS", "true").lower() == "true"
    
    @classmethod
    def get_mlflow_config(cls) -> Dict[str, Any]:
        """Get MLflow configuration as dictionary"""
        return {
            "tracking_uri": cls.MLFLOW_TRACKING_URI,
            "username": cls.MLFLOW_TRACKING_USERNAME,
            "password": cls.MLFLOW_TRACKING_PASSWORD
        }
    
    @classmethod
    def is_mlflow_configured(cls) -> bool:
        """Check if MLflow is properly configured"""
        return bool(cls.MLFLOW_TRACKING_URI)
    
    @classmethod
    def get_api_config(cls) -> Dict[str, Any]:
        """Get API configuration as dictionary"""
        return {
            "host": cls.API_HOST,
            "port": cls.API_PORT,
            "title": cls.API_TITLE,
            "version": cls.API_VERSION,
            "cors_origins": cls.CORS_ORIGINS
        }
    
    @classmethod
    def setup_logging(cls) -> None:
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, cls.LOG_LEVEL.upper(), logging.INFO),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    @classmethod
    def validate_config(cls) -> Dict[str, bool]:
        """Validate configuration and return status"""
        validation_results = {
            "mlflow_configured": cls.is_mlflow_configured(),
            "api_port_valid": 1000 <= cls.API_PORT <= 65535,
            "log_level_valid": cls.LOG_LEVEL.upper() in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        }
        
        return validation_results
    
    @classmethod
    def get_config_summary(cls) -> Dict[str, Any]:
        """Get a summary of current configuration"""
        return {
            "mlflow": {
                "tracking_uri": cls.MLFLOW_TRACKING_URI or "Not configured",
                "authentication": bool(cls.MLFLOW_TRACKING_USERNAME and cls.MLFLOW_TRACKING_PASSWORD)
            },
            "api": {
                "host": cls.API_HOST,
                "port": cls.API_PORT,
                "version": cls.API_VERSION
            },
            "features": {
                "mock_data_enabled": cls.ENABLE_MOCK_DATA,
                "debug_endpoints_enabled": cls.ENABLE_DEBUG_ENDPOINTS
            },
            "validation": cls.validate_config()
        }

# Initialize logging on module import
Config.setup_logging()