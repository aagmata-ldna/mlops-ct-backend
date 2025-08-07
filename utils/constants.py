"""
Constants for MLOps Control Tower Backend

This module contains shared constants used throughout the application.
"""

from typing import List

# Model Types
class ModelTypes:
    CLASSIFIER = "classifier"
    REGRESSOR = "regressor"
    
    @classmethod
    def get_all(cls) -> List[str]:
        return [cls.CLASSIFIER, cls.REGRESSOR]
    
    @classmethod
    def is_valid(cls, model_type: str) -> bool:
        return model_type in cls.get_all()

# Model Stages
class ModelStages:
    PRODUCTION = "Production"
    STAGING = "Staging"
    ARCHIVED = "Archived"
    NONE = "None"
    
    @classmethod
    def get_all(cls) -> List[str]:
        return [cls.PRODUCTION, cls.STAGING, cls.ARCHIVED, cls.NONE]
    
    @classmethod
    def is_valid(cls, stage: str) -> bool:
        return stage in cls.get_all()

# API Response Messages
class ResponseMessages:
    # Success messages
    MODEL_ADDED_TO_MONITORING = "Model {model_name} v{model_version} added to monitoring as {model_type}"
    MODEL_REMOVED_FROM_MONITORING = "Model {model_name} v{model_version} removed from monitoring"
    CACHE_REFRESH_COMPLETED = "Cache refresh completed"
    
    # Error messages
    MODEL_NOT_FOUND = "Model not found in MLflow"
    MODEL_NOT_IN_PRODUCTION = "Model not found in production"
    MODEL_ALREADY_MONITORED = "Model is already being monitored"
    MODEL_NOT_MONITORED = "Model is not being monitored"
    MLFLOW_CONNECTION_FAILED = "Failed to connect to MLflow"
    INVALID_MODEL_TYPE = "Invalid model type. Must be one of: {valid_types}"
    
    @classmethod
    def format_message(cls, message: str, **kwargs) -> str:
        """Format a message with provided keyword arguments"""
        return message.format(**kwargs)

# Default Metrics by Model Type
class DefaultMetrics:
    CLASSIFIER_METRICS = [
        "accuracy", "precision", "recall", "f1_score", "auc", "roc_auc"
    ]
    
    REGRESSOR_METRICS = [
        "rmse", "mae", "r2_score", "mse", "mean_absolute_error", "root_mean_squared_error"
    ]
    
    @classmethod
    def get_metrics_for_type(cls, model_type: str) -> List[str]:
        """Get default metrics for a given model type"""
        if model_type == ModelTypes.CLASSIFIER:
            return cls.CLASSIFIER_METRICS
        elif model_type == ModelTypes.REGRESSOR:
            return cls.REGRESSOR_METRICS
        else:
            return []

# API Endpoints
class APIEndpoints:
    # Model endpoints
    MODELS = "/models"
    MODELS_ALL = "/models/all"
    MODELS_PRODUCTION = "/models/production"
    MODELS_STAGING = "/models/staging"
    MODELS_MONITORED = "/models/monitored"
    MODELS_MONITORED_CLASSIFIER = "/models/monitored/classifier"
    MODELS_MONITORED_REGRESSOR = "/models/monitored/regressor"
    
    # Monitoring endpoints
    MODEL_MONITOR = "/models/{model_name}/{model_version}/monitor"
    MODEL_MONITOR_STATUS = "/models/{model_name}/{model_version}/monitor/status"
    
    # System endpoints
    HEALTH = "/health"
    CONFIG = "/config"
    CACHE_INFO = "/cache/info"
    CACHE_REFRESH = "/cache/refresh"
    DASHBOARD_SUMMARY = "/dashboard/summary"
    
    # Debug endpoints
    DEBUG_MONITORED = "/debug/monitored"

# HTTP Status Codes
class StatusCodes:
    OK = 200
    CREATED = 201
    BAD_REQUEST = 400
    NOT_FOUND = 404
    INTERNAL_SERVER_ERROR = 500

# Time Constants (in seconds)
class TimeConstants:
    MINUTE = 60
    HOUR = 3600
    DAY = 86400
    WEEK = 604800
    
    # Default timeouts
    DEFAULT_REQUEST_TIMEOUT = 30
    DEFAULT_MLFLOW_TIMEOUT = 60