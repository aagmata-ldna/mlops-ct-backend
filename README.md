# LifeDNA MLOps Control Tower API

A FastAPI-based control tower for managing and monitoring MLflow models using object-oriented programming and real-time data tracking.

## 🏗️ Architecture Overview

This backend leverages object-oriented programming with the `TrackedModel` pattern to provide real-time MLflow model tracking without database dependencies.

### Key Components

- **TrackedModel**: Object-oriented wrapper for MLflow runs with comprehensive model properties
- **ModelTracker**: Manager class for handling collections of TrackedModel instances
- **MonitoringManager**: In-memory monitoring system for tracked models
- **Config**: Centralized configuration management
- **Constants**: Shared constants and utilities

### Project Structure

```
├── main.py                    # FastAPI application entry point
├── start_server.py           # Server startup script with configuration
├── requirements.txt          # Python dependencies
├── utils/                    # All utilities and configuration modules
│   ├── __init__.py          # Package exports
│   ├── tracked_model.py     # TrackedModel & ModelTracker classes
│   ├── monitoring_manager.py # MonitoringManager for monitoring
│   ├── config.py            # Configuration management
│   └── constants.py         # Shared constants and enums
├── .env                     # Environment configuration (create this)
└── README.md               # This documentation
```

### Benefits

- ✅ **High Performance**: Intelligent caching system loads models at startup for fast responses
- ✅ **Real-time data**: Background refresh keeps cache current with MLflow changes
- ✅ **Object-oriented**: Clean encapsulation of model properties and methods  
- ✅ **No database**: Eliminates database dependencies and sync complexity
- ✅ **Scalable**: Easy to extend with new model tracking features
- ✅ **Type-safe**: Full Pydantic model validation
- ✅ **Intelligent fallback**: Direct MLflow calls when cache unavailable

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
Create a `.env` file in the project root:
```bash
# MLflow Configuration (Required)
MLFLOW_TRACKING_URI=https://your-mlflow-server.com
MLFLOW_TRACKING_USERNAME=your_username
MLFLOW_TRACKING_PASSWORD=your_password

# API Configuration (Optional - defaults shown)
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO

# Feature Flags (Optional - defaults shown)
ENABLE_MOCK_DATA=false
ENABLE_DEBUG_ENDPOINTS=true
```

### 3. Start the Server

Choose any method:

```bash
# Method 1: Using startup script (recommended)
python start_server.py

# Method 2: Direct Python
python main.py

# Method 3: Using uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Method 4: With custom options
python start_server.py --host 0.0.0.0 --port 8080 --reload
```

### 4. Access the API

Once running:
- **🌐 API Root**: http://localhost:8000/
- **📚 Interactive Docs**: http://localhost:8000/docs (Swagger UI)
- **📖 Alternative Docs**: http://localhost:8000/redoc (ReDoc)
- **❤️ Health Check**: http://localhost:8000/health

### 5. Quick Test

```bash
# Check if API is running
curl http://localhost:8000/health

# Get all production models
curl http://localhost:8000/models

# Get configuration
curl http://localhost:8000/config
```

## 📚 API Documentation

### System Endpoints

#### GET `/` - Root
**Description**: Basic API information  
**Response**:
```json
{
  "message": "LifeDNA MLOps Control Tower API",
  "status": "ready",
  "version": "1.0.0",
  "docs": "/docs"
}
```

#### GET `/health` - Health Check
**Description**: Check API and MLflow connectivity  
**Response**:
```json
{
  "status": "healthy",
  "mlflow_connection": "connected",
  "experiments_count": 5
}
```

#### GET `/config` - Configuration
**Description**: Get current configuration summary  
**Response**:
```json
{
  "mlflow": {
    "tracking_uri": "https://your-mlflow-server.com",
    "authentication": true
  },
  "api": {
    "host": "0.0.0.0",
    "port": 8000,
    "version": "1.0.0"
  },
  "features": {
    "mock_data_enabled": false,
    "debug_endpoints_enabled": true
  },
  "validation": {
    "mlflow_configured": true,
    "api_port_valid": true,
    "log_level_valid": true
  }
}
```

### Model Endpoints

#### GET `/models` - Production Models
**Description**: Get all models in production stage  
**Response**: Array of `ModelInfo` objects
```json
[
  {
    "name": "DNA_Analysis_Model",
    "version": "1",
    "stage": "Production",
    "creation_timestamp": "2023-12-01T10:00:00",
    "last_updated_timestamp": "2023-12-05T15:30:00",
    "description": "Main DNA sequence analysis model",
    "tags": {"team": "genomics", "type": "classification"},
    "metrics": {
      "accuracy": 0.945,
      "precision": 0.928,
      "recall": 0.952,
      "f1_score": 0.940
    },
    "model_type": "classifier",
    "run_id": "abc123def456"
  }
]
```

#### GET `/models/all` - All Models
**Description**: Get all registered models regardless of stage  
**Response**: Array of `ModelInfo` objects

#### GET `/models/staging` - Staging Models  
**Description**: Get all models in staging stage  
**Response**: Array of `ModelInfo` objects

#### GET `/models/production` - Production Models (Detailed)
**Description**: Same as `/models` but explicit endpoint  
**Response**: Array of `ModelInfo` objects

### Monitoring Endpoints

#### GET `/models/monitored` - Monitored Models
**Description**: Get all models currently being monitored with real-time MLflow data  
**Response**: Array of `ModelInfo` objects

#### GET `/models/monitored/classifier` - Monitored Classifiers
**Description**: Get monitored models filtered by classifier type  
**Response**: Array of `ModelInfo` objects

#### GET `/models/monitored/regressor` - Monitored Regressors
**Description**: Get monitored models filtered by regressor type  
**Response**: Array of `ModelInfo` objects

#### POST `/models/{model_name}/{model_version}/monitor` - Add to Monitoring
**Description**: Add a model to the monitoring system  
**Path Parameters**:
- `model_name`: Name of the model (string)
- `model_version`: Version of the model (string)

**Request Body**:
```json
{
  "model_type": "classifier",    // "classifier" or "regressor"
  "registered_by": "user_name"   // Who is adding this model
}
```

**Example Request**:
```bash
curl -X POST "http://localhost:8000/models/DNA_Analysis_Model/1/monitor" \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "classifier",
    "registered_by": "data_scientist_john"
  }'
```

**Response**:
```json
{
  "message": "Model DNA_Analysis_Model v1 added to monitoring as classifier",
  "model_name": "DNA_Analysis_Model",
  "model_version": "1",
  "model_type": "classifier",
  "registered_at": "2023-12-07T10:30:00"
}
```

#### DELETE `/models/{model_name}/{model_version}/monitor` - Remove from Monitoring
**Description**: Remove a model from monitoring  
**Path Parameters**:
- `model_name`: Name of the model (string)
- `model_version`: Version of the model (string)

**Example Request**:
```bash
curl -X DELETE "http://localhost:8000/models/DNA_Analysis_Model/1/monitor"
```

**Response**:
```json
{
  "message": "Model DNA_Analysis_Model v1 removed from monitoring",
  "model_name": "DNA_Analysis_Model",
  "model_version": "1",
  "removed_at": "2023-12-07T11:00:00"
}
```

#### GET `/models/{model_name}/{model_version}/monitor/status` - Monitoring Status
**Description**: Check if a model is being monitored  
**Response**:
```json
{
  "model_name": "DNA_Analysis_Model",
  "model_version": "1",
  "is_monitored": true
}
```

### Experiment and Run Endpoints

#### GET `/experiments` - List Experiments
**Description**: Get all MLflow experiments  
**Response**:
```json
[
  {
    "experiment_id": "1",
    "name": "DNA_Analysis_Experiment",
    "artifact_location": "/mlflow/artifacts/1",
    "lifecycle_stage": "active",
    "creation_time": "2023-12-01T08:00:00",
    "last_update_time": "2023-12-07T12:00:00",
    "tags": {"team": "genomics"}
  }
]
```

#### GET `/runs/{experiment_id}` - Experiment Runs
**Description**: Get runs for a specific experiment  
**Query Parameters**:
- `limit`: Maximum number of runs to return (default: 100)

**Example**: `GET /runs/1?limit=50`

**Response**:
```json
[
  {
    "run_id": "abc123def456",
    "experiment_id": "1",
    "status": "FINISHED",
    "start_time": "2023-12-05T10:00:00",
    "end_time": "2023-12-05T12:00:00",
    "metrics": {"accuracy": 0.945, "loss": 0.123},
    "params": {"learning_rate": "0.001", "epochs": "100"},
    "tags": {"mlflow.user": "john_doe"}
  }
]
```

### Model Analysis Endpoints

#### GET `/model/{model_name}/versions` - Model Versions
**Description**: Get all versions of a specific model  
**Response**:
```json
[
  {
    "version": "1",
    "stage": "Production",
    "creation_timestamp": "2023-12-01T10:00:00",
    "last_updated_timestamp": "2023-12-05T15:30:00",
    "description": "Initial production model",
    "tags": {"validated": "true"},
    "metrics": {"accuracy": 0.945},
    "run_id": "abc123def456"
  }
]
```

#### GET `/model/{model_name}/drift` - Model Drift Analysis
**Description**: Get drift analysis for a model over time  
**Query Parameters**:
- `days`: Number of days to analyze (default: 30)

**Example**: `GET /model/DNA_Analysis_Model/drift?days=7`

**Response**:
```json
[
  {
    "version": "1",
    "stage": "Production",
    "metrics_over_time": [
      {
        "date": "2023-12-01T00:00:00",
        "accuracy": 0.945,
        "precision": 0.928,
        "recall": 0.952,
        "f1_score": 0.940
      }
    ]
  }
]
```

### System Management Endpoints

#### GET `/cache/info` - Cache and System Information
**Description**: Get information about the model tracking system and cache performance  
**Response**:
```json
{
  "cache_info": {
    "total_models": 15,
    "models_by_stage": {
      "Production": 3,
      "Staging": 5,
      "Archived": 2,
      "None": 5
    },
    "last_refresh": "2023-12-07T12:30:00",
    "is_refreshing": false,
    "cache_age_minutes": 15.2,
    "is_stale": false
  },
  "tracking_info": {
    "total_models": 15,
    "production_models": 3,
    "staging_models": 5,
    "monitored_models": 2
  },
  "system_status": {
    "mlflow_connected": true,
    "tracking_method": "cached",
    "cache_enabled": true
  }
}
```

#### POST `/cache/refresh` - Force Cache Refresh
**Description**: Force an immediate refresh of the model cache from MLflow  
**Response**:
```json
{
  "message": "Cache refreshed successfully",
  "models_synced": 15,
  "timestamp": "2023-12-07T12:30:00",
  "cache_stats": {
    "total_models": 15,
    "models_by_stage": {
      "Production": 3,
      "Staging": 5,
      "Archived": 2,
      "None": 5
    },
    "last_refresh": "2023-12-07T12:30:00",
    "is_refreshing": false,
    "cache_age_minutes": 0,
    "is_stale": false,
    "persistence": {
      "enabled": true,
      "cache_directory": "./cache",
      "cache_file_exists": true,
      "cache_file_age_hours": 0.0,
      "cache_file_size_bytes": 1024,
      "max_age_hours": 24
    }
  }
}
```

#### DELETE `/cache/clear` - Clear Persistent Cache
**Description**: Clear the persistent cache file (forces fresh fetch on next startup)  
**Response**:
```json
{
  "message": "Persistent cache cleared successfully",
  "timestamp": "2023-12-07T12:35:00",
  "note": "Next startup will fetch fresh data from MLflow"
}
```

#### GET `/dashboard/summary` - Dashboard Summary
**Description**: Get summary statistics for dashboard  
**Response**:
```json
{
  "total_experiments": 5,
  "active_experiments": 3,
  "total_models": 15,
  "monitored_models": 2,
  "production_models": 3,
  "staging_models": 5,
  "total_runs": 75,
  "last_updated": "2023-12-07T12:30:00"
}
```

### Debug Endpoints

#### GET `/debug/monitored` - Debug Monitored Models
**Description**: Debug endpoint to inspect monitored models data  
**Response**:
```json
{
  "monitored_count": 2,
  "monitored_list": [
    {
      "model_name": "DNA_Analysis_Model",
      "model_version": "1",
      "model_type": "classifier",
      "registered_at": "2023-12-07T10:30:00",
      "registered_by": "john_doe",
      "run_id": "abc123def456"
    }
  ],
  "mlflow_status": "connected",
  "model_details": [...]
}
```

## 💻 Usage Examples

### Using curl

```bash
# Check API health
curl http://localhost:8000/health

# Get all production models
curl http://localhost:8000/models

# Add model to monitoring
curl -X POST "http://localhost:8000/models/MyModel/1/monitor" \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "classifier",
    "registered_by": "data_scientist"
  }'

# Remove model from monitoring
curl -X DELETE "http://localhost:8000/models/MyModel/1/monitor"

# Get dashboard summary
curl http://localhost:8000/dashboard/summary

# Check monitoring status
curl http://localhost:8000/models/MyModel/1/monitor/status

# Check cache performance and statistics
curl http://localhost:8000/cache/info

# Force cache refresh
curl -X POST http://localhost:8000/cache/refresh
```

### Using Python requests

```python
import requests

# Base URL
BASE_URL = "http://localhost:8000"

# Check health
response = requests.get(f"{BASE_URL}/health")
print(response.json())

# Get production models
models = requests.get(f"{BASE_URL}/models").json()
print(f"Found {len(models)} production models")

# Add model to monitoring
monitoring_request = {
    "model_type": "classifier",
    "registered_by": "python_script"
}
response = requests.post(
    f"{BASE_URL}/models/MyModel/1/monitor",
    json=monitoring_request
)
print(response.json())

# Get monitored models
monitored = requests.get(f"{BASE_URL}/models/monitored").json()
print(f"Monitoring {len(monitored)} models")

# Remove from monitoring
response = requests.delete(f"{BASE_URL}/models/MyModel/1/monitor")
print(response.json())

# Check cache performance
cache_info = requests.get(f"{BASE_URL}/cache/info").json()
print(f"Cache has {cache_info['cache_info']['total_models']} models")
print(f"Cache age: {cache_info['cache_info']['cache_age_minutes']:.1f} minutes")

# Force cache refresh
refresh_result = requests.post(f"{BASE_URL}/cache/refresh").json()
print(f"Refreshed {refresh_result['models_synced']} models")
```

### Using Python with error handling

```python
import requests
from requests.exceptions import RequestException

BASE_URL = "http://localhost:8000"

def api_request(method, endpoint, **kwargs):
    """Make API request with error handling"""
    try:
        response = requests.request(method, f"{BASE_URL}{endpoint}", **kwargs)
        response.raise_for_status()
        return response.json()
    except RequestException as e:
        print(f"API Error: {e}")
        return None

# Usage
health = api_request("GET", "/health")
if health:
    print(f"API Status: {health['status']}")

models = api_request("GET", "/models")
if models:
    print(f"Production models: {len(models)}")
```

## 🔧 Configuration Options

The API can be configured through environment variables in `.env` file:

### MLflow Configuration
```bash
MLFLOW_TRACKING_URI=https://your-mlflow-server.com  # Required
MLFLOW_TRACKING_USERNAME=your_username              # Optional
MLFLOW_TRACKING_PASSWORD=your_password              # Optional
```

### API Configuration
```bash
API_HOST=0.0.0.0                    # Server host (default: 0.0.0.0)
API_PORT=8000                       # Server port (default: 8000)  
LOG_LEVEL=INFO                      # Logging level (DEBUG/INFO/WARNING/ERROR)
```

### Performance & Caching
```bash
CACHE_REFRESH_INTERVAL_MINUTES=30      # Cache refresh interval (default: 30 minutes)
CACHE_TIMEOUT_SECONDS=300              # Cache initialization timeout (default: 300s/5 minutes)
ENABLE_CACHE=true                      # Enable/disable caching (default: true)
MAX_MODELS_RETURNED=1000               # Maximum models per request

# Persistent Cache Configuration (NEW!)
ENABLE_CACHE_PERSISTENCE=true         # Enable persistent cache storage (default: true)
CACHE_DIRECTORY=./cache                # Directory for cache files (default: ./cache)
CACHE_MAX_AGE_HOURS=24                 # Max age before cache refresh (default: 24 hours)
```

### Feature Flags
```bash
ENABLE_MOCK_DATA=false              # Enable mock data when MLflow unavailable
ENABLE_DEBUG_ENDPOINTS=true         # Enable debug endpoints
```

## 📊 HTTP Status Codes

- **200**: Success
- **201**: Created (for POST requests)
- **400**: Bad Request (invalid input)
- **404**: Not Found (model/resource doesn't exist)
- **500**: Internal Server Error

## 🚨 Error Handling

All endpoints return appropriate HTTP status codes and error messages:

```json
{
  "detail": "Model not found in MLflow"
}
```

Common error scenarios:
- **Model not found**: 404 with descriptive message
- **Already monitoring**: 400 "Model is already being monitored"
- **Invalid model type**: 400 "Invalid model type. Must be classifier or regressor"
- **MLflow connection issues**: 500 with connection error details

## 🛠️ Development

### Server Options

```bash
# Development with auto-reload
python start_server.py --reload

# Custom host and port
python start_server.py --host 0.0.0.0 --port 8080

# Show configuration without starting
python start_server.py --config
```

### Testing

The API includes comprehensive input validation and error handling. Test your integration using the examples above or the interactive docs at `/docs`.

## 🤝 Support

- **Interactive Documentation**: Visit `/docs` when the server is running
- **Health Check**: Use `/health` endpoint to verify connectivity
- **Configuration**: Check `/config` endpoint for current settings
- **Debug Information**: Use `/debug/monitored` for troubleshooting

## 🐛 Troubleshooting

### Server Hanging on Startup

If the server seems stuck during startup:

1. **Check the logs** for cache initialization messages
2. **Disable cache temporarily**:
   ```bash
   export ENABLE_CACHE=false
   python main.py
   ```
3. **Reduce cache timeout**:
   ```bash
   export CACHE_TIMEOUT_SECONDS=30
   python main.py
   ```
4. **Check MLflow connectivity**:
   ```bash
   curl http://localhost:8000/health
   ```

### Cache Issues

- **Cache not loading**: Check MLflow connection and credentials
- **Slow responses**: Cache may still be initializing - check `/health` endpoint
- **Memory issues**: Reduce `MAX_MODELS_RETURNED` or disable cache
- **Force refresh**: Use `POST /cache/refresh` to manually update cache

### Common Configuration Issues

```bash
# Disable cache for faster startup during development
ENABLE_CACHE=false

# Reduce timeout for faster failure detection  
CACHE_TIMEOUT_SECONDS=30

# Less frequent cache updates to reduce server load
CACHE_REFRESH_INTERVAL_MINUTES=60
```

## ⚡ Performance

### Intelligent Caching System with Persistence

The API implements a high-performance caching system with persistent storage that addresses the slow `client.search_registered_models()` problem:

**🚀 Startup Cache Loading**
- **NEW**: Loads from persistent cache file if available (instant startup!)
- Falls back to MLflow if cache is stale or missing
- Background thread periodically refreshes cache (default: 30 minutes)  
- Fast in-memory lookups for all model queries

**💾 Persistent Cache Storage**
- **NEW**: Automatically saves processed model data to local JSON files
- **Super fast restarts**: No need to re-fetch from MLflow if cache is fresh
- Configurable cache expiry (default: 24 hours)
- Atomic file writes prevent corruption
- Graceful degradation if cache files are corrupted

**📊 Performance Benefits**
- **First startup**: 10-30 seconds (fetches from MLflow)
- **Subsequent startups**: ~1-2 seconds (loads from cache!)
- **Production models**: ~50ms (vs ~5-10 seconds direct MLflow)
- **All models**: ~20ms (vs ~10-30 seconds direct MLflow)
- **Staging models**: ~30ms (vs ~3-8 seconds direct MLflow)

**🔄 Cache Management**
- Automatic background refresh keeps data current
- Persistent cache automatically saved after refresh
- Manual refresh available via `/cache/refresh` endpoint
- Cache clearing available via `/cache/clear` endpoint
- Intelligent fallback to direct MLflow calls if cache fails
- Cache statistics available via `/cache/info` endpoint

**⚙️ Configuration**
```bash
# Cache behavior
CACHE_REFRESH_INTERVAL_MINUTES=15      # More frequent updates
CACHE_REFRESH_INTERVAL_MINUTES=60      # Less frequent updates

# Persistent cache settings
ENABLE_CACHE_PERSISTENCE=true          # Enable persistent storage
CACHE_DIRECTORY=./cache                 # Where to store cache files
CACHE_MAX_AGE_HOURS=24                  # When to refresh from MLflow
CACHE_MAX_AGE_HOURS=1                   # For testing/development

# To disable persistence (always fetch from MLflow)
ENABLE_CACHE_PERSISTENCE=false
```

## 📝 Notes

- The API uses intelligent caching with background refresh for optimal performance
- All model tracking is done in-memory for monitoring management
- The system is designed to be stateless and horizontally scalable
- Object-oriented design makes it easy to extend with new features
- Fallback to direct MLflow calls ensures reliability even if cache fails