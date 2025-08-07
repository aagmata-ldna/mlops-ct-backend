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

- ✅ **Real-time data**: Direct MLflow integration, no stale cache data
- ✅ **Object-oriented**: Clean encapsulation of model properties and methods
- ✅ **No database**: Eliminates database dependencies and sync complexity
- ✅ **Scalable**: Easy to extend with new model tracking features
- ✅ **Type-safe**: Full Pydantic model validation

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

#### GET `/cache/info` - System Information
**Description**: Get information about the model tracking system  
**Response**:
```json
{
  "tracking_info": {
    "total_models": 15,
    "production_models": 3,
    "staging_models": 5,
    "monitored_models": 2
  },
  "system_status": {
    "mlflow_connected": true,
    "tracking_method": "real-time"
  }
}
```

#### POST `/cache/refresh` - Force Refresh
**Description**: Force refresh of model data (no-op since we use real-time data)  
**Response**:
```json
{
  "message": "No cache refresh needed - using real-time MLflow data",
  "models_available": 15,
  "timestamp": "2023-12-07T12:30:00"
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

### Feature Flags
```bash
ENABLE_MOCK_DATA=false              # Enable mock data when MLflow unavailable
ENABLE_DEBUG_ENDPOINTS=true         # Enable debug endpoints
MAX_MODELS_RETURNED=1000            # Maximum models per request
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

## 📝 Notes

- The API uses real-time MLflow data, eliminating cache inconsistencies
- All model tracking is done in-memory for monitoring management
- The system is designed to be stateless and horizontally scalable
- Object-oriented design makes it easy to extend with new features