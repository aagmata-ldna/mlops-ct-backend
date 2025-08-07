import mlflow
from mlflow.tracking import MlflowClient
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging
import os

logger = logging.getLogger(__name__)

class TrackedModel:
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.client = MlflowClient()
        self.run = self._load_run()
        self._model_version = None
        self._registered_model = None

    def _load_run(self):
        try:
            return self.client.get_run(self.run_id)
        except Exception as e:
            raise RuntimeError(f"Failed to load run {self.run_id}: {str(e)}")

    def _load_model_version(self, model_name: str, version: str = None):
        """Load model version information if not already loaded"""
        if self._model_version is None:
            try:
                if version:
                    self._model_version = self.client.get_model_version(model_name, version)
                else:
                    # Try to find the model version by run_id
                    model_versions = self.client.search_model_versions(f"run_id='{self.run_id}'")
                    if model_versions:
                        self._model_version = model_versions[0]
            except Exception as e:
                logger.warning(f"Could not load model version for run {self.run_id}: {str(e)}")

    def serialize(self) -> Dict[str, Any]:
        """Serialize the tracked model to a dictionary"""
        run_info = self.run.info
        run_data = self.run.data

        return {
            "run_id": self.run_id,
            "experiment_id": run_info.experiment_id,
            "status": run_info.status,
            "start_time": run_info.start_time,
            "end_time": run_info.end_time,
            "artifact_uri": run_info.artifact_uri,
            "params": run_data.params,
            "metrics": run_data.metrics,
            "tags": run_data.tags,
        }

    def get_param(self, name: str) -> Optional[str]:
        """Get a parameter value by name"""
        return self.run.data.params.get(name)

    def get_metric(self, name: str) -> Optional[float]:
        """Get a metric value by name"""
        return self.run.data.metrics.get(name)

    def get_tag(self, name: str) -> Optional[str]:
        """Get a tag value by name"""
        return self.run.data.tags.get(name)

    def get_artifact_uri(self, subpath: str = "") -> str:
        """Get the artifact URI, optionally with a subpath"""
        base = self.run.info.artifact_uri.rstrip("/")
        return f"{base}/{subpath.lstrip('/')}" if subpath else base

    def get_all_metrics(self) -> Dict[str, float]:
        """Get all metrics for this run"""
        return dict(self.run.data.metrics) if self.run.data.metrics else {}

    def get_all_params(self) -> Dict[str, str]:
        """Get all parameters for this run"""
        return dict(self.run.data.params) if self.run.data.params else {}

    def get_all_tags(self) -> Dict[str, str]:
        """Get all tags for this run"""
        return dict(self.run.data.tags) if self.run.data.tags else {}

    def get_model_info(self, model_name: str = None, version: str = None) -> Dict[str, Any]:
        """Get model information if this run is associated with a registered model"""
        if model_name and version:
            self._load_model_version(model_name, version)
        elif model_name is None:
            # Try to find model version by run_id
            self._load_model_version("", None)

        if self._model_version:
            return {
                "name": self._model_version.name,
                "version": self._model_version.version,
                "stage": self._model_version.current_stage,
                "creation_timestamp": datetime.fromtimestamp(self._model_version.creation_timestamp / 1000),
                "last_updated_timestamp": datetime.fromtimestamp(self._model_version.last_updated_timestamp / 1000),
                "description": self._model_version.description,
                "tags": self._model_version.tags or {},
                "run_id": self._model_version.run_id
            }
        return {}

    def is_production_model(self) -> bool:
        """Check if this model is in production stage"""
        model_info = self.get_model_info()
        return model_info.get("stage") == "Production"

    def is_staging_model(self) -> bool:
        """Check if this model is in staging stage"""
        model_info = self.get_model_info()
        return model_info.get("stage") == "Staging"

    def get_experiment_name(self) -> str:
        """Get the experiment name for this run"""
        try:
            experiment = self.client.get_experiment(self.run.info.experiment_id)
            return experiment.name
        except Exception as e:
            logger.warning(f"Could not get experiment name: {str(e)}")
            return f"experiment_{self.run.info.experiment_id}"

class ModelTracker:
    """Manager class for tracked models"""
    
    def __init__(self):
        self.client = MlflowClient()
        self._configure_mlflow()

    def _configure_mlflow(self):
        """Configure MLflow connection"""
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        if os.getenv("MLFLOW_TRACKING_USERNAME") and os.getenv("MLFLOW_TRACKING_PASSWORD"):
            os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
            os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")

    def get_tracked_model(self, run_id: str) -> TrackedModel:
        """Get a TrackedModel instance for the given run_id"""
        return TrackedModel(run_id)

    def get_production_models(self) -> List[TrackedModel]:
        """Get all production models (now uses cache for better performance)"""
        try:
            from .model_cache import get_cached_production_models
            cached_models = get_cached_production_models()
            
            # Convert cached data to TrackedModel instances
            tracked_models = []
            for model_data in cached_models:
                try:
                    tracked_models.append(TrackedModel(model_data['run_id']))
                except Exception as e:
                    logger.warning(f"Failed to create TrackedModel for {model_data['name']}: {str(e)}")
                    continue
            
            return tracked_models
        except Exception as e:
            logger.error(f"Failed to get production models from cache: {str(e)}")
            return self._get_production_models_direct()

    def get_staging_models(self) -> List[TrackedModel]:
        """Get all staging models (now uses cache for better performance)"""
        try:
            from .model_cache import get_cached_staging_models
            cached_models = get_cached_staging_models()
            
            # Convert cached data to TrackedModel instances
            tracked_models = []
            for model_data in cached_models:
                try:
                    tracked_models.append(TrackedModel(model_data['run_id']))
                except Exception as e:
                    logger.warning(f"Failed to create TrackedModel for {model_data['name']}: {str(e)}")
                    continue
            
            return tracked_models
        except Exception as e:
            logger.error(f"Failed to get staging models from cache: {str(e)}")
            return self._get_staging_models_direct()

    def get_all_registered_models(self) -> List[TrackedModel]:
        """Get all registered models regardless of stage (now uses cache for better performance)"""
        try:
            from .model_cache import get_cached_all_models
            cached_models = get_cached_all_models()
            
            # Convert cached data to TrackedModel instances
            tracked_models = []
            for model_data in cached_models:
                try:
                    tracked_models.append(TrackedModel(model_data['run_id']))
                except Exception as e:
                    logger.warning(f"Failed to create TrackedModel for {model_data['name']}: {str(e)}")
                    continue
            
            return tracked_models
        except Exception as e:
            logger.error(f"Failed to get all models from cache: {str(e)}")
            return self._get_all_models_direct()

    def _get_production_models_direct(self) -> List[TrackedModel]:
        """Direct MLflow call for production models (fallback)"""
        try:
            models = self.client.search_registered_models()
            production_models = []
            
            for model in models:
                for version in model.latest_versions:
                    if version.current_stage == "Production":
                        production_models.append(TrackedModel(version.run_id))
            
            return production_models
        except Exception as e:
            logger.error(f"Failed to get production models directly: {str(e)}")
            return []

    def _get_staging_models_direct(self) -> List[TrackedModel]:
        """Direct MLflow call for staging models (fallback)"""
        try:
            models = self.client.search_registered_models()
            staging_models = []
            
            for model in models:
                for version in model.latest_versions:
                    if version.current_stage == "Staging":
                        staging_models.append(TrackedModel(version.run_id))
            
            return staging_models
        except Exception as e:
            logger.error(f"Failed to get staging models directly: {str(e)}")
            return []

    def _get_all_models_direct(self) -> List[TrackedModel]:
        """Direct MLflow call for all models (fallback)"""
        try:
            models = self.client.search_registered_models()
            all_models = []
            
            for model in models:
                for version in model.latest_versions:
                    all_models.append(TrackedModel(version.run_id))
            
            return all_models
        except Exception as e:
            logger.error(f"Failed to get all models directly: {str(e)}")
            return []

    def get_model_by_name_version(self, model_name: str, version: str) -> Optional[TrackedModel]:
        """Get a specific model by name and version"""
        try:
            model_version = self.client.get_model_version(model_name, version)
            return TrackedModel(model_version.run_id)
        except Exception as e:
            logger.error(f"Failed to get model {model_name} v{version}: {str(e)}")
            return None

    def search_models_by_tag(self, tag_key: str, tag_value: str = None) -> List[TrackedModel]:
        """Search for models by tag"""
        try:
            if tag_value:
                filter_string = f"tags.{tag_key}='{tag_value}'"
            else:
                filter_string = f"tags.{tag_key} LIKE '%'"
            
            model_versions = self.client.search_model_versions(filter_string)
            return [TrackedModel(version.run_id) for version in model_versions]
        except Exception as e:
            logger.error(f"Failed to search models by tag: {str(e)}")
            return []

    def get_experiments(self) -> List[Dict[str, Any]]:
        """Get all experiments"""
        try:
            experiments = self.client.search_experiments()
            return [{
                "experiment_id": exp.experiment_id,
                "name": exp.name,
                "artifact_location": exp.artifact_location,
                "lifecycle_stage": exp.lifecycle_stage,
                "creation_time": datetime.fromtimestamp(exp.creation_time / 1000),
                "last_update_time": datetime.fromtimestamp(exp.last_update_time / 1000) if exp.last_update_time else datetime.fromtimestamp(exp.creation_time / 1000),
                "tags": exp.tags or {}
            } for exp in experiments]
        except Exception as e:
            logger.error(f"Failed to get experiments: {str(e)}")
            return []