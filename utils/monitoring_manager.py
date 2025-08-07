from typing import Dict, List, Set, Tuple
from datetime import datetime
import logging
from .tracked_model import ModelTracker, TrackedModel

logger = logging.getLogger(__name__)

class MonitoringManager:
    """In-memory manager for monitored models"""
    
    def __init__(self):
        self.model_tracker = ModelTracker()
        # Store monitored models as (model_name, version) -> model_type mapping
        self._monitored_models: Dict[Tuple[str, str], Dict] = {}

    def add_monitored_model(self, model_name: str, model_version: str, 
                          model_type: str = "classifier", registered_by: str = "user") -> bool:
        """Add a model to monitoring"""
        try:
            # Verify the model exists in MLflow
            tracked_model = self.model_tracker.get_model_by_name_version(model_name, model_version)
            if not tracked_model:
                logger.error(f"Model {model_name} v{model_version} not found in MLflow")
                return False
            
            model_key = (model_name, model_version)
            self._monitored_models[model_key] = {
                "model_type": model_type,
                "registered_at": datetime.now().isoformat(),
                "registered_by": registered_by,
                "run_id": tracked_model.run_id
            }
            
            logger.info(f"Added {model_name} v{model_version} to monitoring as {model_type}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add model to monitoring: {str(e)}")
            return False

    def remove_monitored_model(self, model_name: str, model_version: str) -> bool:
        """Remove a model from monitoring"""
        try:
            model_key = (model_name, model_version)
            if model_key in self._monitored_models:
                del self._monitored_models[model_key]
                logger.info(f"Removed {model_name} v{model_version} from monitoring")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to remove model from monitoring: {str(e)}")
            return False

    def is_model_monitored(self, model_name: str, model_version: str) -> bool:
        """Check if a model is being monitored"""
        model_key = (model_name, model_version)
        return model_key in self._monitored_models

    def get_monitored_models_list(self) -> List[Dict]:
        """Get list of monitored models with metadata"""
        result = []
        for (model_name, model_version), metadata in self._monitored_models.items():
            result.append({
                "model_name": model_name,
                "model_version": model_version,
                "model_type": metadata["model_type"],
                "registered_at": metadata["registered_at"],
                "registered_by": metadata["registered_by"],
                "run_id": metadata["run_id"]
            })
        return result

    def get_monitored_models(self) -> List[TrackedModel]:
        """Get all monitored models as TrackedModel instances with real-time data"""
        tracked_models = []
        
        for (model_name, model_version), metadata in self._monitored_models.items():
            try:
                # Get fresh data from MLflow for each monitored model
                tracked_model = self.model_tracker.get_model_by_name_version(model_name, model_version)
                if tracked_model:
                    tracked_models.append(tracked_model)
                else:
                    logger.warning(f"Monitored model {model_name} v{model_version} not found in MLflow")
            except Exception as e:
                logger.error(f"Failed to get tracked model for {model_name} v{model_version}: {str(e)}")
                continue
        
        return tracked_models

    def get_monitored_classifier_models(self) -> List[TrackedModel]:
        """Get monitored classifier models"""
        classifier_models = []
        
        for (model_name, model_version), metadata in self._monitored_models.items():
            if metadata["model_type"] == "classifier":
                try:
                    tracked_model = self.model_tracker.get_model_by_name_version(model_name, model_version)
                    if tracked_model:
                        classifier_models.append(tracked_model)
                except Exception as e:
                    logger.error(f"Failed to get classifier model {model_name} v{model_version}: {str(e)}")
                    continue
        
        return classifier_models

    def get_monitored_regressor_models(self) -> List[TrackedModel]:
        """Get monitored regressor models"""
        regressor_models = []
        
        for (model_name, model_version), metadata in self._monitored_models.items():
            if metadata["model_type"] == "regressor":
                try:
                    tracked_model = self.model_tracker.get_model_by_name_version(model_name, model_version)
                    if tracked_model:
                        regressor_models.append(tracked_model)
                except Exception as e:
                    logger.error(f"Failed to get regressor model {model_name} v{model_version}: {str(e)}")
                    continue
        
        return regressor_models

    def get_monitored_count(self) -> int:
        """Get the count of monitored models"""
        return len(self._monitored_models)