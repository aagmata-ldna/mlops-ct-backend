"""
Model Cache Manager for MLOps Control Tower Backend

This module provides caching functionality for MLflow models to improve performance.
Models are loaded at startup and cached in memory with optional background refresh.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from threading import Lock
import mlflow
from mlflow.tracking import MlflowClient
from .config import Config

logger = logging.getLogger(__name__)

class ModelCache:
    """
    In-memory cache for MLflow models with background refresh capability.
    
    This cache improves performance by:
    - Loading all models at startup
    - Providing fast lookups by stage
    - Supporting background refresh
    - Maintaining freshness metadata
    """
    
    def __init__(self, refresh_interval_minutes: int = 30):
        self.client = MlflowClient()
        self.refresh_interval = refresh_interval_minutes * 60  # Convert to seconds
        self._configure_mlflow()
        
        # Cache storage
        self._models_by_stage: Dict[str, List[Dict]] = {}
        self._all_models: List[Dict] = []
        self._models_by_name: Dict[str, Dict] = {}
        
        # Cache metadata
        self._last_refresh: Optional[datetime] = None
        self._is_refreshing: bool = False
        self._refresh_lock = Lock()
        
        # Background refresh task
        self._refresh_task: Optional[asyncio.Task] = None
        self._should_stop_refresh: bool = False
    
    def _configure_mlflow(self):
        """Configure MLflow connection"""
        mlflow_config = Config.get_mlflow_config()
        if mlflow_config["tracking_uri"]:
            mlflow.set_tracking_uri(mlflow_config["tracking_uri"])
            
            if mlflow_config["username"] and mlflow_config["password"]:
                import os
                os.environ["MLFLOW_TRACKING_USERNAME"] = mlflow_config["username"]
                os.environ["MLFLOW_TRACKING_PASSWORD"] = mlflow_config["password"]
    
    async def initialize(self) -> bool:
        """
        Initialize the cache by loading all models.
        Should be called at application startup.
        """
        logger.info("Initializing model cache...")
        success = await self._refresh_cache()
        
        if success:
            logger.info(f"Model cache initialized with {len(self._all_models)} models")
            # Start background refresh task
            self._start_background_refresh()
        else:
            logger.warning("Model cache initialization failed")
        
        return success
    
    async def _refresh_cache(self) -> bool:
        """Refresh the cache with fresh data from MLflow"""
        if self._is_refreshing:
            logger.debug("Cache refresh already in progress, skipping")
            return True
        
        with self._refresh_lock:
            if self._is_refreshing:
                return True
            self._is_refreshing = True
        
        try:
            logger.info("Refreshing model cache from MLflow...")
            start_time = datetime.now()
            
            # Fetch all registered models
            models = await self._fetch_all_models()
            
            if models is None:
                logger.error("Failed to fetch models from MLflow")
                return False
            
            # Process and organize models
            models_by_stage = {
                "Production": [],
                "Staging": [],
                "Archived": [],
                "None": []
            }
            models_by_name = {}
            
            for model_data in models:
                stage = model_data.get("stage", "None")
                if stage not in models_by_stage:
                    models_by_stage[stage] = []
                
                models_by_stage[stage].append(model_data)
                
                # Index by name for quick lookup
                model_key = f"{model_data['name']}:{model_data['version']}"
                models_by_name[model_key] = model_data
            
            # Update cache atomically
            self._models_by_stage = models_by_stage
            self._all_models = models
            self._models_by_name = models_by_name
            self._last_refresh = datetime.now()
            
            refresh_duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"Cache refreshed successfully in {refresh_duration:.2f}s - {len(models)} models loaded")
            
            return True
            
        except Exception as e:
            logger.error(f"Cache refresh failed: {str(e)}")
            return False
        finally:
            self._is_refreshing = False
    
    async def _fetch_all_models(self) -> Optional[List[Dict]]:
        """Fetch all models from MLflow and convert to dictionary format"""
        try:
            # Use asyncio to prevent blocking with timeout
            def fetch_models():
                logger.info("Fetching registered models from MLflow...")
                
                try:
                    models = self.client.search_registered_models(max_results=1000)
                    logger.info(f"Found {len(models)} registered models, processing versions...")
                    model_data = []
                    processed_count = 0
                    
                    # Only process models in staging or production stages
                    target_stages = {"Production", "Staging"}
                    
                    for model in models:
                        for version in model.latest_versions:
                            try:
                                # Skip models not in target stages
                                if version.current_stage not in target_stages:
                                    continue
                                
                                processed_count += 1
                                if processed_count % 10 == 0:
                                    logger.info(f"Processing {version.current_stage} model {processed_count}: {model.name} v{version.version}")
                                
                                # Get run details for metrics (this can be slow)
                                run = self.client.get_run(version.run_id)
                                
                                # Extract metrics
                                metrics = {}
                                if run.data.metrics:
                                    metrics = dict(run.data.metrics)
                                
                                model_info = {
                                    'name': model.name,
                                    'version': version.version,
                                    'stage': version.current_stage,
                                    'creation_timestamp': datetime.fromtimestamp(version.creation_timestamp / 1000).isoformat(),
                                    'last_updated_timestamp': datetime.fromtimestamp(version.last_updated_timestamp / 1000).isoformat(),
                                    'description': version.description,
                                    'tags': version.tags or {},
                                    'metrics': metrics,
                                    'run_id': version.run_id,
                                    'cached_at': datetime.now().isoformat()
                                }
                                model_data.append(model_info)
                                
                            except Exception as e:
                                logger.warning(f"Failed to process model {model.name} v{version.version}: {str(e)}")
                                continue
                    
                    logger.info(f"Successfully processed {len(model_data)} model versions")
                    return model_data
                    
                except Exception as e:
                    logger.error(f"Failed to fetch models from MLflow: {str(e)}")
                    raise
            
            # Run in thread pool with timeout
            loop = asyncio.get_event_loop()
            models = await asyncio.wait_for(
                loop.run_in_executor(None, fetch_models),
                timeout=Config.CACHE_TIMEOUT_SECONDS
            )
            return models
            
        except asyncio.TimeoutError:
            logger.error("Timeout while fetching models from MLflow")
            return None
        except Exception as e:
            logger.error(f"Failed to fetch models from MLflow: {str(e)}")
            return None
    
    def get_production_models(self) -> List[Dict]:
        """Get cached production models"""
        return self._models_by_stage.get("Production", []).copy()
    
    def get_staging_models(self) -> List[Dict]:
        """Get cached staging models"""
        return self._models_by_stage.get("Staging", []).copy()
    
    def get_all_models(self) -> List[Dict]:
        """Get all cached models"""
        return self._all_models.copy()
    
    def get_models_by_stage(self, stage: str) -> List[Dict]:
        """Get cached models by stage"""
        return self._models_by_stage.get(stage, []).copy()
    
    def get_model_by_name_version(self, name: str, version: str) -> Optional[Dict]:
        """Get a specific model from cache"""
        model_key = f"{name}:{version}"
        return self._models_by_name.get(model_key)
    
    def search_models_by_name(self, name_pattern: str) -> List[Dict]:
        """Search models by name pattern"""
        matching_models = []
        for model in self._all_models:
            if name_pattern.lower() in model['name'].lower():
                matching_models.append(model)
        return matching_models
    
    def get_cache_stats(self) -> Dict[str, any]:
        """Get cache statistics"""
        return {
            "total_models": len(self._all_models),
            "models_by_stage": {stage: len(models) for stage, models in self._models_by_stage.items()},
            "last_refresh": self._last_refresh.isoformat() if self._last_refresh else None,
            "is_refreshing": self._is_refreshing,
            "cache_age_minutes": (
                (datetime.now() - self._last_refresh).total_seconds() / 60 
                if self._last_refresh else None
            ),
            "is_stale": self.is_cache_stale()
        }
    
    def is_cache_stale(self, max_age_minutes: int = None) -> bool:
        """Check if cache is stale"""
        if not self._last_refresh:
            return True
        
        max_age = max_age_minutes or (self.refresh_interval / 60)
        age = datetime.now() - self._last_refresh
        return age > timedelta(minutes=max_age)
    
    async def force_refresh(self) -> bool:
        """Force an immediate cache refresh"""
        logger.info("Forcing cache refresh...")
        return await self._refresh_cache()
    
    def _start_background_refresh(self):
        """Start background refresh task"""
        if self._refresh_task and not self._refresh_task.done():
            return
        
        async def background_refresh():
            """Background task to periodically refresh cache"""
            while not self._should_stop_refresh:
                try:
                    await asyncio.sleep(self.refresh_interval)
                    if not self._should_stop_refresh:
                        await self._refresh_cache()
                except asyncio.CancelledError:
                    logger.info("Background refresh task cancelled")
                    break
                except Exception as e:
                    logger.error(f"Background refresh failed: {str(e)}")
        
        loop = asyncio.get_event_loop()
        self._refresh_task = loop.create_task(background_refresh())
        logger.info(f"Background refresh started (interval: {self.refresh_interval/60:.1f} minutes)")
    
    def stop_background_refresh(self):
        """Stop background refresh task"""
        self._should_stop_refresh = True
        if self._refresh_task and not self._refresh_task.done():
            self._refresh_task.cancel()
            logger.info("Background refresh stopped")
    
    def __del__(self):
        """Cleanup on destruction"""
        self.stop_background_refresh()

# Global cache instance
_model_cache: Optional[ModelCache] = None

def get_model_cache() -> ModelCache:
    """Get the global model cache instance"""
    global _model_cache
    if _model_cache is None:
        # Use config for refresh interval
        refresh_interval = Config.CACHE_REFRESH_INTERVAL_MINUTES
        _model_cache = ModelCache(refresh_interval_minutes=refresh_interval)
    return _model_cache

async def initialize_model_cache() -> bool:
    """Initialize the global model cache"""
    cache = get_model_cache()
    return await cache.initialize()

def get_cached_production_models() -> List[Dict]:
    """Get production models from cache"""
    return get_model_cache().get_production_models()

def get_cached_staging_models() -> List[Dict]:
    """Get staging models from cache"""
    return get_model_cache().get_staging_models()

def get_cached_all_models() -> List[Dict]:
    """Get all models from cache"""
    return get_model_cache().get_all_models()

def get_cached_models_by_stage(stage: str) -> List[Dict]:
    """Get models by stage from cache"""
    return get_model_cache().get_models_by_stage(stage)