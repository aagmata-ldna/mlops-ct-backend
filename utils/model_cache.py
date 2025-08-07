"""
Model Cache Manager for MLOps Control Tower Backend

This module provides caching functionality for MLflow models to improve performance.
Models are loaded at startup and cached in memory with optional background refresh.
"""

import asyncio
import logging
import json
import os
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
        
        # Persistence settings
        self._cache_dir = Config.CACHE_DIRECTORY
        self._cache_file_path = os.path.join(self._cache_dir, Config.CACHE_FILE_NAME)
        self._enable_persistence = Config.ENABLE_CACHE_PERSISTENCE
        self._max_age_hours = Config.CACHE_MAX_AGE_HOURS
    
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
        First tries to load from persistent storage, then from MLflow if needed.
        Should be called at application startup.
        """
        logger.info("Initializing model cache...")
        
        # First try to load from persistent cache
        if self._enable_persistence:
            logger.info(f"🔍 Attempting to load from disk: {self._cache_file_path}")
            disk_result = await self._load_from_disk()
            logger.info(f"🔍 Disk load result: {disk_result}")
            if disk_result:
                logger.info(f"✅ Loaded {len(self._all_models)} models from persistent cache")
                logger.info(f"🔍 Production models after disk load: {len(self._models_by_stage.get('Production', []))}")
                # Start background refresh for future updates
                self._start_background_refresh()
                return True
            else:
                logger.warning("❌ Failed to load from persistent cache")
        
        # If persistent cache failed or disabled, fetch from MLflow
        success = await self._refresh_cache()
        
        if success:
            logger.info(f"Model cache initialized with {len(self._all_models)} models from MLflow")
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
            
            # Save to persistent storage if enabled
            if self._enable_persistence:
                await self._save_to_disk()
            
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
    
    async def _save_to_disk(self):
        """Save current cache to persistent storage"""
        if not self._enable_persistence:
            return
        
        try:
            # Create cache directory if it doesn't exist
            os.makedirs(self._cache_dir, exist_ok=True)
            
            # Prepare cache data for JSON serialization
            cache_data = {
                "metadata": {
                    "last_refresh": self._last_refresh.isoformat() if self._last_refresh else None,
                    "cached_at": datetime.now().isoformat(),
                    "total_models": len(self._all_models),
                    "cache_version": "1.0"
                },
                "models_by_stage": self._models_by_stage,
                "all_models": self._all_models,
                "models_by_name": self._models_by_name
            }
            
            # Write to temporary file first, then rename for atomicity
            temp_file_path = self._cache_file_path + ".tmp"
            with open(temp_file_path, 'w') as f:
                json.dump(cache_data, f, indent=2, default=str)
            
            # Atomic rename
            os.rename(temp_file_path, self._cache_file_path)
            logger.info(f"💾 Cache saved to {self._cache_file_path} ({len(self._all_models)} models)")
            
        except Exception as e:
            logger.error(f"Failed to save cache to disk: {str(e)}")
    
    async def _load_from_disk(self) -> bool:
        """Load cache from persistent storage if it exists and is fresh"""
        logger.info(f"🔍 _load_from_disk: persistence={self._enable_persistence}, file_exists={os.path.exists(self._cache_file_path)}")
        
        if not self._enable_persistence:
            logger.info("❌ Persistence disabled")
            return False
            
        if not os.path.exists(self._cache_file_path):
            logger.info(f"❌ Cache file not found: {self._cache_file_path}")
            return False
        
        try:
            # Check cache file age
            file_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(self._cache_file_path))
            logger.info(f"🔍 Cache file age: {file_age} (max age: {self._max_age_hours} hours)")
            
            if file_age > timedelta(hours=self._max_age_hours):
                logger.info(f"❌ Persistent cache is too old ({file_age}), will refresh from MLflow")
                return False
            
            # Load cache data
            logger.info("🔍 Reading cache file...")
            with open(self._cache_file_path, 'r') as f:
                cache_data = json.load(f)
            
            logger.info(f"🔍 Cache data keys: {list(cache_data.keys())}")
            
            # Validate cache structure
            required_keys = ["metadata", "models_by_stage", "all_models", "models_by_name"]
            missing_keys = [key for key in required_keys if key not in cache_data]
            
            if missing_keys:
                logger.warning(f"❌ Invalid cache file structure, missing keys: {missing_keys}")
                return False
            
            logger.info("✅ Cache file structure is valid")
            
            # Load data into memory
            self._models_by_stage = cache_data["models_by_stage"]
            self._all_models = cache_data["all_models"]
            self._models_by_name = cache_data["models_by_name"]
            
            # Restore metadata
            metadata = cache_data["metadata"]
            if metadata.get("last_refresh"):
                self._last_refresh = datetime.fromisoformat(metadata["last_refresh"])
            
            # Debug logging
            production_count = len(self._models_by_stage.get("Production", []))
            staging_count = len(self._models_by_stage.get("Staging", []))
            logger.info(f"📂 Loaded cache from disk: {metadata['total_models']} models (age: {file_age})")
            logger.info(f"📊 Cache breakdown: Production={production_count}, Staging={staging_count}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load cache from disk: {str(e)}")
            return False
    
    def _clear_persistent_cache(self):
        """Clear persistent cache file"""
        if os.path.exists(self._cache_file_path):
            try:
                os.remove(self._cache_file_path)
                logger.info("🗑️ Persistent cache file cleared")
            except Exception as e:
                logger.error(f"Failed to clear persistent cache: {str(e)}")
    
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
        # Check persistent cache file info
        cache_file_exists = os.path.exists(self._cache_file_path)
        cache_file_age = None
        cache_file_size = None
        
        if cache_file_exists:
            try:
                file_stat = os.stat(self._cache_file_path)
                cache_file_age = (datetime.now() - datetime.fromtimestamp(file_stat.st_mtime)).total_seconds() / 3600  # hours
                cache_file_size = file_stat.st_size
            except Exception:
                pass
        
        return {
            "total_models": len(self._all_models),
            "models_by_stage": {stage: len(models) for stage, models in self._models_by_stage.items()},
            "last_refresh": self._last_refresh.isoformat() if self._last_refresh else None,
            "is_refreshing": self._is_refreshing,
            "cache_age_minutes": (
                (datetime.now() - self._last_refresh).total_seconds() / 60 
                if self._last_refresh else None
            ),
            "is_stale": self.is_cache_stale(),
            "persistence": {
                "enabled": self._enable_persistence,
                "cache_directory": self._cache_dir,
                "cache_file_exists": cache_file_exists,
                "cache_file_age_hours": round(cache_file_age, 2) if cache_file_age else None,
                "cache_file_size_bytes": cache_file_size,
                "max_age_hours": self._max_age_hours
            }
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
                    # Use shorter sleep intervals to make cancellation more responsive
                    sleep_time = self.refresh_interval
                    while sleep_time > 0 and not self._should_stop_refresh:
                        # Sleep in 10-second intervals to check for stop signal
                        interval = min(10, sleep_time)
                        await asyncio.sleep(interval)
                        sleep_time -= interval
                    
                    if not self._should_stop_refresh:
                        await self._refresh_cache()
                except asyncio.CancelledError:
                    logger.info("Background refresh task cancelled")
                    break
                except Exception as e:
                    logger.error(f"Background refresh failed: {str(e)}")
            
            logger.info("Background refresh task stopped")
        
        loop = asyncio.get_event_loop()
        self._refresh_task = loop.create_task(background_refresh())
        logger.info(f"Background refresh started (interval: {self.refresh_interval/60:.1f} minutes)")
    
    def stop_background_refresh(self):
        """Stop background refresh task"""
        self._should_stop_refresh = True
        if self._refresh_task and not self._refresh_task.done():
            self._refresh_task.cancel()
            logger.info("Background refresh task cancellation requested")
    
    async def stop_background_refresh_async(self):
        """Stop background refresh task and wait for it to complete"""
        self._should_stop_refresh = True
        if self._refresh_task and not self._refresh_task.done():
            self._refresh_task.cancel()
            try:
                await asyncio.wait_for(self._refresh_task, timeout=5.0)
                logger.info("✅ Background refresh task stopped gracefully")
            except asyncio.TimeoutError:
                logger.warning("⏰ Background refresh task didn't stop within timeout")
            except asyncio.CancelledError:
                logger.info("✅ Background refresh task cancelled successfully")
    
    def __del__(self):
        """Cleanup on destruction"""
        self.stop_background_refresh()

# Global cache instance
_model_cache: Optional[ModelCache] = None

def get_model_cache() -> ModelCache:
    """Get the global model cache instance"""
    global _model_cache
    if _model_cache is None:
        logger.info("🏗️ Creating new ModelCache instance (singleton)")
        # Use config for refresh interval
        refresh_interval = Config.CACHE_REFRESH_INTERVAL_MINUTES
        _model_cache = ModelCache(refresh_interval_minutes=refresh_interval)
    else:
        logger.debug(f"♻️ Reusing existing ModelCache instance (cache has {len(_model_cache._all_models)} models)")
    return _model_cache

async def initialize_model_cache() -> bool:
    """Initialize the global model cache"""
    logger.info("🔧 initialize_model_cache() called")
    cache = get_model_cache()
    logger.info(f"🔧 Cache instance ID: {id(cache)}")
    result = await cache.initialize()
    logger.info(f"🔧 Cache after init: {len(cache._all_models)} total models")
    return result

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