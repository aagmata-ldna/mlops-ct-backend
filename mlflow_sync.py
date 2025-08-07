import asyncio
import mlflow
import mlflow.tracking
from datetime import datetime, timedelta
import logging
from database import ModelDatabase
import os
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class MLflowSyncService:
    def __init__(self, sync_interval_minutes: int = 30):
        self.sync_interval = sync_interval_minutes * 60  # Convert to seconds
        self.db = ModelDatabase()
        self.is_running = False
        
        # Configure MLflow
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        if os.getenv("MLFLOW_TRACKING_USERNAME") and os.getenv("MLFLOW_TRACKING_PASSWORD"):
            os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
            os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")
    
    async def sync_models_from_mlflow(self):
        """Fetch models from MLflow and store in database"""
        try:
            logger.info("Starting MLflow sync...")
            client = mlflow.tracking.MlflowClient()
            
            # Fetch all registered models with max_results
            models = client.search_registered_models(max_results=1000)
            logger.info(f"Found {len(models)} registered models in MLflow")
            
            model_data = []
            processed = 0
            total_versions = sum(len(model.latest_versions) for model in models)
            
            for model in models:
                for version in model.latest_versions:
                    try:
                        processed += 1
                        
                        # Log progress every 50 models
                        if processed % 50 == 0:
                            logger.info(f"Processing model {processed}/{total_versions}: {model.name}")
                        
                        # Get run details for metrics (this is the slow part)
                        run = client.get_run(version.run_id)
                        
                        # Extract metrics
                        metrics = {}
                        if run.data.metrics:
                            metrics = {
                                'accuracy': run.data.metrics.get('accuracy'),
                                'precision': run.data.metrics.get('precision'),
                                'recall': run.data.metrics.get('recall'),
                                'f1_score': run.data.metrics.get('f1_score'),
                                'rmse': run.data.metrics.get('rmse'),
                                'mae': run.data.metrics.get('mae')
                            }
                        
                        model_info = {
                            'name': model.name,
                            'version': version.version,
                            'stage': version.current_stage,
                            'creation_timestamp': datetime.fromtimestamp(version.creation_timestamp / 1000).isoformat(),
                            'last_updated_timestamp': datetime.fromtimestamp(version.last_updated_timestamp / 1000).isoformat(),
                            'description': version.description,
                            'tags': version.tags or {},
                            'metrics': metrics,
                            'run_id': version.run_id
                        }
                        model_data.append(model_info)
                        
                        # Yield control every 10 models to prevent blocking
                        if processed % 10 == 0:
                            await asyncio.sleep(0.01)
                        
                    except Exception as e:
                        logger.error(f"Error processing model {model.name} version {version.version}: {str(e)}")
                        continue
            
            # Store in database
            logger.info(f"Storing {len(model_data)} model versions in database...")
            self.db.store_models(model_data)
            logger.info(f"Successfully synced {len(model_data)} model versions to database")
            
            return len(model_data)
            
        except Exception as e:
            logger.error(f"MLflow sync failed: {str(e)}")
            return 0
    
    async def start_background_sync(self):
        """Start the background sync process"""
        if self.is_running:
            logger.warning("Sync service is already running")
            return
        
        self.is_running = True
        logger.info(f"Starting MLflow sync service (interval: {self.sync_interval/60} minutes)")
        
        # Initial sync
        await self.sync_models_from_mlflow()
        
        # Background sync loop
        while self.is_running:
            try:
                await asyncio.sleep(self.sync_interval)
                if self.is_running:  # Check if still running after sleep
                    await self.sync_models_from_mlflow()
            except Exception as e:
                logger.error(f"Error in background sync: {str(e)}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying
    
    def stop_sync(self):
        """Stop the background sync process"""
        logger.info("Stopping MLflow sync service")
        self.is_running = False
    
    async def force_sync(self):
        """Manually trigger a sync"""
        logger.info("Manual sync triggered")
        return await self.sync_models_from_mlflow()

# Global sync service instance
sync_service = MLflowSyncService()

async def start_mlflow_sync():
    """Initialize and start the MLflow sync service"""
    await sync_service.start_background_sync()

def get_sync_service():
    """Get the global sync service instance"""
    return sync_service