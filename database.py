import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Any

class ModelDatabase:
    def __init__(self, db_path: str = "models_cache.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the SQLite database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create models table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                version TEXT NOT NULL,
                stage TEXT NOT NULL,
                creation_timestamp TEXT NOT NULL,
                last_updated_timestamp TEXT NOT NULL,
                description TEXT,
                tags TEXT,
                metrics TEXT,
                run_id TEXT NOT NULL,
                cached_at TEXT NOT NULL,
                UNIQUE(name, version)
            )
        ''')
        
        # Create cache metadata table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cache_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        ''')
        
        # Create monitored models table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS monitored_models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                model_version TEXT NOT NULL,
                model_type TEXT NOT NULL DEFAULT 'classifier',
                registered_at TEXT NOT NULL,
                registered_by TEXT,
                monitoring_config TEXT,
                is_active BOOLEAN DEFAULT 1,
                UNIQUE(model_name, model_version)
            )
        ''')
        
        # Add model_type column to existing tables if it doesn't exist
        try:
            cursor.execute('ALTER TABLE monitored_models ADD COLUMN model_type TEXT DEFAULT "classifier"')
        except sqlite3.OperationalError:
            # Column already exists
            pass
        
        conn.commit()
        conn.close()
    
    def store_models(self, models: List[Dict[str, Any]]):
        """Store models in the database, replacing existing data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Clear existing models
        cursor.execute('DELETE FROM models')
        
        # Insert new models
        for model in models:
            cursor.execute('''
                INSERT OR REPLACE INTO models 
                (name, version, stage, creation_timestamp, last_updated_timestamp, 
                 description, tags, metrics, run_id, cached_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                model['name'],
                model['version'],
                model['stage'],
                model['creation_timestamp'],
                model['last_updated_timestamp'],
                model.get('description'),
                json.dumps(model.get('tags', {})),
                json.dumps(model.get('metrics', {})),
                model['run_id'],
                datetime.now().isoformat()
            ))
        
        # Update cache metadata
        cursor.execute('''
            INSERT OR REPLACE INTO cache_metadata (key, value, updated_at)
            VALUES (?, ?, ?)
        ''', ('last_mlflow_sync', str(len(models)), datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
    
    def get_production_models(self) -> List[Dict[str, Any]]:
        """Get production models from cache"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT name, version, stage, creation_timestamp, last_updated_timestamp,
                   description, tags, metrics, run_id, cached_at
            FROM models 
            WHERE stage = 'Production'
            ORDER BY last_updated_timestamp DESC
        ''')
        
        models = []
        for row in cursor.fetchall():
            model = {
                'name': row[0],
                'version': row[1],
                'stage': row[2],
                'creation_timestamp': row[3],
                'last_updated_timestamp': row[4],
                'description': row[5],
                'tags': json.loads(row[6]) if row[6] else {},
                'metrics': json.loads(row[7]) if row[7] else {},
                'run_id': row[8],
                'cached_at': row[9]
            }
            models.append(model)
        
        conn.close()
        return models
    
    def get_all_models(self) -> List[Dict[str, Any]]:
        """Get all models from cache"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT name, version, stage, creation_timestamp, last_updated_timestamp,
                   description, tags, metrics, run_id, cached_at
            FROM models 
            ORDER BY last_updated_timestamp DESC
        ''')
        
        models = []
        for row in cursor.fetchall():
            model = {
                'name': row[0],
                'version': row[1],
                'stage': row[2],
                'creation_timestamp': row[3],
                'last_updated_timestamp': row[4],
                'description': row[5],
                'tags': json.loads(row[6]) if row[6] else {},
                'metrics': json.loads(row[7]) if row[7] else {},
                'run_id': row[8],
                'cached_at': row[9]
            }
            models.append(model)
        
        conn.close()
        return models
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache metadata information"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT key, value, updated_at FROM cache_metadata')
        metadata = {}
        for row in cursor.fetchall():
            metadata[row[0]] = {
                'value': row[1],
                'updated_at': row[2]
            }
        
        # Count models by stage
        cursor.execute('SELECT stage, COUNT(*) FROM models GROUP BY stage')
        stage_counts = dict(cursor.fetchall())
        
        conn.close()
        
        return {
            'metadata': metadata,
            'stage_counts': stage_counts,
            'total_models': sum(stage_counts.values())
        }
    
    def get_models_by_stage(self, stage: str) -> List[Dict[str, Any]]:
        """Get models filtered by stage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT name, version, stage, creation_timestamp, last_updated_timestamp,
                   description, tags, metrics, run_id, cached_at
            FROM models 
            WHERE stage = ?
            ORDER BY last_updated_timestamp DESC
        ''', (stage,))
        
        models = []
        for row in cursor.fetchall():
            model = {
                'name': row[0],
                'version': row[1],
                'stage': row[2],
                'creation_timestamp': row[3],
                'last_updated_timestamp': row[4],
                'description': row[5],
                'tags': json.loads(row[6]) if row[6] else {},
                'metrics': json.loads(row[7]) if row[7] else {},
                'run_id': row[8],
                'cached_at': row[9]
            }
            models.append(model)
        
        conn.close()
        return models
    
    def add_monitored_model(self, model_name: str, model_version: str, model_type: str = "classifier", 
                           registered_by: str = None, monitoring_config: Dict[str, Any] = None) -> bool:
        """Add a model to monitoring"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO monitored_models 
                (model_name, model_version, model_type, registered_at, registered_by, monitoring_config, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                model_name,
                model_version,
                model_type,
                datetime.now().isoformat(),
                registered_by,
                json.dumps(monitoring_config) if monitoring_config else None,
                True
            ))
            
            conn.commit()
            return True
        except Exception:
            return False
        finally:
            conn.close()
    
    def remove_monitored_model(self, model_name: str, model_version: str) -> bool:
        """Remove a model from monitoring"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                DELETE FROM monitored_models 
                WHERE model_name = ? AND model_version = ?
            ''', (model_name, model_version))
            
            conn.commit()
            return True
        except Exception:
            return False
        finally:
            conn.close()
    
    def get_monitored_models(self) -> List[Dict[str, Any]]:
        """Get all models being monitored with their full details"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT m.name, m.version, m.stage, m.creation_timestamp, m.last_updated_timestamp,
                   m.description, m.tags, m.metrics, m.run_id, m.cached_at,
                   mm.registered_at, mm.registered_by, mm.monitoring_config
            FROM models m
            JOIN monitored_models mm ON m.name = mm.model_name AND m.version = mm.model_version
            WHERE mm.is_active = 1
            ORDER BY mm.registered_at DESC
        ''')
        
        models = []
        for row in cursor.fetchall():
            model = {
                'name': row[0],
                'version': row[1],
                'stage': row[2],
                'creation_timestamp': row[3],
                'last_updated_timestamp': row[4],
                'description': row[5],
                'tags': json.loads(row[6]) if row[6] else {},
                'metrics': json.loads(row[7]) if row[7] else {},
                'run_id': row[8],
                'cached_at': row[9],
                'monitoring': {
                    'registered_at': row[10],
                    'registered_by': row[11],
                    'config': json.loads(row[12]) if row[12] else {}
                }
            }
            models.append(model)
        
        conn.close()
        return models
    
    def is_model_monitored(self, model_name: str, model_version: str) -> bool:
        """Check if a model is being monitored"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT COUNT(*) FROM monitored_models 
            WHERE model_name = ? AND model_version = ? AND is_active = 1
        ''', (model_name, model_version))
        
        count = cursor.fetchone()[0]
        conn.close()
        
        return count > 0
    
    def get_monitored_models_list(self) -> List[Dict[str, Any]]:
        """Get list of monitored models (just names and versions for real-time MLflow fetching)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT model_name, model_version, model_type, registered_at, registered_by, monitoring_config
            FROM monitored_models 
            WHERE is_active = 1
            ORDER BY registered_at DESC
        ''')
        
        models = []
        for row in cursor.fetchall():
            model = {
                'model_name': row[0],
                'model_version': row[1],
                'model_type': row[2] if row[2] else 'classifier',  # Default to classifier for existing records
                'registered_at': row[3],
                'registered_by': row[4],
                'monitoring_config': json.loads(row[5]) if row[5] else {}
            }
            models.append(model)
        
        conn.close()
        return models