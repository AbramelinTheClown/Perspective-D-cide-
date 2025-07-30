"""
Storage backend for the Perspective D<cide> framework.

Provides abstract storage interface and implementations for different storage backends.
"""

import os
import json
import sqlite3
from pathlib import Path
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

class StorageBackend(ABC):
    """Abstract base class for storage backends."""
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the storage backend."""
        pass
    
    @abstractmethod
    def store(self, key: str, data: Any) -> bool:
        """Store data with the given key."""
        pass
    
    @abstractmethod
    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve data for the given key."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete data for the given key."""
        pass
    
    @abstractmethod
    def list_keys(self, prefix: str = "") -> List[str]:
        """List all keys with optional prefix."""
        pass

class SQLiteBackend(StorageBackend):
    """SQLite-based storage backend."""
    
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = None
    
    def initialize(self) -> None:
        """Initialize SQLite database."""
        try:
            self.connection = sqlite3.connect(self.db_path)
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS storage (
                    key TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self.connection.commit()
            logger.info(f"SQLite storage initialized at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize SQLite storage: {e}")
            raise
    
    def store(self, key: str, data: Any) -> bool:
        """Store data in SQLite."""
        try:
            data_json = json.dumps(data)
            self.connection.execute("""
                INSERT OR REPLACE INTO storage (key, data, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            """, (key, data_json))
            self.connection.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to store data for key {key}: {e}")
            return False
    
    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve data from SQLite."""
        try:
            cursor = self.connection.execute(
                "SELECT data FROM storage WHERE key = ?", (key,)
            )
            result = cursor.fetchone()
            if result:
                return json.loads(result[0])
            return None
        except Exception as e:
            logger.error(f"Failed to retrieve data for key {key}: {e}")
            return None
    
    def delete(self, key: str) -> bool:
        """Delete data from SQLite."""
        try:
            self.connection.execute("DELETE FROM storage WHERE key = ?", (key,))
            self.connection.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to delete data for key {key}: {e}")
            return False
    
    def list_keys(self, prefix: str = "") -> List[str]:
        """List keys with optional prefix."""
        try:
            cursor = self.connection.execute(
                "SELECT key FROM storage WHERE key LIKE ?", (f"{prefix}%",)
            )
            return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to list keys with prefix {prefix}: {e}")
            return []
    
    def close(self) -> None:
        """Close the database connection."""
        if self.connection:
            self.connection.close()

class FileBackend(StorageBackend):
    """File-based storage backend."""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def initialize(self) -> None:
        """Initialize file storage."""
        logger.info(f"File storage initialized at {self.base_path}")
    
    def store(self, key: str, data: Any) -> bool:
        """Store data as JSON file."""
        try:
            file_path = self.base_path / f"{key}.json"
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Failed to store data for key {key}: {e}")
            return False
    
    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve data from JSON file."""
        try:
            file_path = self.base_path / f"{key}.json"
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return None
        except Exception as e:
            logger.error(f"Failed to retrieve data for key {key}: {e}")
            return None
    
    def delete(self, key: str) -> bool:
        """Delete JSON file."""
        try:
            file_path = self.base_path / f"{key}.json"
            if file_path.exists():
                file_path.unlink()
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete data for key {key}: {e}")
            return False
    
    def list_keys(self, prefix: str = "") -> List[str]:
        """List JSON files with optional prefix."""
        try:
            keys = []
            for file_path in self.base_path.rglob("*.json"):
                key = str(file_path.relative_to(self.base_path)).replace('.json', '')
                if key.startswith(prefix):
                    keys.append(key)
            return keys
        except Exception as e:
            logger.error(f"Failed to list keys with prefix {prefix}: {e}")
            return []

# Global storage instance
_storage_backend: Optional[StorageBackend] = None

def initialize(config: Dict[str, Any]) -> None:
    """Initialize the storage backend based on configuration."""
    global _storage_backend
    
    backend_type = config.get('storage_backend', 'sqlite')
    storage_path = config.get('storage_path', '~/.perspective_dcide')
    
    if backend_type == 'sqlite':
        db_path = Path(storage_path) / 'framework.db'
        _storage_backend = SQLiteBackend(str(db_path))
    elif backend_type == 'file':
        _storage_backend = FileBackend(storage_path)
    else:
        raise ValueError(f"Unsupported storage backend: {backend_type}")
    
    _storage_backend.initialize()

def get_backend() -> StorageBackend:
    """Get the global storage backend instance."""
    if _storage_backend is None:
        raise RuntimeError("Storage backend not initialized. Call initialize() first.")
    return _storage_backend 