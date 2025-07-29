"""
File watcher for monitoring directories and detecting changes.
"""

import hashlib
import time
import threading
from pathlib import Path
from typing import Dict, List, Set, Callable, Optional, Any
from datetime import datetime
from dataclasses import dataclass

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent, FileModifiedEvent, FileDeletedEvent

from schemas.base import FileMetadata
from cli.utils.logging import gola_logger

@dataclass
class FileEvent:
    """File system event."""
    event_type: str  # 'created', 'modified', 'deleted'
    file_path: Path
    timestamp: datetime
    file_sha256: Optional[str] = None
    size_bytes: Optional[int] = None

class FileWatcher(FileSystemEventHandler):
    """File system watcher for Gola ingestion."""
    
    def __init__(self, watch_paths: List[Path], 
                 supported_extensions: Optional[List[str]] = None,
                 exclude_patterns: Optional[List[str]] = None):
        """
        Initialize file watcher.
        
        Args:
            watch_paths: Paths to watch for changes
            supported_extensions: File extensions to process (None for all)
            exclude_patterns: Patterns to exclude (e.g., ['*.tmp', '*.log'])
        """
        self.watch_paths = [Path(p) for p in watch_paths]
        self.supported_extensions = supported_extensions or [
            '.txt', '.md', '.pdf', '.docx', '.html', '.json', '.csv', '.xml'
        ]
        self.exclude_patterns = exclude_patterns or []
        
        self.observer = Observer()
        self.is_watching = False
        self.file_cache: Dict[str, FileMetadata] = {}
        self.event_callbacks: List[Callable[[FileEvent], None]] = []
        self.processed_files: Set[str] = set()
        
        # Threading
        self.lock = threading.Lock()
        self.event_queue: List[FileEvent] = []
        self.processing_thread: Optional[threading.Thread] = None
        
        gola_logger.info(f"File watcher initialized for paths: {watch_paths}")
    
    def add_callback(self, callback: Callable[[FileEvent], None]) -> None:
        """
        Add callback for file events.
        
        Args:
            callback: Function to call when file events occur
        """
        self.event_callbacks.append(callback)
    
    def start_watching(self) -> None:
        """Start watching for file changes."""
        if self.is_watching:
            gola_logger.warning("File watcher already started")
            return
        
        # Schedule observers for each path
        for watch_path in self.watch_paths:
            if not watch_path.exists():
                gola_logger.warning(f"Watch path does not exist: {watch_path}")
                continue
            
            self.observer.schedule(self, str(watch_path), recursive=True)
            gola_logger.info(f"Watching directory: {watch_path}")
        
        # Start observer
        self.observer.start()
        self.is_watching = True
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._process_events, daemon=True)
        self.processing_thread.start()
        
        gola_logger.info("File watcher started")
    
    def stop_watching(self) -> None:
        """Stop watching for file changes."""
        if not self.is_watching:
            return
        
        self.is_watching = False
        self.observer.stop()
        self.observer.join()
        
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)
        
        gola_logger.info("File watcher stopped")
    
    def on_created(self, event):
        """Handle file creation events."""
        if not event.is_directory:
            self._handle_file_event('created', Path(event.src_path))
    
    def on_modified(self, event):
        """Handle file modification events."""
        if not event.is_directory:
            self._handle_file_event('modified', Path(event.src_path))
    
    def on_deleted(self, event):
        """Handle file deletion events."""
        if not event.is_directory:
            self._handle_file_event('deleted', Path(event.src_path))
    
    def _handle_file_event(self, event_type: str, file_path: Path) -> None:
        """
        Handle file system event.
        
        Args:
            event_type: Type of event ('created', 'modified', 'deleted')
            file_path: Path to the file
        """
        # Check if file should be processed
        if not self._should_process_file(file_path):
            return
        
        # Create file event
        file_event = FileEvent(
            event_type=event_type,
            file_path=file_path,
            timestamp=datetime.utcnow()
        )
        
        # For created/modified events, get file metadata
        if event_type in ['created', 'modified']:
            try:
                file_metadata = self._get_file_metadata(file_path)
                if file_metadata:
                    file_event.file_sha256 = file_metadata.file_sha256
                    file_event.size_bytes = file_metadata.size_bytes
            except Exception as e:
                gola_logger.error(f"Error getting file metadata for {file_path}: {e}")
        
        # Add to event queue
        with self.lock:
            self.event_queue.append(file_event)
    
    def _should_process_file(self, file_path: Path) -> bool:
        """
        Check if file should be processed.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if file should be processed
        """
        # Check file extension
        if file_path.suffix.lower() not in self.supported_extensions:
            return False
        
        # Check exclude patterns
        for pattern in self.exclude_patterns:
            if file_path.match(pattern):
                return False
        
        # Check if file is accessible
        try:
            if not file_path.exists() or not file_path.is_file():
                return False
        except Exception:
            return False
        
        return True
    
    def _get_file_metadata(self, file_path: Path) -> Optional[FileMetadata]:
        """
        Get file metadata.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File metadata or None if error
        """
        try:
            stat = file_path.stat()
            
            # Calculate SHA256 hash
            sha256_hash = hashlib.sha256()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            
            # Check if already processed
            file_hash = sha256_hash.hexdigest()
            if file_hash in self.processed_files:
                return None
            
            # Detect MIME type
            mime_type = self._detect_mime_type(file_path)
            
            # Detect language (placeholder)
            language = self._detect_language(file_path)
            
            # Detect PII level (placeholder)
            pii_level = self._detect_pii_level(file_path)
            
            metadata = FileMetadata(
                path=file_path,
                file_sha256=file_hash,
                size_bytes=stat.st_size,
                mtime_utc=datetime.fromtimestamp(stat.st_mtime),
                mime_type=mime_type,
                language=language,
                pii_level=pii_level
            )
            
            # Cache metadata
            self.file_cache[file_hash] = metadata
            
            return metadata
        
        except Exception as e:
            gola_logger.error(f"Error getting metadata for {file_path}: {e}")
            return None
    
    def _detect_mime_type(self, file_path: Path) -> Optional[str]:
        """
        Detect MIME type of file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            MIME type or None
        """
        extension_map = {
            '.txt': 'text/plain',
            '.md': 'text/markdown',
            '.pdf': 'application/pdf',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.html': 'text/html',
            '.json': 'application/json',
            '.csv': 'text/csv',
            '.xml': 'application/xml'
        }
        
        return extension_map.get(file_path.suffix.lower())
    
    def _detect_language(self, file_path: Path) -> Optional[str]:
        """
        Detect language of file content.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Language code or None
        """
        # Placeholder implementation
        # In production, use language detection library
        return None
    
    def _detect_pii_level(self, file_path: Path) -> int:
        """
        Detect PII sensitivity level.
        
        Args:
            file_path: Path to the file
            
        Returns:
            PII level (0-3, where 0 is no PII, 3 is highly sensitive)
        """
        # Placeholder implementation
        # In production, use PII detection library
        return 0
    
    def _process_events(self) -> None:
        """Process file events in background thread."""
        while self.is_watching:
            events_to_process = []
            
            # Get events from queue
            with self.lock:
                if self.event_queue:
                    events_to_process = self.event_queue.copy()
                    self.event_queue.clear()
            
            # Process events
            for event in events_to_process:
                try:
                    # Call callbacks
                    for callback in self.event_callbacks:
                        try:
                            callback(event)
                        except Exception as e:
                            gola_logger.error(f"Error in file event callback: {e}")
                    
                    # Mark as processed
                    if event.file_sha256:
                        self.processed_files.add(event.file_sha256)
                    
                    gola_logger.info(f"Processed file event: {event.event_type} - {event.file_path}")
                
                except Exception as e:
                    gola_logger.error(f"Error processing file event: {e}")
            
            # Sleep between processing cycles
            time.sleep(1.0)
    
    def scan_existing_files(self) -> List[FileMetadata]:
        """
        Scan for existing files in watch paths.
        
        Returns:
            List of file metadata for existing files
        """
        existing_files = []
        
        for watch_path in self.watch_paths:
            if not watch_path.exists():
                continue
            
            for file_path in watch_path.rglob('*'):
                if file_path.is_file() and self._should_process_file(file_path):
                    metadata = self._get_file_metadata(file_path)
                    if metadata:
                        existing_files.append(metadata)
        
        gola_logger.info(f"Found {len(existing_files)} existing files")
        return existing_files
    
    def get_file_metadata(self, file_hash: str) -> Optional[FileMetadata]:
        """
        Get cached file metadata.
        
        Args:
            file_hash: File SHA256 hash
            
        Returns:
            File metadata or None if not found
        """
        return self.file_cache.get(file_hash)
    
    def is_file_processed(self, file_hash: str) -> bool:
        """
        Check if file has been processed.
        
        Args:
            file_hash: File SHA256 hash
            
        Returns:
            True if file has been processed
        """
        return file_hash in self.processed_files
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get watcher statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            "is_watching": self.is_watching,
            "watch_paths": [str(p) for p in self.watch_paths],
            "supported_extensions": self.supported_extensions,
            "exclude_patterns": self.exclude_patterns,
            "cached_files": len(self.file_cache),
            "processed_files": len(self.processed_files),
            "pending_events": len(self.event_queue)
        }
    
    def __del__(self):
        """Cleanup on destruction."""
        if hasattr(self, 'is_watching') and self.is_watching:
            self.stop_watching() 