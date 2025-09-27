"""
File management utilities for Arabic STT Platform
"""

import os
import json
import shutil
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
import sqlite3
import uuid
from werkzeug.utils import secure_filename

# Encryption support
try:
    from cryptography.fernet import Fernet
    ENCRYPTION_AVAILABLE = True
except ImportError:
    ENCRYPTION_AVAILABLE = False

import base64

from config import Config
from core.time_estimation_engine import TimeEstimationEngine

logger = logging.getLogger(__name__)

class FileManager:
    """Manages file operations and job tracking."""
    
    def __init__(self):
        self.config = Config()
        self.db_path = self.config.BASE_DIR / 'jobs.db'
        self.encryption_key_path = self.config.BASE_DIR / '.key'
        self.time_estimator = TimeEstimationEngine()
        self._init_encryption()
        self._init_database()
    
    def _init_encryption(self):
        """Initialize encryption for sensitive data."""
        try:
            if ENCRYPTION_AVAILABLE:
                from cryptography.fernet import Fernet  # Import here to avoid issues
                if self.encryption_key_path.exists():
                    with open(self.encryption_key_path, 'rb') as f:
                        key = f.read()
                else:
                    key = Fernet.generate_key()
                    os.makedirs(self.config.BASE_DIR, exist_ok=True)
                    with open(self.encryption_key_path, 'wb') as f:
                        f.write(key)
                    # Make key file readable only by owner
                    os.chmod(self.encryption_key_path, 0o600)
                    logger.info("Generated new encryption key for settings")
                
                self.cipher = Fernet(key)
            else:
                self.cipher = None
                logger.warning("Cryptography package not available - using base64 encoding")
        except Exception as e:
            logger.error(f"Failed to initialize encryption: {str(e)}")
            self.cipher = None
    
    def _init_database(self):
        """Initialize SQLite database for job tracking."""
        try:
            os.makedirs(self.config.BASE_DIR, exist_ok=True)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS jobs (
                        job_id TEXT PRIMARY KEY,
                        original_filename TEXT,
                        upload_path TEXT,
                        status TEXT,
                        message TEXT,
                        created_at TEXT,
                        updated_at TEXT,
                        completed_at TEXT,
                        options TEXT,
                        results TEXT,
                        processing_info TEXT
                    )
                ''')
                
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS job_files (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        job_id TEXT,
                        file_type TEXT,
                        file_path TEXT,
                        created_at TEXT,
                        FOREIGN KEY (job_id) REFERENCES jobs (job_id)
                    )
                ''')
                
                # Add processing_info column if it doesn't exist
                try:
                    conn.execute('ALTER TABLE jobs ADD COLUMN processing_info TEXT')
                    logger.info("Added processing_info column to jobs table")
                except sqlite3.OperationalError:
                    # Column already exists
                    pass
                
                # New settings table for persistent configuration
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS settings (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        key TEXT UNIQUE NOT NULL,
                        value TEXT,
                        encrypted BOOLEAN DEFAULT FALSE,
                        description TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create uploaded_files table for file management system
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS uploaded_files (
                        id TEXT PRIMARY KEY,
                        filename TEXT NOT NULL,
                        upload_path TEXT NOT NULL,
                        size INTEGER,
                        status TEXT DEFAULT 'uploaded',
                        job_id TEXT,
                        message TEXT,
                        created_at TEXT,
                        updated_at TEXT,
                        completed_at TEXT,
                        duration REAL,
                        processing_time REAL
                    )
                ''')
                
                conn.commit()
                logger.info("Database initialized successfully with settings and uploaded_files tables")
                
        except Exception as e:
            logger.error(f"Database initialization failed: {str(e)}")
            raise
    
    def save_upload(self, file, job_id: str, filename: str) -> str:
        """Save uploaded file and create job record."""
        try:
            # Ensure upload directory exists
            os.makedirs(self.config.UPLOAD_FOLDER, exist_ok=True)
            
            # Generate secure filename
            secure_name = secure_filename(filename)
            file_ext = Path(secure_name).suffix
            upload_filename = f"{job_id}_{secure_name}"
            upload_path = self.config.UPLOAD_FOLDER / upload_filename
            
            # Save file
            file.save(str(upload_path))
            
            # Create job record
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO jobs (
                        job_id, original_filename, upload_path, status, 
                        message, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    job_id,
                    filename,
                    str(upload_path),
                    'uploaded',
                    'File uploaded successfully',
                    datetime.now().isoformat(),
                    datetime.now().isoformat()
                ))
                conn.commit()
            
            logger.info(f"File saved: {upload_path}")
            return str(upload_path)
            
        except Exception as e:
            logger.error(f"File upload failed: {str(e)}")
            raise
    
    def update_job_status(self, job_id: str, status: str, message: str = "", processing_info: Optional[Dict[str, Any]] = None):
        """Update job status in database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Convert processing_info to JSON string if provided
                processing_info_json = json.dumps(processing_info) if processing_info else None
                
                if status == 'completed':
                    if processing_info_json:
                        conn.execute('''
                            UPDATE jobs 
                            SET status = ?, message = ?, updated_at = ?, completed_at = ?, processing_info = ?
                            WHERE job_id = ?
                        ''', [status, message, datetime.now().isoformat(), datetime.now().isoformat(), processing_info_json, job_id])
                    else:
                        conn.execute('''
                            UPDATE jobs 
                            SET status = ?, message = ?, updated_at = ?, completed_at = ?
                            WHERE job_id = ?
                        ''', [status, message, datetime.now().isoformat(), datetime.now().isoformat(), job_id])
                else:
                    if processing_info_json:
                        conn.execute('''
                            UPDATE jobs 
                            SET status = ?, message = ?, updated_at = ?, processing_info = ?
                            WHERE job_id = ?
                        ''', [status, message, datetime.now().isoformat(), processing_info_json, job_id])
                    else:
                        conn.execute('''
                            UPDATE jobs 
                            SET status = ?, message = ?, updated_at = ?
                            WHERE job_id = ?
                        ''', [status, message, datetime.now().isoformat(), job_id])
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Status update failed for {job_id}: {str(e)}")
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job status and information."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute('''
                    SELECT * FROM jobs WHERE job_id = ?
                ''', (job_id,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                # Get associated files
                file_cursor = conn.execute('''
                    SELECT file_type, file_path FROM job_files 
                    WHERE job_id = ?
                ''', (job_id,))
                
                files = {}
                for file_row in file_cursor.fetchall():
                    files[file_row['file_type']] = file_row['file_path']
                
                return {
                    'job_id': row['job_id'],
                    'original_filename': row['original_filename'],
                    'status': row['status'],
                    'message': row['message'],
                    'created_at': row['created_at'],
                    'updated_at': row['updated_at'],
                    'completed_at': row['completed_at'],
                    'files': files,
                    'options': json.loads(row['options']) if row['options'] else {},
                    'results': json.loads(row['results']) if row['results'] else {},
                    'processing_info': json.loads(row['processing_info']) if row['processing_info'] else {}
                }
                
        except Exception as e:
            logger.error(f"Failed to get job status for {job_id}: {str(e)}")
            return None
    
    def save_job_result(self, job_id: str, result_data: Dict[str, Any]):
        """Save job results and register output files."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Update job with results
                conn.execute('''
                    UPDATE jobs 
                    SET results = ?, options = ?
                    WHERE job_id = ?
                ''', (
                    json.dumps(result_data.get('results', {})),
                    json.dumps(result_data.get('options', {})),
                    job_id
                ))
                
                # Register output files
                for file_type, file_path in result_data.get('results', {}).items():
                    conn.execute('''
                        INSERT INTO job_files (job_id, file_type, file_path, created_at)
                        VALUES (?, ?, ?, ?)
                    ''', (
                        job_id,
                        file_type,
                        file_path,
                        datetime.now().isoformat()
                    ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to save job result for {job_id}: {str(e)}")
    
    def get_result_path(self, job_id: str, format: str) -> Optional[str]:
        """Get path to result file."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT file_path FROM job_files 
                    WHERE job_id = ? AND file_type = ?
                ''', (job_id, format))
                
                row = cursor.fetchone()
                return row[0] if row else None
                
        except Exception as e:
            logger.error(f"Failed to get result path for {job_id}.{format}: {str(e)}")
            return None
    
    def get_processing_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get processing history."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute('''
                    SELECT job_id, original_filename, status, created_at, 
                           completed_at, message
                    FROM jobs 
                    ORDER BY created_at DESC 
                    LIMIT ?
                ''', (limit,))
                
                history = []
                for row in cursor.fetchall():
                    history.append({
                        'job_id': row['job_id'],
                        'original_filename': row['original_filename'],
                        'status': row['status'],
                        'created_at': row['created_at'],
                        'completed_at': row['completed_at'],
                        'message': row['message']
                    })
                
                return history
                
        except Exception as e:
            logger.error(f"Failed to get processing history: {str(e)}")
            return []
    
    def cleanup_old_jobs(self, days_old: int = 30):
        """Clean up old jobs and associated files."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_old)
            
            with sqlite3.connect(self.db_path) as conn:
                # Get old jobs
                cursor = conn.execute('''
                    SELECT job_id, upload_path FROM jobs 
                    WHERE created_at < ? AND status IN ('completed', 'failed')
                ''', (cutoff_date.isoformat(),))
                
                old_jobs = cursor.fetchall()
                
                for job_id, upload_path in old_jobs:
                    # Get associated files
                    file_cursor = conn.execute('''
                        SELECT file_path FROM job_files WHERE job_id = ?
                    ''', (job_id,))
                    
                    files_to_delete = [upload_path]
                    files_to_delete.extend([row[0] for row in file_cursor.fetchall()])
                    
                    # Delete files
                    for file_path in files_to_delete:
                        try:
                            if file_path and Path(file_path).exists():
                                Path(file_path).unlink()
                        except Exception as e:
                            logger.warning(f"Failed to delete file {file_path}: {str(e)}")
                    
                    # Delete database records
                    conn.execute('DELETE FROM job_files WHERE job_id = ?', (job_id,))
                    conn.execute('DELETE FROM jobs WHERE job_id = ?', (job_id,))
                
                conn.commit()
                logger.info(f"Cleaned up {len(old_jobs)} old jobs")
                
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage usage statistics."""
        try:
            stats = {
                'total_jobs': 0,
                'active_jobs': 0,
                'completed_jobs': 0,
                'failed_jobs': 0,
                'storage_used_mb': 0,
                'uploads_size_mb': 0,
                'results_size_mb': 0
            }
            
            with sqlite3.connect(self.db_path) as conn:
                # Job counts
                cursor = conn.execute('SELECT status, COUNT(*) FROM jobs GROUP BY status')
                for status, count in cursor.fetchall():
                    stats[f'{status}_jobs'] = count
                    stats['total_jobs'] += count
            
            # Storage usage
            def get_directory_size(path: Path) -> int:
                if not path.exists():
                    return 0
                return sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
            
            uploads_size = get_directory_size(self.config.UPLOAD_FOLDER)
            results_size = get_directory_size(self.config.RESULTS_FOLDER)
            
            stats['uploads_size_mb'] = round(uploads_size / (1024 * 1024), 2)
            stats['results_size_mb'] = round(results_size / (1024 * 1024), 2)
            stats['storage_used_mb'] = stats['uploads_size_mb'] + stats['results_size_mb']
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get storage stats: {str(e)}")
            return {}
    
    def validate_job_id(self, job_id: str) -> bool:
        """Validate job ID format and existence."""
        try:
            # Check format (should be UUID)
            uuid.UUID(job_id)
            
            # Check existence in database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('SELECT 1 FROM jobs WHERE job_id = ?', (job_id,))
                return cursor.fetchone() is not None
                
        except (ValueError, Exception):
            return False
    
    def ensure_directories(self):
        """Ensure all required directories exist."""
        directories = [
            self.config.UPLOAD_FOLDER,
            self.config.RESULTS_FOLDER,
            self.config.TEMP_FOLDER,
            self.config.MODELS_FOLDER,
            self.config.LOG_FILE.parent
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logger.debug(f"Ensured directory exists: {directory}")
    
    def get_job_files(self, job_id: str) -> List[Dict[str, str]]:
        """Get all files associated with a job."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute('''
                    SELECT file_type, file_path, created_at 
                    FROM job_files 
                    WHERE job_id = ?
                    ORDER BY created_at
                ''', (job_id,))
                
                return [
                    {
                        'type': row['file_type'],
                        'path': row['file_path'],
                        'created_at': row['created_at'],
                        'exists': Path(row['file_path']).exists() if row['file_path'] else False
                    }
                    for row in cursor.fetchall()
                ]
                
        except Exception as e:
            logger.error(f"Failed to get job files for {job_id}: {str(e)}")
            return []

    def _encrypt_value(self, value: str) -> str:
        """Encrypt sensitive value."""
        if not value:
            return value
        
        try:
            if self.cipher and ENCRYPTION_AVAILABLE:
                return self.cipher.encrypt(value.encode()).decode()
            else:
                # Fallback encoding
                return base64.b64encode(value.encode()).decode()
        except Exception as e:
            logger.error(f"Encryption failed: {str(e)}")
            return value  # Return plain text as fallback
    
    def _decrypt_value(self, encrypted_value: str) -> str:
        """Decrypt sensitive value."""
        if not encrypted_value:
            return encrypted_value
        
        try:
            if self.cipher and ENCRYPTION_AVAILABLE:
                return self.cipher.decrypt(encrypted_value.encode()).decode()
            else:
                # Fallback decoding
                return base64.b64decode(encrypted_value.encode()).decode()
        except Exception as e:
            logger.error(f"Decryption failed: {str(e)}")
            return encrypted_value  # Return as-is if decryption fails
    
    def save_setting(self, key: str, value: str, encrypted: bool = False, description: Optional[str] = None) -> bool:
        """Save a setting to the database."""
        try:
            processed_value = self._encrypt_value(value) if encrypted else value
            now = datetime.now().isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO settings 
                    (key, value, encrypted, description, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                ''', (key, processed_value, encrypted, description, now))
                conn.commit()
            
            logger.info(f"Setting '{key}' saved successfully (encrypted: {encrypted})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save setting '{key}': {str(e)}")
            return False
    
    def get_setting(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get a setting from the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    'SELECT value, encrypted FROM settings WHERE key = ?', 
                    (key,)
                )
                row = cursor.fetchone()
                
                if row:
                    value, is_encrypted = row
                    if is_encrypted:
                        return self._decrypt_value(value)
                    return value
                
                return default
                
        except Exception as e:
            logger.error(f"Failed to get setting '{key}': {str(e)}")
            return default
    
    def get_all_settings(self) -> Dict[str, Any]:
        """Get all settings from the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT key, value, encrypted, description, updated_at 
                    FROM settings 
                    ORDER BY key
                ''')
                
                settings = {}
                for row in cursor.fetchall():
                    key, value, is_encrypted, description, updated_at = row
                    
                    # Decrypt sensitive values
                    if is_encrypted:
                        try:
                            decrypted_value = self._decrypt_value(value)
                            # Don't expose sensitive values in full
                            if 'api_key' in key.lower() or 'secret' in key.lower():
                                settings[key] = {
                                    'value': f"{decrypted_value[:10]}...{decrypted_value[-4:]}" if len(decrypted_value) > 14 else "***",
                                    'configured': bool(decrypted_value),
                                    'description': description,
                                    'updated_at': updated_at,
                                    'encrypted': True
                                }
                            else:
                                settings[key] = {
                                    'value': decrypted_value,
                                    'description': description,
                                    'updated_at': updated_at,
                                    'encrypted': True
                                }
                        except Exception as e:
                            logger.error(f"Failed to decrypt setting '{key}': {str(e)}")
                            settings[key] = {
                                'value': None,
                                'configured': False,
                                'description': description,
                                'updated_at': updated_at,
                                'encrypted': True,
                                'error': 'Decryption failed'
                            }
                    else:
                        settings[key] = {
                            'value': value,
                            'description': description,
                            'updated_at': updated_at,
                            'encrypted': False
                        }
                
                return settings
                
        except Exception as e:
            logger.error(f"Failed to get all settings: {str(e)}")
            return {}
    
    def delete_setting(self, key: str) -> bool:
        """Delete a setting from the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('DELETE FROM settings WHERE key = ?', (key,))
                conn.commit()
                
                deleted = cursor.rowcount > 0
                if deleted:
                    logger.info(f"Setting '{key}' deleted successfully")
                else:
                    logger.warning(f"Setting '{key}' not found for deletion")
                
                return deleted
                
        except Exception as e:
            logger.error(f"Failed to delete setting '{key}': {str(e)}")
            return False

    async def estimate_processing_time(
        self,
        audio_path: Optional[str] = None,
        file_duration: Optional[float] = None,
        file_size: Optional[int] = None,
        model_size: str = 'medium',
        processing_mode: str = 'local',
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Estimate processing time for a transcription job.
        
        Args:
            audio_path: Path to audio file
            file_duration: Duration in seconds (if known)
            file_size: File size in bytes (if known)
            model_size: Whisper model size
            processing_mode: 'local' or 'api'
            options: Processing options (multi_speaker, noise_reduction, etc.)
            
        Returns:
            Dict with time estimates and metadata
        """
        return await self.time_estimator.estimate_processing_time(
            audio_path=audio_path,
            file_duration=file_duration,
            file_size=file_size,
            model_size=model_size,
            processing_mode=processing_mode,
            options=options
        )

    async def record_processing_time(
        self,
        job_id: str,
        estimated_time: float,
        actual_time: float,
        model_size: str,
        processing_mode: str,
        file_duration: float,
        file_size: int,
        options: Dict[str, Any]
    ):
        """Record actual processing time for learning and improvement."""
        await self.time_estimator.record_actual_time(
            job_id=job_id,
            estimated_time=estimated_time,
            actual_time=actual_time,
            model_size=model_size,
            processing_mode=processing_mode,
            file_duration=file_duration,
            file_size=file_size,
            options=options
        )

    async def get_time_estimation_stats(self) -> Dict[str, Any]:
        """Get time estimation performance statistics."""
        return await self.time_estimator.get_performance_stats()

    def save_uploaded_file(self, file, file_id: str, filename: str) -> str:
        """Save uploaded file for later processing."""
        try:
            # Ensure upload directory exists
            os.makedirs(self.config.UPLOAD_FOLDER, exist_ok=True)
            
            # Generate secure filename
            secure_name = secure_filename(filename)
            upload_filename = f"{file_id}_{secure_name}"
            upload_path = self.config.UPLOAD_FOLDER / upload_filename
            
            # Save file
            file.save(str(upload_path))
            
            # Store file info in database
            with sqlite3.connect(self.db_path) as conn:
                # Create uploaded_files table if not exists
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS uploaded_files (
                        id TEXT PRIMARY KEY,
                        filename TEXT NOT NULL,
                        upload_path TEXT NOT NULL,
                        size INTEGER,
                        status TEXT DEFAULT 'uploaded',
                        job_id TEXT,
                        message TEXT,
                        created_at TEXT,
                        updated_at TEXT,
                        completed_at TEXT,
                        duration REAL,
                        processing_time REAL
                    )
                ''')
                
                # Get file size
                file_size = Path(upload_path).stat().st_size
                
                conn.execute('''
                    INSERT INTO uploaded_files (
                        id, filename, upload_path, size, status,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    file_id,
                    filename,
                    str(upload_path),
                    file_size,
                    'uploaded',
                    datetime.now().isoformat(),
                    datetime.now().isoformat()
                ))
                conn.commit()
            
            logger.info(f"File uploaded: {upload_path}")
            return str(upload_path)
            
        except Exception as e:
            logger.error(f"File upload failed: {str(e)}")
            raise

    def get_uploaded_files(self) -> List[Dict[str, Any]]:
        """Get list of uploaded files."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute('''
                    SELECT id, filename, size, status, job_id, message,
                           created_at, updated_at, completed_at, duration, processing_time
                    FROM uploaded_files 
                    ORDER BY created_at DESC
                ''')
                
                files = []
                for row in cursor.fetchall():
                    files.append({
                        'id': row['id'],
                        'filename': row['filename'],
                        'size': row['size'],
                        'status': row['status'],
                        'job_id': row['job_id'],
                        'message': row['message'],
                        'created_at': row['created_at'],
                        'updated_at': row['updated_at'],
                        'completed_at': row['completed_at'],
                        'duration': row['duration'],
                        'processing_time': row['processing_time']
                    })
                
                return files
                
        except Exception as e:
            logger.error(f"Failed to get uploaded files: {str(e)}")
            return []

    def get_uploaded_file(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Get specific uploaded file info."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute('''
                    SELECT * FROM uploaded_files WHERE id = ?
                ''', (file_id,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                return {
                    'id': row['id'],
                    'filename': row['filename'],
                    'upload_path': row['upload_path'],
                    'size': row['size'],
                    'status': row['status'],
                    'job_id': row['job_id'],
                    'message': row['message'],
                    'created_at': row['created_at'],
                    'updated_at': row['updated_at'],
                    'completed_at': row['completed_at'],
                    'duration': row['duration'],
                    'processing_time': row['processing_time']
                }
                
        except Exception as e:
            logger.error(f"Failed to get uploaded file {file_id}: {str(e)}")
            return None

    def update_file_status(self, file_id: str, status: str, job_id: Optional[str] = None, 
                          message: Optional[str] = None, processing_time: Optional[float] = None):
        """Update uploaded file status."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                update_fields = ['status = ?', 'updated_at = ?']
                params: List[Any] = [status, datetime.now().isoformat()]
                
                if job_id is not None:
                    update_fields.append('job_id = ?')
                    params.append(job_id)
                
                if message is not None:
                    update_fields.append('message = ?')
                    params.append(message)
                
                if processing_time is not None:
                    update_fields.append('processing_time = ?')
                    params.append(processing_time)
                
                if status == 'completed':
                    update_fields.append('completed_at = ?')
                    params.append(datetime.now().isoformat())
                
                params.append(file_id)
                
                conn.execute(f'''
                    UPDATE uploaded_files 
                    SET {', '.join(update_fields)}
                    WHERE id = ?
                ''', params)
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to update file status for {file_id}: {str(e)}")

    def delete_uploaded_files(self, file_ids: List[str]) -> int:
        """Delete uploaded files."""
        deleted_count = 0
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                for file_id in file_ids:
                    # Get file info
                    cursor = conn.execute('''
                        SELECT upload_path FROM uploaded_files WHERE id = ?
                    ''', (file_id,))
                    
                    row = cursor.fetchone()
                    if row:
                        upload_path = row[0]
                        
                        # Delete physical file
                        try:
                            if upload_path and Path(upload_path).exists():
                                Path(upload_path).unlink()
                        except Exception as e:
                            logger.warning(f"Failed to delete file {upload_path}: {str(e)}")
                        
                        # Delete from database
                        conn.execute('DELETE FROM uploaded_files WHERE id = ?', (file_id,))
                        deleted_count += 1
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to delete files: {str(e)}")
        
        return deleted_count
