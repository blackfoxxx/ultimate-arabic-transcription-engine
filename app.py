#!/usr/bin/env python3
"""
Arabic Speech-to-Text Platform
Self-hosted STT solution with audio enhancement and Whisper integration
"""

import os
import logging
import asyncio
import threading
import subprocess
import time
import gc
import psutil
import torch
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_file, redirect, url_for
from flask_cors import CORS
from werkzeug.utils import secure_filename
import uuid
from typing import Optional, Dict, Any, List
import json
import sqlite3
import signal
import sys

from core.audio_processor import AudioProcessor
from core.unified_transcription_service_v3 import UnifiedTranscriptionService
from core.enhanced_transcription_service import EnhancedTranscriptionService
from core.output_generator import OutputGenerator
from utils.file_manager import FileManager
from utils.settings_manager import SettingsManager
from utils.logger import setup_logger
from config import Config

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)
CORS(app)

# Initialize components
logger = setup_logger(__name__)
audio_processor = AudioProcessor()
transcription_service = UnifiedTranscriptionService()
enhanced_transcription_service = EnhancedTranscriptionService()
output_generator = OutputGenerator()
file_manager = FileManager()
settings_manager = SettingsManager(file_manager)

# Memory management globals
_last_memory_cleanup = time.time()
_memory_cleanup_lock = threading.Lock()

def cleanup_memory(force: bool = False):
    """Perform comprehensive memory cleanup to prevent crashes."""
    global _last_memory_cleanup
    current_time = time.time()
    
    # Only cleanup if enough time has passed or forced
    if not force and (current_time - _last_memory_cleanup) < 30:
        return
        
    with _memory_cleanup_lock:
        try:
            logger.debug("Performing memory cleanup...")
            
            # Force garbage collection
            gc.collect()
            
            # Clear GPU cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                # Log GPU memory usage
                try:
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory
                    gpu_allocated = torch.cuda.memory_allocated(0)
                    gpu_cached = torch.cuda.memory_reserved(0)
                    
                    logger.debug(f"GPU Memory - Total: {gpu_memory/1024**3:.1f}GB, "
                               f"Allocated: {gpu_allocated/1024**3:.1f}GB, "
                               f"Cached: {gpu_cached/1024**3:.1f}GB")
                except Exception:
                    pass
            
            # Log system memory usage
            try:
                process = psutil.Process()
                memory_info = process.memory_info()
                logger.debug(f"System Memory - RSS: {memory_info.rss/1024**2:.1f}MB, "
                           f"VMS: {memory_info.vms/1024**2:.1f}MB")
            except Exception:
                pass
            
            _last_memory_cleanup = current_time
            
        except Exception as e:
            logger.error(f"Memory cleanup error: {e}")

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    
    try:
        # Cleanup transcription services
        if hasattr(transcription_service, 'shutdown'):
            transcription_service.shutdown()
        if hasattr(enhanced_transcription_service, 'shutdown'):
            enhanced_transcription_service.shutdown()
        
        # Final memory cleanup
        cleanup_memory(force=True)
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")
    
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Initialize default settings on startup
settings_manager.initialize_defaults()

# Initialize enhanced transcription service with LLM support
def initialize_enhanced_services():
    """Initialize enhanced services in a separate thread."""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(enhanced_transcription_service.initialize())
        if result:
            logger.info("✅ Enhanced transcription service with LLM support initialized successfully")
        else:
            logger.warning("⚠️ Enhanced transcription service initialization failed")
        loop.close()
    except Exception as e:
        logger.error(f"❌ Failed to initialize enhanced services: {str(e)}")

# Start initialization in background thread
import threading
init_thread = threading.Thread(target=initialize_enhanced_services, daemon=True)
init_thread.start()

# Load saved settings into runtime config
saved_api_key = settings_manager.get_openai_api_key()
if saved_api_key:
    app.config['OPENAI_API_KEY'] = saved_api_key
    transcription_service.config.OPENAI_API_KEY = saved_api_key

saved_mode = settings_manager.get_processing_mode()
if saved_mode:
    app.config['PROCESSING_MODE'] = saved_mode
    transcription_service.config.PROCESSING_MODE = saved_mode

logger.info(f"Loaded settings - Processing mode: {saved_mode}, API configured: {settings_manager.is_api_configured()}")

# Supported file formats
SUPPORTED_FORMATS = {
    'video': ['.mp4', '.mov', '.avi', '.mkv', '.webm'],
    'audio': ['.mp3', '.wav', '.flac', '.m4a', '.ogg', '.aac', '.aiff']
}

@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('index.html')

@app.route('/settings')
def settings():
    """Settings page for API configuration."""
    return render_template('settings.html')

@app.route('/api/settings', methods=['GET', 'POST'])
def api_settings():
    """Get or update API settings using persistent storage."""
    if request.method == 'GET':
        try:
            return jsonify({
                'processing_mode': settings_manager.get_processing_mode(),
                'openai_api_configured': settings_manager.is_api_configured(),
                'supported_modes': app.config.get('PROCESSING_MODES', {'local': 'Local Processing', 'api': 'OpenAI API'}),
                'whisper_models': app.config.get('WHISPER_MODELS', {}),
                'settings_metadata': settings_manager.get_all()
            })
        except Exception as e:
            logger.error(f"Failed to get settings: {str(e)}")
            return jsonify({'error': 'Failed to load settings'}), 500
    
    elif request.method == 'POST':
        try:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No data provided'}), 400
            
            # Update OpenAI API key with validation and encryption
            if 'openai_api_key' in data:
                api_key = data['openai_api_key'].strip()
                if api_key:
                    if not api_key.startswith('sk-'):
                        return jsonify({'error': 'Invalid OpenAI API key format'}), 400
                    
                    # Save to persistent storage (encrypted)
                    success = settings_manager.set_openai_api_key(api_key)
                    if not success:
                        return jsonify({'error': 'Failed to save API key'}), 500
                    
                    # Update runtime config
                    app.config['OPENAI_API_KEY'] = api_key
                    transcription_service.config.OPENAI_API_KEY = api_key
                    # Refresh the API engine to use new config
                    transcription_service.refresh_api_engine()
                    
                    logger.info("OpenAI API key updated successfully")
            
            # Update processing mode
            if 'processing_mode' in data:
                mode = data['processing_mode']
                if mode not in ['local', 'api']:
                    return jsonify({'error': f'Invalid processing mode: {mode}'}), 400
                
                # Validate API mode requirements
                if mode == 'api' and not settings_manager.validate_api_mode_switch():
                    return jsonify({'error': 'Cannot switch to API mode: OpenAI API key not configured'}), 400
                
                # Save to persistent storage
                success = settings_manager.set_processing_mode(mode)
                if not success:
                    return jsonify({'error': 'Failed to save processing mode'}), 500
                
                # Update runtime config
                transcription_service.config.PROCESSING_MODE = mode
                app.config['PROCESSING_MODE'] = mode
                logger.info(f"Switched to {mode} processing mode")
            
            return jsonify({
                'success': True, 
                'message': 'Settings updated successfully',
                'processing_mode': settings_manager.get_processing_mode(),
                'openai_api_configured': settings_manager.is_api_configured()
            })
            
        except Exception as e:
            logger.error(f"Settings update error: {str(e)}")
            return jsonify({'error': 'Failed to update settings'}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and initiate processing."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Validate file format
        filename = secure_filename(file.filename or "")
        file_ext = Path(filename).suffix.lower()
        
        if not any(file_ext in formats for formats in SUPPORTED_FORMATS.values()):
            return jsonify({'error': f'Unsupported file format: {file_ext}'}), 400
        
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Save uploaded file
        upload_path = file_manager.save_upload(file, job_id, filename)
        
        # Get processing options
        options = {
            'noise_reduction': request.form.get('noise_reduction', 'auto'),
            'model_size': request.form.get('model_size', 'medium'),
            'output_formats': request.form.getlist('output_formats') or ['txt', 'srt'],
            'language': request.form.get('language', 'ar'),
            'processing_mode': request.form.get('processing_mode', app.config.get('PROCESSING_MODE', 'local')),
            # Multi-speaker processing options
            'enable_diarization': request.form.get('enable_diarization') == 'on',
            'max_speakers': int(request.form.get('max_speakers', 3)),
            'voice_enhancement': request.form.get('voice_enhancement') == 'on',
            'speaker_profiles': request.form.get('speaker_profiles') == 'on',
            # Aya Arabic Enhancement option
            'enable_aya_enhancement': request.form.get('enable_aya_enhancement') == 'on'
        }
        
        # Start processing in background using thread
        def run_async_processing():
            """Run async processing in a separate thread with its own event loop."""
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(process_file_async(job_id, upload_path, filename, options))
                loop.close()
            except Exception as e:
                logger.error(f"Background processing failed for job {job_id}: {str(e)}")
                file_manager.update_job_status(job_id, 'failed', f'Processing failed: {str(e)}')
        
        # Start processing thread
        processing_thread = threading.Thread(target=run_async_processing)
        processing_thread.daemon = True
        processing_thread.start()
        
        return jsonify({
            'job_id': job_id,
            'status': 'processing',
            'message': 'File uploaded successfully. Processing started.'
        }), 202
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({'error': 'Upload failed'}), 500

@app.route('/status/<job_id>')
def get_status(job_id: str):
    """Get processing status for a job."""
    try:
        status_info = file_manager.get_job_status(job_id)
        if not status_info:
            return jsonify({'error': 'Job not found'}), 404
        
        return jsonify(status_info)
    
    except Exception as e:
        logger.error(f"Status check error: {str(e)}")
        return jsonify({'error': 'Status check failed'}), 500

@app.route('/download/<job_id>/<format>')
def download_result(job_id: str, format: str):
    """Download processing results."""
    try:
        file_path = file_manager.get_result_path(job_id, format)
        if not file_path or not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        return send_file(
            file_path,
            as_attachment=True,
            download_name=f"transcript_{job_id}.{format}"
        )
    
    except Exception as e:
        logger.error(f"Download error: {str(e)}")
        return jsonify({'error': 'Download failed'}), 500

@app.route('/history')
def get_history():
    """Get processing history."""
    try:
        history = file_manager.get_processing_history()
        return jsonify(history)
    
    except Exception as e:
        logger.error(f"History error: {str(e)}")
        return jsonify({'error': 'Failed to get history'}), 500

@app.route('/api/transcribe', methods=['POST'])
def api_transcribe():
    """API endpoint for programmatic access."""
    try:
        # Validate API key if configured
        api_key = request.headers.get('Authorization')
        if app.config.get('API_KEY_REQUIRED') and not validate_api_key(api_key):
            return jsonify({'error': 'Invalid API key'}), 401
        
        # Process same as upload but return structured response
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Validate file format
        filename = secure_filename(file.filename or "")
        file_ext = Path(filename).suffix.lower()
        
        if not any(file_ext in formats for formats in SUPPORTED_FORMATS.values()):
            return jsonify({'error': f'Unsupported file format: {file_ext}'}), 400
        
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Save uploaded file
        upload_path = file_manager.save_upload(file, job_id, filename)
        
        # Get processing options
        options = {
            'noise_reduction': request.form.get('noise_reduction', 'auto'),
            'model_size': request.form.get('model_size', 'medium'),
            'output_formats': request.form.getlist('output_formats') or ['txt', 'srt'],
            'language': request.form.get('language', 'ar'),
            'processing_mode': request.form.get('processing_mode', app.config.get('PROCESSING_MODE', 'local')),
            # Multi-speaker processing options
            'enable_diarization': request.form.get('enable_diarization') == 'on',
            'max_speakers': int(request.form.get('max_speakers', 3)),
            'voice_enhancement': request.form.get('voice_enhancement') == 'on',
            'speaker_profiles': request.form.get('speaker_profiles') == 'on',
            # Aya Arabic Enhancement option
            'enable_aya_enhancement': request.form.get('enable_aya_enhancement') == 'on'
        }
        
        # Start processing in background using thread
        def run_async_processing():
            """Run async processing in a separate thread with its own event loop."""
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(process_file_async(job_id, upload_path, filename, options))
                loop.close()
            except Exception as e:
                logger.error(f"Background processing failed for job {job_id}: {str(e)}")
                file_manager.update_job_status(job_id, 'failed', f'Processing failed: {str(e)}')
        
        # Start processing thread
        processing_thread = threading.Thread(target=run_async_processing)
        processing_thread.daemon = True
        processing_thread.start()
        
        return jsonify({
            'success': True,
            'job_id': job_id,
            'status_url': f'/status/{job_id}'
        })
    
    except Exception as e:
        logger.error(f"API transcribe error: {str(e)}")
        return jsonify({'error': 'Transcription failed'}), 500

async def process_file_async(job_id: str, file_path: str, original_filename: str, options: Dict[str, Any]):
    """Process audio file asynchronously with enhanced monitoring for large files."""
    try:
        logger.info(f"Processing job {job_id}: {original_filename}")
        
        # Initial memory cleanup before processing
        cleanup_memory()
        
        # Check file size for enhanced monitoring
        file_size_mb = Path(file_path).stat().st_size / (1024 * 1024)
        is_large_file = file_size_mb > 50  # 50MB threshold
        
        if is_large_file:
            logger.info(f"Large file detected ({file_size_mb:.1f}MB), enabling enhanced monitoring")
            # Force memory cleanup for large files
            cleanup_memory(force=True)
        
        # Initialize progress tracking
        def update_progress(step: str, progress: int, details: str = ""):
            """Update job progress with step information."""
            processing_info = {
                'step': step,
                'step_progress': progress,
                'details': details,
                'file_size_mb': file_size_mb,
                'is_large_file': is_large_file
            }
            status_message = f'{step}: {details}' if details else step
            file_manager.update_job_status(job_id, 'processing', status_message, processing_info)
        
        # Step 1: Initialize processing
        update_progress("Initializing", 5, f"Processing {original_filename} ({file_size_mb:.1f}MB)")
        
        # Process with options
        processing_mode = options.get('processing_mode', 'local')
        
        # Step 2: Preparing transcription
        update_progress("Preparing", 10, f"Setting up {processing_mode} transcription")
        
        # Choose transcription service based on mode
        if processing_mode == 'api' and app.config.get('OPENAI_API_KEY'):
            # Step 3: API transcription
            update_progress("Transcribing", 20, "Starting API transcription")
            # Use enhanced service for API processing
            transcript_data = await enhanced_transcription_service.transcribe(
                audio_path=file_path,
                processing_mode=processing_mode,
                model_size=options.get('model_size', 'medium'),
                language=options.get('language', 'ar'),
                job_id=job_id,
                **{k: v for k, v in options.items() if k not in ['processing_mode', 'model_size', 'language']}
            )
            update_progress("Transcribing", 70, "API transcription completed")
        else:
            # Step 3: Local transcription
            update_progress("Transcribing", 20, "Starting local transcription")
            # Use unified service for local processing
            transcript_data = await transcription_service.transcribe(
                audio_path=file_path,
                processing_mode=processing_mode,
                model_size=options.get('model_size', 'medium'),
                language=options.get('language', 'ar'),
                enable_aya_enhancement=options.get('enable_aya_enhancement', False),
                job_id=job_id,
                **{k: v for k, v in options.items() if k not in ['processing_mode', 'model_size', 'language', 'enable_aya_enhancement']}
            )
            update_progress("Transcribing", 70, "Local transcription completed")
        
        # Step 4: Generate output files
        update_progress("Generating outputs", 75, "Creating output files")
        output_formats = options.get('output_formats', ['txt'])
        results = {}
        
        for i, format_type in enumerate(output_formats):
            progress = 75 + (15 * (i + 1) // len(output_formats))
            update_progress("Generating outputs", progress, f"Creating {format_type.upper()} file")
            result_path = await output_generator.generate_output(
                transcript_data, job_id, format_type
            )
            results[format_type] = result_path
        
        # Step 5: Finalizing
        update_progress("Finalizing", 95, "Saving results")
        
        # Save job results
        processing_info = transcript_data.get('processing_info', {})
        
        job_result = {
            'job_id': job_id,
            'original_filename': original_filename,
            'status': 'completed',
            'completed_at': datetime.now().isoformat(),
            'results': results,
            'transcript_data': transcript_data,
            'options': options
        }
        
        file_manager.save_job_result(job_id, job_result)
        file_manager.update_job_status(
            job_id, 'completed', 'Transcription completed successfully!',
            processing_info=processing_info
        )
        
        # Final memory cleanup after successful completion
        cleanup_memory()
        
        logger.info(f"Job {job_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Processing error for job {job_id}: {str(e)}")
        # Cleanup memory even on failure
        cleanup_memory()
        file_manager.update_job_status(job_id, 'failed', f'Processing failed: {str(e)}')

# Monitoring API Endpoints
@app.route('/api/jobs/all', methods=['GET'])
def get_all_jobs():
    """Get all jobs with detailed information for monitoring."""
    try:
        with sqlite3.connect(file_manager.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute('''
                SELECT job_id, original_filename, status, message, created_at, 
                       updated_at, completed_at, restarted_at, options, results, processing_info
                FROM jobs 
                ORDER BY created_at DESC
            ''')
            
            jobs = []
            for row in cursor.fetchall():
                job = dict(row)
                
                # Parse JSON fields safely
                try:
                    job['options'] = json.loads(row['options']) if row['options'] else {}
                    job['results'] = json.loads(row['results']) if row['results'] else {}
                    job['processing_info'] = json.loads(row['processing_info']) if row['processing_info'] else {}
                except (json.JSONDecodeError, TypeError):
                    job['options'] = {}
                    job['results'] = {}
                    job['processing_info'] = {}
                
                jobs.append(job)
            
            return jsonify(jobs)
            
    except Exception as e:
        logger.error(f"Failed to get all jobs: {str(e)}")
        return jsonify({'error': 'Failed to load jobs'}), 500

@app.route('/api/jobs/<job_id>/debug', methods=['GET'])
def get_job_debug(job_id: str):
    """Get comprehensive debug information for a job."""
    try:
        # Get job status from database
        job_info = file_manager.get_job_status(job_id)
        if not job_info:
            return jsonify({'error': 'Job not found'}), 404
        
        debug_info = {
            'job_info': job_info,
            'files': {},
            'system': {},
            'timestamps': {}
        }
        
        # Check file status
        upload_folder = Path(app.config['UPLOAD_FOLDER'])
        temp_folder = Path(app.config['TEMP_FOLDER'])
        results_folder = Path(app.config['RESULTS_FOLDER'])
        
        # Upload files
        upload_files = list(upload_folder.glob(f"{job_id}_*"))
        debug_info['files']['upload'] = []
        for f in upload_files:
            debug_info['files']['upload'].append({
                'name': f.name,
                'size': f.stat().st_size,
                'modified': f.stat().st_mtime
            })
        
        # Temp files
        temp_files = list(temp_folder.glob(f"{job_id}*"))
        debug_info['files']['temp'] = []
        for f in temp_files:
            debug_info['files']['temp'].append({
                'name': f.name,
                'size': f.stat().st_size,
                'modified': f.stat().st_mtime
            })
        
        # Result files
        result_files = list(results_folder.glob(f"{job_id}.*"))
        debug_info['files']['results'] = []
        for f in result_files:
            debug_info['files']['results'].append({
                'name': f.name,
                'size': f.stat().st_size,
                'modified': f.stat().st_mtime
            })
        
        # Processing duration
        if job_info.get('created_at'):
            try:
                created = datetime.fromisoformat(job_info['created_at'])
                now = datetime.now()
                debug_info['timestamps']['processing_duration'] = int((now - created).total_seconds())
                debug_info['timestamps']['created_at'] = job_info['created_at']
                debug_info['timestamps']['updated_at'] = job_info.get('updated_at')
                debug_info['timestamps']['completed_at'] = job_info.get('completed_at')
            except:
                pass
        
        # System info (basic)
        try:
            import psutil
            debug_info['system'] = {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent
            }
        except ImportError:
            debug_info['system'] = {
                'cpu_percent': 'psutil not available',
                'memory_percent': 'psutil not available',
                'disk_usage': 'psutil not available'
            }
        
        return jsonify(debug_info)
        
    except Exception as e:
        logger.error(f"Failed to get debug info for {job_id}: {str(e)}")
        return jsonify({'error': 'Failed to get debug info'}), 500

@app.route('/api/estimate-time', methods=['POST'])
def estimate_processing_time():
    """Get processing time estimate for uploaded parameters."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Create a temporary event loop for async call
        def get_estimate():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    file_manager.estimate_processing_time(
                        file_duration=data.get('file_duration'),
                        file_size=data.get('file_size'),
                        model_size=data.get('model_size', 'medium'),
                        processing_mode=data.get('processing_mode', 'local'),
                        options=data.get('options', {})
                    )
                )
                return result
            finally:
                loop.close()
        
        estimate = get_estimate()
        return jsonify(estimate)
        
    except Exception as e:
        logger.error(f"Time estimation error: {str(e)}")
        return jsonify({'error': 'Time estimation failed'}), 500

@app.route('/api/performance-stats')
def get_performance_stats():
    """Get time estimation performance statistics."""
    try:
        def get_stats():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(file_manager.get_time_estimation_stats())
                return result
            finally:
                loop.close()
        
        stats = get_stats()
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Performance stats error: {str(e)}")
        return jsonify({'error': 'Failed to get performance stats'}), 500

@app.route('/api/jobs/clear-completed', methods=['POST'])
def clear_completed_jobs():
    """Clear completed jobs from the database."""
    try:
        with sqlite3.connect(file_manager.db_path) as conn:
            # Get completed job IDs before deletion
            cursor = conn.execute("SELECT job_id FROM jobs WHERE status = 'completed'")
            completed_jobs = [row[0] for row in cursor.fetchall()]
            
            # Delete completed jobs
            conn.execute("DELETE FROM jobs WHERE status = 'completed'")
            
            return jsonify({
                'success': True,
                'cleared_count': len(completed_jobs),
                'cleared_jobs': completed_jobs
            })
            
    except Exception as e:
        logger.error(f"Failed to clear completed jobs: {str(e)}")
        return jsonify({'error': 'Failed to clear completed jobs'}), 500

@app.route('/api/jobs/stats', methods=['GET'])
def get_job_stats():
    """Get job statistics."""
    try:
        with sqlite3.connect(file_manager.db_path) as conn:
            cursor = conn.execute('''
                SELECT status, COUNT(*) as count 
                FROM jobs 
                GROUP BY status
            ''')
            
            stats = {}
            total = 0
            for row in cursor.fetchall():
                stats[row[0]] = row[1]
                total += row[1]
            
            # Get average processing time for completed jobs
            cursor = conn.execute('''
                SELECT AVG(
                    (julianday(completed_at) - julianday(created_at)) * 24 * 60 * 60
                ) as avg_duration
                FROM jobs 
                WHERE status = 'completed' AND completed_at IS NOT NULL
            ''')
            
            avg_duration = cursor.fetchone()[0] or 0
            
            return jsonify({
                'total_jobs': total,
                'by_status': stats,
                'average_duration': int(avg_duration)
            })
            
    except Exception as e:
        logger.error(f"Failed to get job stats: {str(e)}")
        return jsonify({'error': 'Failed to get statistics'}), 500

@app.route('/monitor')
def monitor_page():
    """Job monitoring dashboard page."""
    return render_template('monitor_new.html')

@app.route('/api/jobs/<job_id>/kill', methods=['POST'])
def kill_job(job_id: str):
    """Kill a running job."""
    try:
        # Get job info
        job_info = file_manager.get_job_status(job_id)
        if not job_info:
            return jsonify({'error': 'Job not found'}), 404
        
        if job_info['status'] != 'processing':
            return jsonify({'error': 'Job is not currently processing'}), 400
        
        # Try to kill any processes related to this job
        try:
            import psutil
            for process in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = ' '.join(process.info['cmdline'] or [])
                    if job_id in cmdline:
                        logger.info(f"Killing process {process.info['pid']} for job {job_id}")
                        process.terminate()
                        # Give process time to terminate gracefully
                        try:
                            process.wait(timeout=5)
                        except psutil.TimeoutExpired:
                            process.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except ImportError:
            logger.warning("psutil not available - cannot kill processes")
        
        # Clean up temporary files
        temp_folder = Path(app.config['TEMP_FOLDER'])
        cleaned_files = 0
        for temp_file in temp_folder.glob(f"{job_id}*"):
            try:
                temp_file.unlink()
                cleaned_files += 1
                logger.info(f"Cleaned up temp file: {temp_file.name}")
            except Exception as e:
                logger.warning(f"Failed to clean up {temp_file}: {str(e)}")
        
        # Update job status
        file_manager.update_job_status(job_id, 'failed', 'Job killed by user request')
        
        return jsonify({
            'success': True, 
            'message': f'Job killed successfully. Cleaned up {cleaned_files} temporary files.',
            'cleaned_files': cleaned_files
        })
        
    except Exception as e:
        logger.error(f"Failed to kill job {job_id}: {str(e)}")
        return jsonify({'error': 'Failed to kill job'}), 500

@app.route('/api/jobs/<job_id>/restart', methods=['POST'])
def restart_job(job_id: str):
    """Restart a failed or completed job."""
    try:
        # Get job info
        job_info = file_manager.get_job_status(job_id)
        if not job_info:
            return jsonify({'error': 'Job not found'}), 404
        
        if job_info['status'] not in ['failed', 'completed']:
            return jsonify({'error': 'Job can only be restarted if it has failed or completed'}), 400
        
        # Get original file path from job info
        original_filename = job_info.get('original_filename', 'unknown')
        
        # Check if original file still exists in uploads
        upload_folder = Path(app.config['UPLOAD_FOLDER'])
        original_file_path = None
        
        # Look for the original file in uploads folder
        for file_path in upload_folder.glob(f"{job_id}*"):
            if file_path.is_file():
                original_file_path = file_path
                break
        
        if not original_file_path or not original_file_path.exists():
            return jsonify({'error': 'Original file not found. Cannot restart job.'}), 404
        
        # Clean up any existing temporary files for this job
        temp_folder = Path(app.config['TEMP_FOLDER'])
        results_folder = Path(app.config['RESULTS_FOLDER'])
        cleaned_files = 0
        
        # Clean temp files
        for temp_file in temp_folder.glob(f"{job_id}*"):
            try:
                temp_file.unlink()
                cleaned_files += 1
                logger.info(f"Cleaned up temp file: {temp_file.name}")
            except Exception as e:
                logger.warning(f"Failed to clean up {temp_file}: {str(e)}")
        
        # Clean result files
        for result_file in results_folder.glob(f"{job_id}*"):
            try:
                result_file.unlink()
                cleaned_files += 1
                logger.info(f"Cleaned up result file: {result_file.name}")
            except Exception as e:
                logger.warning(f"Failed to clean up {result_file}: {str(e)}")
        
        # Reset job status to processing
        file_manager.update_job_status(job_id, 'processing', 'Job restarted - processing...', is_restart=True)
        
        # Get original options from job info
        try:
            options = json.loads(job_info.get('options', '{}')) if job_info.get('options') else {}
        except (json.JSONDecodeError, TypeError):
            options = {}
        
        # Set default options if not present
        if not options:
            options = {
                'noise_reduction': 'auto',
                'model_size': 'medium',
                'processing_mode': 'local',
                'output_formats': ['txt', 'srt']
            }
        
        # Start processing in a new thread
        def run_async_processing():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(
                    process_file_async(job_id, str(original_file_path), original_filename, options)
                )
                loop.close()
            except Exception as e:
                logger.error(f"Restart processing error for job {job_id}: {str(e)}")
                file_manager.update_job_status(job_id, 'failed', f'Restart failed: {str(e)}')
        
        processing_thread = threading.Thread(target=run_async_processing)
        processing_thread.daemon = True
        processing_thread.start()
        
        logger.info(f"Job {job_id} restarted successfully")
        
        return jsonify({
            'success': True, 
            'message': f'Job restarted successfully. Cleaned up {cleaned_files} files and started reprocessing.',
            'cleaned_files': cleaned_files,
            'job_id': job_id
        })
        
    except Exception as e:
        logger.error(f"Failed to restart job {job_id}: {str(e)}")
        return jsonify({'error': 'Failed to restart job'}), 500

@app.route('/api/status')
def api_status():
    """Get overall API status."""
    try:
        return jsonify({
            'status': 'online',
            'version': '1.0.0',
            'platform': 'Arabic STT Platform',
            'processing_modes': {
                'local': 'available',
                'api': 'available' if app.config.get('OPENAI_API_KEY') else 'no_api_key'
            },
            'timestamp': time.time()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/processing-modes')
def api_processing_modes():
    """Get available processing modes with validation status."""
    try:
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(transcription_service.get_processing_modes())
        loop.close()
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/jobs/export', methods=['GET'])
def export_jobs():
    """Export all jobs data as JSON."""
    try:
        format_type = request.args.get('format', 'json')
        
        # Get all jobs
        with sqlite3.connect(file_manager.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute('''
                SELECT * FROM jobs 
                ORDER BY created_at DESC
            ''')
            
            jobs = []
            for row in cursor.fetchall():
                job = dict(row)
                # Parse JSON fields
                try:
                    job['options'] = json.loads(row['options']) if row['options'] else {}
                    job['results'] = json.loads(row['results']) if row['results'] else {}
                    job['processing_info'] = json.loads(row['processing_info']) if row['processing_info'] else {}
                except:
                    job['options'] = {}
                    job['results'] = {}
                    job['processing_info'] = {}
                jobs.append(job)
        
        if format_type == 'csv':
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=[
                'job_id', 'original_filename', 'status', 'message', 
                'created_at', 'updated_at', 'completed_at'
            ])
            writer.writeheader()
            
            for job in jobs:
                csv_row = {k: v for k, v in job.items() if k != 'processing_info'}
                writer.writerow(csv_row)
            
            response_data = output.getvalue()
            response = app.response_class(
                response=response_data,
                mimetype='text/csv',
                headers={'Content-Disposition': f'attachment; filename=jobs_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'}
            )
            return response
        else:
            # JSON export
            response_data = {
                'export_timestamp': datetime.now().isoformat(),
                'total_jobs': len(jobs),
                'jobs': jobs
            }
            
            return jsonify(response_data)
            
    except Exception as e:
        logger.error(f"Failed to export jobs: {str(e)}")
        return jsonify({'error': 'Failed to export jobs'}), 500

@app.route('/time-estimation-demo')
def time_estimation_demo():
    """Time estimation demo page."""
    return render_template('time_estimation_demo.html')

@app.route('/file-manager')
def file_manager_page():
    """File management page."""
    return render_template('file_manager.html')

@app.route('/api/upload-file', methods=['POST'])
def upload_file_only():
    """Upload file without starting processing immediately."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Validate file format
        filename = secure_filename(file.filename or "")
        file_ext = Path(filename).suffix.lower()
        
        if not any(file_ext in formats for formats in SUPPORTED_FORMATS.values()):
            return jsonify({'error': f'Unsupported file format: {file_ext}'}), 400
        
        # Generate unique file ID
        file_id = str(uuid.uuid4())
        
        # Save file using file manager
        upload_path = file_manager.save_uploaded_file(file, file_id, filename)
        
        # Get file duration if possible (for audio/video files)
        duration = None
        try:
            import librosa
            duration = librosa.get_duration(filename=upload_path)
        except:
            pass  # Duration detection failed, that's ok
        
        # Get file size
        file_size = Path(upload_path).stat().st_size
        
        return jsonify({
            'id': file_id,
            'filename': filename,
            'upload_path': upload_path,
            'size': file_size,
            'duration': duration,
            'status': 'uploaded',
            'created_at': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"File upload error: {str(e)}")
        return jsonify({'error': 'Upload failed'}), 500

@app.route('/api/uploaded-files')
def get_uploaded_files():
    """Get list of uploaded files."""
    try:
        files = file_manager.get_uploaded_files()
        return jsonify(files)
    except Exception as e:
        logger.error(f"Failed to get uploaded files: {str(e)}")
        return jsonify({'error': 'Failed to load files'}), 500

@app.route('/api/process-files', methods=['POST'])
def process_uploaded_files():
    """Start processing for uploaded files."""
    try:
        data = request.get_json()
        if not data or 'file_ids' not in data:
            return jsonify({'error': 'No file IDs provided'}), 400
        
        file_ids = data['file_ids']
        if not isinstance(file_ids, list) or len(file_ids) == 0:
            return jsonify({'error': 'Invalid file IDs'}), 400
        
        # Processing options
        options = {
            'processing_mode': data.get('processing_mode', 'local'),
            'model_size': data.get('model_size', 'medium'),
            'language': data.get('language', 'ar'),
            'noise_reduction': data.get('noise_reduction', 'auto'),
            'output_formats': data.get('output_formats', ['txt', 'srt'])
        }
        
        started_count = 0
        failed_files = []
        
        for file_id in file_ids:
            try:
                file_info = file_manager.get_uploaded_file(file_id)
                if not file_info:
                    failed_files.append(f"{file_id}: File not found")
                    continue
                
                if file_info.get('status') not in ['uploaded', 'failed']:
                    failed_files.append(f"{file_info.get('filename', file_id)}: Already processing or completed")
                    continue
                
                # Create processing job
                job_id = str(uuid.uuid4())
                
                # Update file status
                file_manager.update_file_status(file_id, 'processing', job_id=job_id)
                
                # Start processing in background
                def run_async_processing():
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        loop.run_until_complete(
                            process_file_async(
                                job_id, 
                                file_info['upload_path'], 
                                file_info['filename'], 
                                {**options, 'file_size': file_info.get('size', 0)}
                            )
                        )
                        loop.close()
                        
                        # Update file status on completion
                        file_manager.update_file_status(file_id, 'completed', job_id=job_id)
                        
                    except Exception as e:
                        logger.error(f"Processing failed for file {file_id}: {str(e)}")
                        file_manager.update_file_status(file_id, 'failed', job_id=job_id, message=str(e))
                
                processing_thread = threading.Thread(target=run_async_processing)
                processing_thread.daemon = True
                processing_thread.start()
                
                started_count += 1
                
            except Exception as e:
                logger.error(f"Failed to start processing for file {file_id}: {str(e)}")
                failed_files.append(f"{file_id}: {str(e)}")
        
        response = {
            'started_count': started_count,
            'total_count': len(file_ids)
        }
        
        if failed_files:
            response['failed_files'] = failed_files
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Process files error: {str(e)}")
        return jsonify({'error': 'Failed to start processing'}), 500

@app.route('/api/delete-files', methods=['POST'])
def delete_uploaded_files():
    """Delete uploaded files."""
    try:
        data = request.get_json()
        if not data or 'file_ids' not in data:
            return jsonify({'error': 'No file IDs provided'}), 400
        
        file_ids = data['file_ids']
        if not isinstance(file_ids, list):
            return jsonify({'error': 'Invalid file IDs'}), 400
        
        deleted_count = file_manager.delete_uploaded_files(file_ids)
        
        return jsonify({
            'deleted_count': deleted_count,
            'total_count': len(file_ids)
        })
        
    except Exception as e:
        logger.error(f"Delete files error: {str(e)}")
        return jsonify({'error': 'Failed to delete files'}), 500

@app.route('/api/download-results/<file_id>')
def download_file_results(file_id: str):
    """Download results for a specific file."""
    try:
        file_info = file_manager.get_uploaded_file(file_id)
        if not file_info or not file_info.get('job_id'):
            return jsonify({'error': 'File or results not found'}), 404
        
        job_id = file_info['job_id']
        
        # Try to find result files
        results_folder = Path(app.config['RESULTS_FOLDER'])
        result_files = list(results_folder.glob(f"{job_id}.*"))
        
        if not result_files:
            return jsonify({'error': 'No result files found'}), 404
        
        # Return first result file (prefer .txt)
        txt_files = [f for f in result_files if f.suffix == '.txt']
        result_file = txt_files[0] if txt_files else result_files[0]
        
        return send_file(
            result_file,
            as_attachment=True,
            download_name=f"transcript_{file_info['filename']}{result_file.suffix}"
        )
        
    except Exception as e:
        logger.error(f"Download results error: {str(e)}")
        return jsonify({'error': 'Download failed'}), 500

# LLM Model Management API Endpoints
@app.route('/api/v1/llm-status', methods=['GET'])
def api_v1_llm_status():
    """Get LLM service status and health information."""
    try:
        # Use the global enhanced_transcription_service instance
        try:
            # Get LLM status using async call
            def get_llm_status():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(enhanced_transcription_service.get_llm_status())
                    return result
                finally:
                    loop.close()
            
            status = get_llm_status()
            return jsonify(status)
            
        except ImportError:
            return jsonify({
                'llm_enabled': False,
                'initialized': False,
                'error': 'LLM components not available - enhanced transcription service not found',
                'services': {},
                'available_models': [],
                'configuration': {}
            })
        
    except Exception as e:
        logger.error(f"Failed to get LLM status: {str(e)}")
        return jsonify({
            'llm_enabled': False,
            'initialized': False,
            'error': f'LLM status check failed: {str(e)}',
            'services': {},
            'available_models': [],
            'configuration': {}
        }), 500

@app.route('/api/v1/models', methods=['GET'])
def api_v1_models():
    """Get available LLM models."""
    try:
        # Use the global enhanced_transcription_service instance
        def get_models():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                # Get LLM status first
                status = loop.run_until_complete(enhanced_transcription_service.get_llm_status())
                
                if not status.get('llm_enabled', False):
                    return {
                        'available_models': [],
                        'recommended_models': [],
                        'statistics': {
                            'total_available': 0,
                            'total_installed_size_gb': 0,
                            'total_recommended': 0
                        },
                        'ollama_status': {
                            'running': False,
                            'version': None
                        },
                        'error': 'LLM service not configured. LLM features are disabled.'
                    }
                
                # Get available models from LLM service
                models = loop.run_until_complete(enhanced_transcription_service.llm_service.get_available_models())
                
                # Calculate statistics
                total_size_gb = 0
                for model in models:
                    if 'size' in model:
                        # Convert size to GB (assuming size is in bytes)
                        size_bytes = model.get('size', 0)
                        total_size_gb += size_bytes / (1024**3)
                
                # Check Ollama status
                ollama_running = loop.run_until_complete(enhanced_transcription_service.llm_service.health_check())
                
                return {
                    'available_models': models,
                    'recommended_models': [
                        {'name': 'llama3.2:3b', 'size': '2.0GB', 'installed': any(m.get('name') == 'llama3.2:3b' for m in models)},
                        {'name': 'aya:8b', 'size': '4.8GB', 'installed': any(m.get('name') == 'aya:8b' for m in models)}
                    ],
                    'statistics': {
                        'total_available': len(models),
                        'total_installed_size_gb': round(total_size_gb, 2),
                        'total_recommended': 2
                    },
                    'ollama_status': {
                        'running': ollama_running,
                        'version': 'Unknown'
                    }
                }
            finally:
                loop.close()
        
        result = get_models()
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Failed to get LLM models: {str(e)}")
        return jsonify({
            'available_models': [],
            'recommended_models': [],
            'statistics': {
                'total_available': 0,
                'total_installed_size_gb': 0,
                'total_recommended': 0
            },
            'ollama_status': {
                'running': False,
                'version': None
            },
            'error': f'Failed to load models: {str(e)}'
        }), 500

@app.route('/api/v1/models/install', methods=['POST'])
def api_v1_models_install():
    """Install LLM model (placeholder implementation)."""
    try:
        data = request.get_json()
        if not data or 'model' not in data:
            return jsonify({'error': 'Model name is required'}), 400
        
        return jsonify({
            'error': 'LLM service not configured. Model installation is not available.'
        }), 503
    except Exception as e:
        logger.error(f"Failed to install LLM model: {str(e)}")
        return jsonify({'error': 'LLM service unavailable'}), 500

@app.route('/api/v1/models/set-default', methods=['POST'])
def api_v1_models_set_default():
    """Set default LLM model (placeholder implementation)."""
    try:
        data = request.get_json()
        if not data or 'model' not in data:
            return jsonify({'error': 'Model name is required'}), 400
        
        return jsonify({
            'error': 'LLM service not configured. Default model setting is not available.'
        }), 503
    except Exception as e:
        logger.error(f"Failed to set default LLM model: {str(e)}")
        return jsonify({'error': 'LLM service unavailable'}), 500

@app.route('/api/v1/models/install-status/<model>', methods=['GET'])
def api_v1_models_install_status(model: str):
    """Get LLM model installation status (placeholder implementation)."""
    try:
        return jsonify({
            'model': model,
            'status': 'not_available',
            'error': 'LLM service not configured.'
        })
    except Exception as e:
        logger.error(f"Failed to get installation status: {str(e)}")
        return jsonify({'error': 'LLM service unavailable'}), 500

def validate_api_key(api_key: Optional[str]) -> bool:
    """Validate API key if provided."""
    if not api_key:
        return False
    
    # Remove 'Bearer ' prefix if present
    if api_key.startswith('Bearer '):
        api_key = api_key[7:]
    
    # In production, validate against database or config
    return api_key == app.config.get('API_KEY')

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Ensure required directories exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
    os.makedirs(app.config['TEMP_FOLDER'], exist_ok=True)
    
    # Run the application
    app.run(
        host=app.config.get('HOST', '0.0.0.0'),
        port=app.config.get('PORT', 5002),
        debug=app.config.get('DEBUG', False)
    )
