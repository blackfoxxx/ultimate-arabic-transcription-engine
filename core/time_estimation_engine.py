"""
Time estimation engine for Arabic STT Platform
Provides accurate processing time predictions based on hardware, model, and file characteristics
"""

import asyncio
import logging
import json
import sqlite3
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from datetime import datetime, timedelta
import psutil
import platform
import subprocess
import librosa
import numpy as np
import math

from config import Config

logger = logging.getLogger(__name__)

class TimeEstimationEngine:
    """Estimates processing time for transcription jobs based on multiple factors."""
    
    def __init__(self):
        self.config = Config()
        self.db_path = self.config.BASE_DIR / 'time_estimates.db'
        self._init_database()
        self.hardware_specs = self._detect_hardware()
        self.model_performance_data = self._get_model_performance_data()
        
    def _init_database(self):
        """Initialize database for storing historical performance data."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Table for storing actual processing times for learning
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS processing_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        job_id TEXT,
                        file_duration REAL,
                        file_size REAL,
                        model_size TEXT,
                        processing_mode TEXT,
                        hardware_profile TEXT,
                        multi_speaker BOOLEAN,
                        noise_reduction TEXT,
                        estimated_time REAL,
                        actual_time REAL,
                        accuracy_ratio REAL,
                        created_at TEXT
                    )
                ''')
                
                # Table for hardware performance benchmarks
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS hardware_benchmarks (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        hardware_profile TEXT UNIQUE,
                        model_size TEXT,
                        processing_mode TEXT,
                        base_multiplier REAL,
                        updated_at TEXT
                    )
                ''')
                
                conn.commit()
                logger.info("Time estimation database initialized")
                
        except Exception as e:
            logger.error(f"Failed to initialize time estimation database: {str(e)}")
    
    def _detect_hardware(self) -> Dict[str, Any]:
        """Detect system hardware specifications."""
        try:
            specs = {
                'cpu_count': psutil.cpu_count(logical=True),
                'cpu_freq': psutil.cpu_freq().max if psutil.cpu_freq() else 0,
                'memory_total': psutil.virtual_memory().total,
                'memory_available': psutil.virtual_memory().available,
                'platform': platform.system(),
                'architecture': platform.machine(),
                'gpu_available': False,
                'gpu_memory': 0,
                'gpu_name': None
            }
            
            # Detect GPU
            try:
                # Check for NVIDIA GPU
                result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    gpu_info = result.stdout.strip().split('\n')[0].split(', ')
                    specs['gpu_available'] = True
                    specs['gpu_name'] = gpu_info[0]
                    specs['gpu_memory'] = float(gpu_info[1]) * 1024 * 1024  # Convert MB to bytes
                    logger.info(f"Detected NVIDIA GPU: {specs['gpu_name']} with {specs['gpu_memory']/1e9:.1f}GB")
            except Exception:
                pass
            
            # Check for Apple Silicon
            if specs['platform'] == 'Darwin' and 'arm' in specs['architecture'].lower():
                specs['apple_silicon'] = True
                specs['gpu_available'] = True  # Apple Silicon has integrated GPU
                logger.info("Detected Apple Silicon with integrated GPU")
            else:
                specs['apple_silicon'] = False
            
            # Create hardware profile identifier
            specs['profile'] = self._create_hardware_profile(specs)
            
            logger.info(f"Hardware detected: {specs['profile']}")
            return specs
            
        except Exception as e:
            logger.error(f"Hardware detection failed: {str(e)}")
            return {
                'cpu_count': 4,
                'memory_total': 8 * 1024**3,  # 8GB default
                'gpu_available': False,
                'profile': 'unknown_hardware'
            }
    
    def _create_hardware_profile(self, specs: Dict[str, Any]) -> str:
        """Create a hardware profile identifier for benchmarking."""
        # Categorize CPU performance
        cpu_tier = "low"
        if specs['cpu_count'] >= 8:
            cpu_tier = "high"
        elif specs['cpu_count'] >= 4:
            cpu_tier = "medium"
        
        # Categorize memory
        memory_gb = specs['memory_total'] / (1024**3)
        memory_tier = "low"
        if memory_gb >= 32:
            memory_tier = "high"
        elif memory_gb >= 16:
            memory_tier = "medium"
        
        # GPU category
        gpu_tier = "none"
        if specs['gpu_available']:
            if specs.get('apple_silicon'):
                gpu_tier = "apple_silicon"
            elif specs['gpu_memory'] > 8 * 1024**3:  # 8GB+
                gpu_tier = "high"
            elif specs['gpu_memory'] > 4 * 1024**3:  # 4GB+
                gpu_tier = "medium"
            else:
                gpu_tier = "low"
        
        return f"{cpu_tier}_{memory_tier}_{gpu_tier}"
    
    def _get_model_performance_data(self) -> Dict[str, Dict[str, float]]:
        """Get base performance data for different models and configurations."""
        # Base multipliers relative to audio duration (1.0 = real-time processing)
        return {
            'local': {
                'tiny': 0.25,    # 4x real-time (very fast)
                'base': 0.5,     # 2x real-time (fast)
                'small': 0.8,    # 1.25x real-time (balanced)
                'medium': 1.0,   # 1x real-time (recommended)
                'large': 2.0     # 0.5x real-time (slow but accurate)
            },
            'api': {
                'whisper-1': 0.25  # 4x real-time (cloud processing)
            }
        }
    
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
            audio_path: Path to audio file (optional, for analysis)
            file_duration: Duration in seconds (if known)
            file_size: File size in bytes (if known)
            model_size: Whisper model size or API model name
            processing_mode: 'local' or 'api'
            options: Additional processing options
            
        Returns:
            Dict with time estimates and metadata
        """
        try:
            options = options or {}
            
            # Get or calculate file properties
            if audio_path and not file_duration:
                file_duration = await self._get_audio_duration(audio_path)
            if audio_path and not file_size:
                file_size = Path(audio_path).stat().st_size
            
            if not file_duration:
                raise ValueError("File duration must be provided or audio_path must be accessible")
            
            # Get base processing multiplier
            base_multiplier = self._get_base_multiplier(model_size, processing_mode)
            
            # Apply hardware adjustments
            hardware_multiplier = self._get_hardware_multiplier(processing_mode)
            
            # Apply feature-specific multipliers
            feature_multiplier = self._get_feature_multiplier(options)
            
            # Apply file complexity multiplier
            complexity_multiplier = await self._get_complexity_multiplier(
                audio_path, file_duration, file_size, options
            )
            
            # Calculate final time estimate
            total_multiplier = base_multiplier * hardware_multiplier * feature_multiplier * complexity_multiplier
            estimated_seconds = file_duration * total_multiplier
            
            # Get historical accuracy
            accuracy_info = await self._get_historical_accuracy(
                model_size, processing_mode, self.hardware_specs['profile']
            )
            
            # Apply learning adjustments
            if accuracy_info['count'] > 5:  # Have enough data
                learning_adjustment = accuracy_info['avg_ratio']
                estimated_seconds *= learning_adjustment
            
            # Create estimate ranges
            confidence = min(0.9, 0.5 + (accuracy_info['count'] * 0.05))  # Max 90% confidence
            variance = max(0.15, 0.5 - (accuracy_info['count'] * 0.02))   # Min 15% variance
            
            min_estimate = estimated_seconds * (1 - variance)
            max_estimate = estimated_seconds * (1 + variance)
            
            # Format results
            result = {
                'estimated_time_seconds': round(estimated_seconds),
                'estimated_time_formatted': self._format_duration(estimated_seconds),
                'min_estimate_seconds': round(min_estimate),
                'max_estimate_seconds': round(max_estimate),
                'confidence_percentage': round(confidence * 100),
                'file_duration_seconds': file_duration,
                'processing_speed_ratio': round(1 / total_multiplier, 2),
                'factors': {
                    'base_multiplier': base_multiplier,
                    'hardware_multiplier': hardware_multiplier,
                    'feature_multiplier': feature_multiplier,
                    'complexity_multiplier': complexity_multiplier,
                    'total_multiplier': total_multiplier
                },
                'hardware_profile': self.hardware_specs['profile'],
                'model_size': model_size,
                'processing_mode': processing_mode,
                'historical_jobs': accuracy_info['count'],
                'estimated_completion': self._format_completion_time(estimated_seconds)
            }
            
            # Add cost estimate for API mode
            if processing_mode == 'api':
                cost_info = self._estimate_api_cost(file_duration)
                result.update(cost_info)
            
            logger.info(f"Time estimation: {result['estimated_time_formatted']} for {file_duration:.1f}s audio")
            return result
            
        except Exception as e:
            logger.error(f"Time estimation failed: {str(e)}")
            # Return fallback estimate
            fallback_time = (file_duration or 300) * 1.0  # Assume 1x real-time
            return {
                'estimated_time_seconds': round(fallback_time),
                'estimated_time_formatted': self._format_duration(fallback_time),
                'error': str(e),
                'confidence_percentage': 30
            }
    
    async def _get_audio_duration(self, audio_path: str) -> float:
        """Get audio file duration efficiently without loading large files."""
        try:
            # First try with soundfile for efficiency
            import soundfile as sf
            info = sf.info(audio_path)
            return float(info.duration)
        except Exception:
            try:
                # Fallback to librosa for compatibility, but more efficient method
                duration = librosa.get_duration(path=audio_path)
                return float(duration)
            except Exception as e:
                logger.error(f"Failed to get audio duration: {str(e)}")
                # Final fallback: estimate from file size (very rough)
                file_size = Path(audio_path).stat().st_size
                return max(60.0, file_size / 32000)  # Rough estimate for 16kHz mono
    
    def _get_base_multiplier(self, model_size: str, processing_mode: str) -> float:
        """Get base processing time multiplier for model and mode."""
        try:
            return self.model_performance_data[processing_mode][model_size]
        except KeyError:
            logger.warning(f"Unknown model {model_size} for mode {processing_mode}, using default")
            return 1.0
    
    def _get_hardware_multiplier(self, processing_mode: str) -> float:
        """Calculate hardware performance multiplier."""
        if processing_mode == 'api':
            return 1.0  # Hardware doesn't affect API processing
        
        specs = self.hardware_specs
        multiplier = 1.0
        
        # CPU adjustment
        if specs['cpu_count'] >= 8:
            multiplier *= 0.7  # Faster with more cores
        elif specs['cpu_count'] <= 2:
            multiplier *= 1.5  # Slower with fewer cores
        
        # Memory adjustment
        memory_gb = specs['memory_total'] / (1024**3)
        if memory_gb < 8:
            multiplier *= 1.3  # Slower with less memory
        elif memory_gb >= 32:
            multiplier *= 0.8  # Faster with more memory
        
        # GPU acceleration
        if specs['gpu_available']:
            if specs.get('apple_silicon'):
                multiplier *= 0.6  # Apple Silicon is very efficient
            elif specs['gpu_memory'] > 8 * 1024**3:
                multiplier *= 0.4  # High-end GPU
            elif specs['gpu_memory'] > 4 * 1024**3:
                multiplier *= 0.6  # Mid-range GPU
            else:
                multiplier *= 0.8  # Low-end GPU
        
        return multiplier
    
    def _get_feature_multiplier(self, options: Dict[str, Any]) -> float:
        """Calculate multiplier based on enabled features."""
        multiplier = 1.0
        
        # Multi-speaker processing (speaker diarization)
        if options.get('multi_speaker', False):
            multiplier *= 2.5  # Significantly slower
        
        # Noise reduction
        noise_reduction = options.get('noise_reduction', 'none')
        if noise_reduction == 'rnnoise':
            multiplier *= 1.2  # AI-based noise reduction adds overhead
        elif noise_reduction == 'traditional':
            multiplier *= 1.1  # Traditional filters add minimal overhead
        
        # High-quality settings
        if options.get('beam_size', 5) > 5:
            multiplier *= 1.2  # More beam search paths
        
        if options.get('best_of', 5) > 5:
            multiplier *= 1.3  # More candidate evaluations
        
        # Word-level timestamps
        if options.get('word_timestamps', False):
            multiplier *= 1.1  # Slight overhead for word alignment
        
        return multiplier
    
    async def _get_complexity_multiplier(
        self,
        audio_path: Optional[str],
        duration: float,
        file_size: Optional[int],
        options: Dict[str, Any]
    ) -> float:
        """Calculate multiplier based on audio complexity."""
        multiplier = 1.0
        
        try:
            # File size to duration ratio (indicates quality/complexity)
            if file_size and duration > 0:
                bitrate = file_size * 8 / duration / 1000  # kbps
                if bitrate > 320:  # High quality
                    multiplier *= 1.1
                elif bitrate < 64:  # Low quality (may be harder to transcribe)
                    multiplier *= 1.2
            
            # Analyze audio characteristics if file is available
            if audio_path and Path(audio_path).exists():
                try:
                    # For large files, use more efficient analysis
                    file_size_mb = Path(audio_path).stat().st_size / (1024 * 1024)
                    
                    if file_size_mb > 50:  # Large file, use minimal analysis
                        # Use soundfile for efficient duration check
                        import soundfile as sf
                        info = sf.info(audio_path)
                        
                        # Simple heuristic based on file properties
                        if info.channels > 1:
                            multiplier *= 1.1  # Stereo is slightly more complex
                        
                        # Large files typically have more complexity
                        if file_size_mb > 100:
                            multiplier *= 1.2
                        
                    else:
                        # Load a small sample for detailed analysis (smaller files only)
                        y, sr = librosa.load(audio_path, duration=min(30.0, duration), sr=16000)
                        
                        # Check for silence/speech ratio
                        silence_threshold = 0.01
                        silent_frames = np.sum(np.abs(y) < silence_threshold)
                        silence_ratio = silent_frames / len(y)
                        
                        if silence_ratio > 0.7:  # Lots of silence
                            multiplier *= 0.8  # Faster processing
                        elif silence_ratio < 0.1:  # Very dense audio
                            multiplier *= 1.2  # Slower processing
                        
                        # Check for audio complexity (spectral features)
                        try:
                            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
                            spectral_variance = np.var(spectral_centroids)
                            
                            if spectral_variance > 1000000:  # Complex audio (music, multiple speakers, etc.)
                                multiplier *= 1.3
                        except Exception:
                            pass  # Skip spectral analysis if it fails
                    
                except Exception as e:
                    logger.debug(f"Audio analysis failed: {str(e)}")
                    
        except Exception as e:
            logger.debug(f"Complexity analysis failed: {str(e)}")
        
        return multiplier
    
    def _estimate_api_cost(self, duration_seconds: float) -> Dict[str, Any]:
        """Estimate API processing cost."""
        minutes = duration_seconds / 60
        cost_per_minute = 0.006  # OpenAI pricing
        total_cost = minutes * cost_per_minute
        
        return {
            'estimated_cost_usd': round(total_cost, 4),
            'cost_per_minute': cost_per_minute,
            'billable_minutes': round(minutes, 2)
        }
    
    async def _get_historical_accuracy(
        self,
        model_size: str,
        processing_mode: str,
        hardware_profile: str
    ) -> Dict[str, Any]:
        """Get historical estimation accuracy for similar configurations."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT COUNT(*), AVG(accuracy_ratio), MIN(accuracy_ratio), MAX(accuracy_ratio)
                    FROM processing_history
                    WHERE model_size = ? AND processing_mode = ? AND hardware_profile = ?
                    AND created_at > datetime('now', '-30 days')
                ''', (model_size, processing_mode, hardware_profile))
                
                row = cursor.fetchone()
                count, avg_ratio, min_ratio, max_ratio = row
                
                return {
                    'count': count or 0,
                    'avg_ratio': avg_ratio or 1.0,
                    'min_ratio': min_ratio or 0.8,
                    'max_ratio': max_ratio or 1.2
                }
        except Exception as e:
            logger.debug(f"Historical accuracy lookup failed: {str(e)}")
            return {'count': 0, 'avg_ratio': 1.0, 'min_ratio': 0.8, 'max_ratio': 1.2}
    
    async def record_actual_time(
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
        try:
            accuracy_ratio = actual_time / estimated_time if estimated_time > 0 else 1.0
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO processing_history (
                        job_id, file_duration, file_size, model_size, processing_mode,
                        hardware_profile, multi_speaker, noise_reduction, estimated_time,
                        actual_time, accuracy_ratio, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    job_id,
                    file_duration,
                    file_size,
                    model_size,
                    processing_mode,
                    self.hardware_specs['profile'],
                    options.get('multi_speaker', False),
                    options.get('noise_reduction', 'none'),
                    estimated_time,
                    actual_time,
                    accuracy_ratio,
                    datetime.now().isoformat()
                ))
                
                conn.commit()
                
            logger.info(f"Recorded processing time: estimated={estimated_time:.1f}s, actual={actual_time:.1f}s, ratio={accuracy_ratio:.2f}")
            
            # Update hardware benchmarks if we have enough data
            await self._update_hardware_benchmarks(model_size, processing_mode)
            
        except Exception as e:
            logger.error(f"Failed to record actual time: {str(e)}")
    
    async def _update_hardware_benchmarks(self, model_size: str, processing_mode: str):
        """Update hardware performance benchmarks based on recent data."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get recent performance data
                cursor = conn.execute('''
                    SELECT AVG(actual_time / file_duration) as avg_multiplier
                    FROM processing_history
                    WHERE model_size = ? AND processing_mode = ? AND hardware_profile = ?
                    AND created_at > datetime('now', '-7 days')
                    GROUP BY hardware_profile
                    HAVING COUNT(*) >= 3
                ''', (model_size, processing_mode, self.hardware_specs['profile']))
                
                row = cursor.fetchone()
                if row:
                    avg_multiplier = row[0]
                    
                    # Update or insert benchmark
                    conn.execute('''
                        INSERT OR REPLACE INTO hardware_benchmarks
                        (hardware_profile, model_size, processing_mode, base_multiplier, updated_at)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (
                        self.hardware_specs['profile'],
                        model_size,
                        processing_mode,
                        avg_multiplier,
                        datetime.now().isoformat()
                    ))
                    
                    conn.commit()
                    logger.info(f"Updated hardware benchmark: {avg_multiplier:.2f}x for {model_size}/{processing_mode}")
                    
        except Exception as e:
            logger.debug(f"Benchmark update failed: {str(e)}")
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format."""
        if seconds < 60:
            return f"{seconds:.0f} seconds"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f} minutes"
        else:
            hours = seconds / 3600
            return f"{hours:.1f} hours"
    
    def _format_completion_time(self, estimated_seconds: float) -> str:
        """Format estimated completion time."""
        completion_time = datetime.now() + timedelta(seconds=estimated_seconds)
        return completion_time.strftime("%H:%M:%S")
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics and accuracy metrics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                # Overall statistics
                cursor = conn.execute('''
                    SELECT 
                        COUNT(*) as total_jobs,
                        AVG(accuracy_ratio) as avg_accuracy,
                        MIN(accuracy_ratio) as min_accuracy,
                        MAX(accuracy_ratio) as max_accuracy,
                        AVG(actual_time) as avg_processing_time,
                        AVG(file_duration) as avg_file_duration
                    FROM processing_history
                    WHERE created_at > datetime('now', '-30 days')
                ''')
                overall = dict(cursor.fetchone())
                
                # Per-model statistics
                cursor = conn.execute('''
                    SELECT 
                        model_size,
                        processing_mode,
                        COUNT(*) as jobs,
                        AVG(accuracy_ratio) as avg_accuracy,
                        AVG(actual_time / file_duration) as avg_speed_ratio
                    FROM processing_history
                    WHERE created_at > datetime('now', '-30 days')
                    GROUP BY model_size, processing_mode
                    ORDER BY jobs DESC
                ''')
                by_model = [dict(row) for row in cursor.fetchall()]
                
                # Hardware profile statistics
                cursor = conn.execute('''
                    SELECT 
                        hardware_profile,
                        COUNT(*) as jobs,
                        AVG(accuracy_ratio) as avg_accuracy,
                        AVG(actual_time / file_duration) as avg_speed_ratio
                    FROM processing_history
                    WHERE created_at > datetime('now', '-30 days')
                    GROUP BY hardware_profile
                    ORDER BY jobs DESC
                ''')
                by_hardware = [dict(row) for row in cursor.fetchall()]
                
                return {
                    'overall': overall,
                    'by_model': by_model,
                    'by_hardware': by_hardware,
                    'current_hardware': self.hardware_specs
                }
                
        except Exception as e:
            logger.error(f"Failed to get performance stats: {str(e)}")
            return {'error': str(e)}
    
    async def cleanup_old_data(self, days_to_keep: int = 90):
        """Clean up old historical data."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    DELETE FROM processing_history
                    WHERE created_at < datetime('now', '-{} days')
                '''.format(days_to_keep))
                
                deleted_count = cursor.rowcount
                conn.commit()
                
                if deleted_count > 0:
                    logger.info(f"Cleaned up {deleted_count} old processing history records")
                    
        except Exception as e:
            logger.error(f"Data cleanup failed: {str(e)}")
