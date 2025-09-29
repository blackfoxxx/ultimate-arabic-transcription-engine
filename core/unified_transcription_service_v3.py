"""
Unified transcription service for Arabic STT Platform
Handles both local and API-based transcription with optional Aya enhancement
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import time
import signal
import psutil
import torch
import gc
import threading
import weakref

from config import Config
from core.transcription_engine import TranscriptionEngine
from core.api_transcription_engine import APITranscriptionEngine
from core.time_estimation_engine import TimeEstimationEngine

# Import enhanced Arabic engine
try:
    from core.enhanced_arabic_transcription_engine import EnhancedArabicTranscriptionEngine
    ENHANCED_ARABIC_AVAILABLE = True
except ImportError:
    ENHANCED_ARABIC_AVAILABLE = False

# Import advanced Arabic engine v2.0
try:
    from core.advanced_arabic_transcription_engine import AdvancedArabicTranscriptionEngine
    ADVANCED_ARABIC_AVAILABLE = True
except ImportError as e:
    ADVANCED_ARABIC_AVAILABLE = False

# Import ultimate Arabic engine v3.0
try:
    from core.ultimate_arabic_transcription_engine import UltimateArabicTranscriptionEngine
    ULTIMATE_ARABIC_AVAILABLE = True
except ImportError as e:
    ULTIMATE_ARABIC_AVAILABLE = False

try:
    from aya_arabic_enhancer import AyaArabicEnhancer
    AYA_AVAILABLE = True
except ImportError:
    AYA_AVAILABLE = False

# Import voice analysis engine
voice_analysis_error = None
try:
    from voice_analysis_engine import VoiceAnalysisEngine
    VOICE_ANALYSIS_AVAILABLE = True
except ImportError as e:
    VOICE_ANALYSIS_AVAILABLE = False
    voice_analysis_error = str(e)

logger = logging.getLogger(__name__)

# Log voice analysis availability after logger is defined
if not VOICE_ANALYSIS_AVAILABLE:
    logger.warning(f"Voice analysis engine not available: {voice_analysis_error}")

class UnifiedTranscriptionService:
    """Unified service that can use either local models or cloud APIs with optional Aya enhancement."""
    
    def __init__(self):
        self.config = Config()
        self.local_engine = None
        self.api_engine = None
        self.enhanced_arabic_engine = None
        self.advanced_arabic_engine = None
        self.ultimate_arabic_engine = None
        self.aya_enhancer = None
        self.voice_analysis_engine = None
        self.time_estimator = TimeEstimationEngine()
        
        # Memory management
        self._memory_lock = threading.Lock()
        self._active_jobs = weakref.WeakSet()
        self._last_cleanup = time.time()
        
        # Initialize engines
        self._initialize_engines()
    
    def _initialize_engines(self):
        """Initialize all available transcription engines with memory management."""
        try:
            # Enhanced Arabic engine (if available)
            if ENHANCED_ARABIC_AVAILABLE:
                try:
                    self.enhanced_arabic_engine = EnhancedArabicTranscriptionEngine()
                    logger.info("✨ Enhanced Arabic transcription engine initialized")
                except Exception as e:
                    logger.error(f"Failed to initialize Enhanced Arabic engine: {e}")
                    self.enhanced_arabic_engine = None
            
            # Advanced Arabic engine v2.0 (if available)
            if ADVANCED_ARABIC_AVAILABLE:
                try:
                    self.advanced_arabic_engine = AdvancedArabicTranscriptionEngine()
                    logger.info("✨ Advanced Arabic transcription engine v2.0 initialized")
                except Exception as e:
                    logger.error(f"Failed to initialize Advanced Arabic engine v2.0: {e}")
                    self.advanced_arabic_engine = None
            
            # Ultimate Arabic engine v3.0 (if available) - HIGHEST PRIORITY
            if ULTIMATE_ARABIC_AVAILABLE:
                try:
                    # Use auto device detection instead of hardcoded CPU
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    # Don't initialize with hardcoded model size - will be set dynamically
                    self.ultimate_arabic_engine = UltimateArabicTranscriptionEngine(device=device)
                    # Don't initialize model here - will be done with user-selected model_size
                    logger.info(f"🔥 Ultimate Arabic transcription engine v3.0 ready on {device.upper()}")
                except Exception as e:
                    logger.error(f"Failed to initialize Ultimate Arabic engine v3.0: {e}")
                    self.ultimate_arabic_engine = None
            
            # Aya enhancer (if available)
            if AYA_AVAILABLE:
                try:
                    self.aya_enhancer = AyaArabicEnhancer()
                    logger.info("✨ Aya Arabic enhancer initialized")
                except Exception as e:
                    logger.error(f"Failed to initialize Aya enhancer: {e}")
                    self.aya_enhancer = None
            
            # Voice analysis engine (if available)
            if VOICE_ANALYSIS_AVAILABLE:
                try:
                    self.voice_analysis_engine = VoiceAnalysisEngine()
                    logger.info("🎤 Voice analysis engine initialized")
                except Exception as e:
                    logger.error(f"Failed to initialize Voice analysis engine: {e}")
                    self.voice_analysis_engine = None
                    
        except Exception as e:
            logger.error(f"Error during engine initialization: {e}")
    
    def _cleanup_memory(self, force: bool = False):
        """Perform memory cleanup to prevent crashes."""
        current_time = time.time()
        
        # Only cleanup if enough time has passed or forced
        if not force and (current_time - self._last_cleanup) < 30:
            return
            
        with self._memory_lock:
            try:
                # Force garbage collection
                gc.collect()
                
                # Clear GPU cache if available
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    # Get GPU memory info
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory
                    gpu_allocated = torch.cuda.memory_allocated(0)
                    gpu_cached = torch.cuda.memory_reserved(0)
                    
                    logger.debug(f"GPU Memory - Total: {gpu_memory/1024**3:.1f}GB, "
                               f"Allocated: {gpu_allocated/1024**3:.1f}GB, "
                               f"Cached: {gpu_cached/1024**3:.1f}GB")
                
                # Log system memory usage
                process = psutil.Process()
                memory_info = process.memory_info()
                logger.debug(f"System Memory - RSS: {memory_info.rss/1024**2:.1f}MB, "
                           f"VMS: {memory_info.vms/1024**2:.1f}MB")
                
                self._last_cleanup = current_time
                
            except Exception as e:
                logger.error(f"Memory cleanup error: {e}")

    async def get_local_engine(self) -> TranscriptionEngine:
        """Get local transcription engine."""
        if self.local_engine is None:
            self.local_engine = TranscriptionEngine()
        return self.local_engine

    async def get_api_engine(self) -> APITranscriptionEngine:
        """Get API transcription engine."""
        if self.api_engine is None:
            self.api_engine = APITranscriptionEngine()
        return self.api_engine

    def get_estimated_time(self, audio_duration: float, processing_mode: str = 'local', model_size: str = 'medium') -> Dict[str, Any]:
        """Get estimated processing time."""
        return self.time_estimator.estimate_processing_time(
            audio_duration=audio_duration,
            processing_mode=processing_mode,
            model_size=model_size
        )

    async def transcribe(
        self,
        audio_path: str,
        language: str = 'auto',
        processing_mode: str = 'local',
        enable_aya_enhancement: bool = False,
        model_size: str = 'medium',
        **kwargs
    ) -> Dict[str, Any]:
        """
        Transcribe audio with the best available engine for the language.
        
        For Arabic language:
        Priority: Ultimate Arabic v3.0 > Advanced Arabic v2.0 > Enhanced Arabic v1.0 > Standard Whisper
        """
        start_time = time.time()
        
        # Cleanup memory before starting
        self._cleanup_memory()
        
        try:
            # Determine Arabic engine priority
            use_ultimate_arabic = (
                language == 'ar' and 
                processing_mode == 'local' and 
                self.ultimate_arabic_engine is not None and
                kwargs.get('use_ultimate_arabic', True)
            )
            
            use_advanced_arabic = (
                language == 'ar' and 
                processing_mode == 'local' and 
                not use_ultimate_arabic and
                self.advanced_arabic_engine is not None and
                kwargs.get('use_advanced_arabic', True)
            )
            
            use_enhanced_arabic = (
                language == 'ar' and 
                processing_mode == 'local' and 
                not use_ultimate_arabic and
                not use_advanced_arabic and
                self.enhanced_arabic_engine is not None and
                kwargs.get('use_enhanced_arabic', True)
            )
            
            if use_ultimate_arabic:
                logger.info("🔥 Using Ultimate Arabic transcription engine v3.0 (Maximum Quality)")
                # Ultimate Arabic transcription - highest quality with user-selected model_size
                transcription_result = self.ultimate_arabic_engine.transcribe(audio_path, model_size=model_size)
                
                # Transform Ultimate Arabic result structure to match expected format
                if transcription_result and transcription_result.get('success'):
                    # Extract text from transcript structure
                    transcript_data = transcription_result.get('transcript', {})
                    text = transcript_data.get('full_text', '')
                    segments = transcript_data.get('segments', [])
                    
                    # Calculate duration from segments if available
                    duration = 0.0
                    if segments:
                        # Get the end time of the last segment
                        last_segment = segments[-1]
                        if isinstance(last_segment, dict) and 'end' in last_segment:
                            duration = last_segment['end']
                        elif hasattr(last_segment, 'end'):
                            duration = last_segment.end
                    
                    # Restructure to match output generator expectations
                    transcription_result = {
                        'text': text,
                        'segments': segments,
                        'duration': duration,
                        'language': transcription_result.get('language', 'ar'),
                        'model_size': transcription_result.get('metadata', {}).get('model_size', model_size),
                        'processing_time': transcription_result.get('processing_time', 0.0),
                        'quality_metrics': transcription_result.get('quality_metrics', {}),
                        'metadata': transcription_result.get('metadata', {}),
                        'engine': transcription_result.get('engine', 'ultimate_arabic_v3'),
                        'success': True
                    }
                
            elif use_advanced_arabic:
                logger.info("🚀 Using Advanced Arabic transcription engine v2.0")
                # Advanced Arabic transcription
                transcription_result = await self.advanced_arabic_engine.transcribe_arabic_advanced(
                    audio_path=audio_path,
                    model_size=model_size,
                    enable_preprocessing=kwargs.get('enable_arabic_preprocessing', True),
                    **{k: v for k, v in kwargs.items() if k not in ['use_advanced_arabic', 'use_enhanced_arabic', 'enable_arabic_preprocessing']}
                )
                
            elif use_enhanced_arabic:
                logger.info("🎯 Using Enhanced Arabic transcription engine")
                # Enhanced Arabic transcription
                transcription_result = await self.enhanced_arabic_engine.transcribe_with_enhanced_arabic(
                    audio_path=audio_path,
                    model_size=model_size,
                    enable_preprocessing=kwargs.get('enable_arabic_preprocessing', True),
                    **{k: v for k, v in kwargs.items() if k not in ['use_enhanced_arabic', 'enable_arabic_preprocessing']}
                )
                
            else:
                # Standard transcription path
                if processing_mode == 'local':
                    engine = await self.get_local_engine()
                else:
                    engine = await self.get_api_engine()
                
                transcription_result = await engine.transcribe(
                    audio_path=audio_path,
                    language=language,
                    model_size=model_size,
                    **kwargs
                )
            
            # Handle errors in transcription result
            if isinstance(transcription_result, dict) and 'error' in transcription_result:
                logger.error(f"Transcription error: {transcription_result['error']}")
                return transcription_result
            
            # Apply Aya enhancement if requested and available
            if enable_aya_enhancement and self.aya_enhancer and language == 'ar':
                try:
                    logger.info("🎨 Applying Aya Arabic enhancement...")
                    enhanced_text = await self.aya_enhancer.enhance_arabic_text(
                        transcription_result.get('text', '')
                    )
                    transcription_result['enhanced_text'] = enhanced_text
                    transcription_result['aya_enhanced'] = True
                except Exception as e:
                    logger.error(f"Aya enhancement failed: {e}")
                    transcription_result['aya_enhanced'] = False
            
            # Apply voice analysis if requested and available
            enable_voice_analysis = kwargs.get('enable_voice_analysis', True)
            if enable_voice_analysis and self.voice_analysis_engine:
                try:
                    logger.info("🎤 Performing voice analysis...")
                    voice_analysis_result = await self.voice_analysis_engine.analyze_audio(
                        audio_path=audio_path,
                        language=language
                    )
                    transcription_result['voice_analysis'] = voice_analysis_result
                    transcription_result['voice_analysis_enabled'] = True
                except Exception as e:
                    logger.error(f"Voice analysis failed: {e}")
                    transcription_result['voice_analysis_enabled'] = False
                    transcription_result['voice_analysis_error'] = str(e)
            
            # Add processing metadata
            processing_time = time.time() - start_time
            transcription_result['processing_metadata'] = {
                'processing_time': processing_time,
                'processing_mode': processing_mode,
                'model_size': model_size,
                'language': language,
                'engine_used': self._get_engine_name(use_ultimate_arabic, use_advanced_arabic, use_enhanced_arabic),
                'aya_enhanced': transcription_result.get('aya_enhanced', False),
                'voice_analysis_enabled': transcription_result.get('voice_analysis_enabled', False)
            }
            
            # Add processing_info for compatibility with job processing system
            transcription_result['processing_info'] = {
                'processing_mode': processing_mode,
                'model_used': self._get_engine_name(use_ultimate_arabic, use_advanced_arabic, use_enhanced_arabic),
                'processing_time': processing_time,
                'language': language,
                'model_size': model_size,
                'engine_used': self._get_engine_name(use_ultimate_arabic, use_advanced_arabic, use_enhanced_arabic)
            }
            
            # Ensure model_size is available at root level for output generation
            if 'model_size' not in transcription_result:
                # Try to get model_size from various locations in the result
                model_size_value = (
                    transcription_result.get('metadata', {}).get('model_size') or  # Ultimate engine
                    transcription_result.get('processing_metadata', {}).get('model_size') or  # Current location
                    model_size  # Fallback to the parameter passed
                )
                transcription_result['model_size'] = model_size_value
            
            return transcription_result
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return {
                'error': str(e),
                'processing_time': time.time() - start_time
            }
        finally:
            # Cleanup memory after transcription
            self._cleanup_memory()
    
    def _get_engine_name(self, use_ultimate: bool, use_advanced: bool, use_enhanced: bool) -> str:
        """Get the name of the engine being used."""
        if use_ultimate:
            return "ultimate_arabic_v3"
        elif use_advanced:
            return "advanced_arabic_v2"
        elif use_enhanced:
            return "enhanced_arabic_v1"
        else:
            return "standard_whisper"

    def get_available_engines(self) -> Dict[str, bool]:
        """Get status of available engines."""
        return {
            'local': True,
            'api': True,
            'enhanced_arabic': self.enhanced_arabic_engine is not None,
            'advanced_arabic': self.advanced_arabic_engine is not None,
            'ultimate_arabic': self.ultimate_arabic_engine is not None,
            'aya_enhancer': self.aya_enhancer is not None,
            'voice_analysis': self.voice_analysis_engine is not None
        }

    def refresh_api_engine(self):
        """Refresh API engine configuration."""
        try:
            # Clear existing API engine
            if self.api_engine:
                del self.api_engine
                self.api_engine = None
            
            # Reinitialize with current config
            self.api_engine = APITranscriptionEngine()
            logger.info("API engine refreshed successfully")
            
            return {
                'success': True,
                'message': 'API engine refreshed successfully',
                'config': {
                    'api_provider': self.config.API_PROVIDER,
                    'api_configured': bool(self.config.OPENAI_API_KEY or self.config.AZURE_API_KEY)
                }
            }
        except Exception as e:
            logger.error(f"Failed to refresh API engine: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all engines."""
        return {
            'status': 'healthy',
            'engines': self.get_available_engines(),
            'memory_usage': psutil.Process().memory_info().rss / 1024 / 1024,  # MB
            'gpu_available': torch.cuda.is_available(),
            'gpu_memory': torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else None
        }

    async def shutdown(self):
        """Clean shutdown of all engines with comprehensive cleanup."""
        logger.info("Shutting down transcription service...")
        
        try:
            # Cleanup engines
            if self.local_engine:
                if hasattr(self.local_engine, 'cleanup'):
                    await self.local_engine.cleanup()
                del self.local_engine
                self.local_engine = None
            
            if self.enhanced_arabic_engine:
                if hasattr(self.enhanced_arabic_engine, 'cleanup'):
                    try:
                        await self.enhanced_arabic_engine.cleanup()
                    except:
                        pass
                del self.enhanced_arabic_engine
                self.enhanced_arabic_engine = None
            
            if self.advanced_arabic_engine:
                if hasattr(self.advanced_arabic_engine, 'cleanup'):
                    try:
                        await self.advanced_arabic_engine.cleanup()
                    except:
                        pass
                del self.advanced_arabic_engine
                self.advanced_arabic_engine = None
            
            if self.ultimate_arabic_engine:
                if hasattr(self.ultimate_arabic_engine, 'cleanup'):
                    try:
                        await self.ultimate_arabic_engine.cleanup()
                    except:
                        pass
                del self.ultimate_arabic_engine
                self.ultimate_arabic_engine = None
            
            if self.aya_enhancer:
                del self.aya_enhancer
                self.aya_enhancer = None
            
            # Force final cleanup
            self._cleanup_memory(force=True)
            
            logger.info("Transcription service shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

# Global instance
_service_instance = None

async def get_transcription_service() -> UnifiedTranscriptionService:
    """Get the global transcription service instance."""
    global _service_instance
    if _service_instance is None:
        _service_instance = UnifiedTranscriptionService()
    return _service_instance

# Graceful shutdown handler
def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown."""
    
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}")
        if _service_instance:
            asyncio.create_task(_service_instance.shutdown())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
