"""
Unified transcription service for Arabic STT Platform
Handles both local and API-based transcription with optional Aya enhancement
"""

import a        # Determine engine priority for Arabic: Ultimate > Advanced > Enhanced > Standard
        use_ultimate_arabic = (
            language == 'ar' and 
            processing_mode == 'local' and 
            getattr(self, 'ultimate_arabic_engine', None) is not None and
            kwargs.get('use_ultimate_arabic', True)  # Allow override
        )
        
        use_advanced_arabic = (
            language == 'ar' and 
            processing_mode == 'local' and 
            not use_ultimate_arabic and  # Fall back if ultimate not available
            self.advanced_arabic_engine is not None and
            kwargs.get('use_advanced_arabic', True)  # Allow override
        )
        
        use_enhanced_arabic = (
            language == 'ar' and 
            processing_mode == 'local' and 
            not use_ultimate_arabic and
            not use_advanced_arabic and  # Fall back to enhanced if advanced not available
            self.enhanced_arabic_engine is not None and
            kwargs.get('use_enhanced_arabic', True)  # Allow override
        )
        
        if use_ultimate_arabic:
            logger.info("🔥 Using Ultimate Arabic transcription engine v3.0")
            # Step 1: Ultimate Arabic transcription (highest quality)
            transcription_result = self.ultimate_arabic_engine.transcribe(audio_path)
        elif use_advanced_arabic:ging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import time
import signal
import psutil

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
    logger.warning("Enhanced Arabic transcription engine not available")

# Import advanced Arabic engine v2.0
try:
    from core.advanced_arabic_transcription_engine import AdvancedArabicTranscriptionEngine
    ADVANCED_ARABIC_AVAILABLE = True
except ImportError as e:
    ADVANCED_ARABIC_AVAILABLE = False
    # Will log warning after logger is initialized

# Import ultimate Arabic engine v3.0
try:
    from core.ultimate_arabic_transcription_engine import UltimateArabicTranscriptionEngine
    ULTIMATE_ARABIC_AVAILABLE = True
except ImportError as e:
    ULTIMATE_ARABIC_AVAILABLE = False
    # Will log warning after logger is initialized

logger = logging.getLogger(__name__)

# Log import warnings after logger is available
if not ENHANCED_ARABIC_AVAILABLE:
    logger.warning("Enhanced Arabic transcription engine not available")
if not ADVANCED_ARABIC_AVAILABLE:
    logger.warning("Advanced Arabic transcription engine v2.0 not available")
if not ULTIMATE_ARABIC_AVAILABLE:
    logger.warning("Ultimate Arabic transcription engine v3.0 not available")
try:
    from aya_arabic_enhancer import AyaArabicEnhancer
    AYA_AVAILABLE = True
except ImportError:
    AYA_AVAILABLE = False

logger = logging.getLogger(__name__)

class UnifiedTranscriptionService:
    """Unified service that can use either local models or cloud APIs with optional Aya enhancement."""
    
    def __init__(self):
        self.config = Config()
        self.time_estimator = TimeEstimationEngine()
        
        # Core transcription engines
        self.local_engine = TranscriptionEngine()
        self.api_engine = APITranscriptionEngine()
        
        # Enhanced Arabic engine (if available)
        self.enhanced_arabic_engine = None
        if ENHANCED_ARABIC_AVAILABLE:
            try:
                self.enhanced_arabic_engine = EnhancedArabicTranscriptionEngine()
                logger.info("✨ Enhanced Arabic transcription engine initialized")
            except Exception as e:
                logger.warning(f"⚠️ Enhanced Arabic engine initialization failed: {e}")
        
        # Advanced Arabic engine v2.0 (if available)
        self.advanced_arabic_engine = None
        if ADVANCED_ARABIC_AVAILABLE:
            try:
                self.advanced_arabic_engine = AdvancedArabicTranscriptionEngine()
                logger.info("✨ Advanced Arabic transcription engine v2.0 initialized")
            except Exception as e:
                logger.warning(f"⚠️ Advanced Arabic engine v2.0 initialization failed: {e}")
        
        # Aya enhancement (optional)
        self.aya_enhancer = None
        if AYA_AVAILABLE and getattr(self.config, 'AYA_ENHANCEMENT_ENABLED', False):
            try:
                self.aya_enhancer = AyaArabicEnhancer()
                logger.info("✨ Aya Arabic enhancement initialized")
            except Exception as e:
                logger.warning(f"⚠️ Aya enhancer initialization failed: {e}")
                self.aya_enhancer = None
        
        # Processing state
        self.current_process = None
        self.is_processing = False

    async def get_local_engine(self) -> TranscriptionEngine:
        """Get or create local transcription engine."""
        if self.local_engine is None:
            self.local_engine = TranscriptionEngine()
        return self.local_engine
    
    async def get_api_engine(self) -> APITranscriptionEngine:
        """Get or create API transcription engine."""
        if self.api_engine is None:
            self.api_engine = APITranscriptionEngine()
            # Pass our config instance to the API engine
            self.api_engine.config = self.config
        return self.api_engine
    
    def refresh_api_engine(self):
        """Refresh the API engine with new configuration."""
        self.api_engine = None
    
    async def transcribe(
        self,
        audio_path: str,
        processing_mode: Optional[str] = None,
        model_size: str = 'medium',
        language: str = 'ar',
        enable_aya_enhancement: bool = False,
        job_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Enhanced transcribe with optional Aya Arabic post-processing.
        
        Args:
            audio_path: Path to audio file
            processing_mode: 'local', 'api', or None (auto-detect)
            model_size: Whisper model size
            language: Language code
            enable_aya_enhancement: Enable Aya model post-processing
            job_id: Job ID for tracking time estimation accuracy
            **kwargs: Additional parameters
        
        Returns:
            Dict containing transcription results with optional enhancement
        """
        
        start_time = time.time()
        
        # Determine which Arabic transcription engine to use
        use_advanced_arabic = (
            language == 'ar' and 
            processing_mode == 'local' and 
            self.advanced_arabic_engine is not None and
            kwargs.get('use_advanced_arabic', True)  # Allow override
        )
        
        use_enhanced_arabic = (
            language == 'ar' and 
            processing_mode == 'local' and 
            not use_advanced_arabic and  # Fall back to enhanced if advanced not available
            self.enhanced_arabic_engine is not None and
            kwargs.get('use_enhanced_arabic', True)  # Allow override
        )
        
        if use_advanced_arabic:
            logger.info("🚀 Using Advanced Arabic transcription engine v2.0")
            # Step 1: Advanced Arabic transcription
            transcription_result = await self.advanced_arabic_engine.transcribe_arabic_advanced(
                audio_path=audio_path,
                model_size=model_size,
                enable_preprocessing=kwargs.get('enable_arabic_preprocessing', True),
                **{k: v for k, v in kwargs.items() if k not in ['use_advanced_arabic', 'use_enhanced_arabic', 'enable_arabic_preprocessing']}
            )
        elif use_enhanced_arabic:
            logger.info("🎯 Using Enhanced Arabic transcription engine")
            # Step 1: Enhanced Arabic transcription
            transcription_result = await self.enhanced_arabic_engine.transcribe_with_enhanced_arabic(
                audio_path=audio_path,
                model_size=model_size,
                enable_preprocessing=kwargs.get('enable_arabic_preprocessing', True),
                **{k: v for k, v in kwargs.items() if k not in ['use_enhanced_arabic', 'enable_arabic_preprocessing']}
            )
        else:
            logger.info(f"🎯 Using standard transcription (Aya enhancement: {'enabled' if enable_aya_enhancement else 'disabled'})")
            # Step 1: Standard transcription
            transcription_result = await self._original_transcribe(
                audio_path=audio_path,
                processing_mode=processing_mode,
                model_size=model_size,
                language=language,
                job_id=job_id,
                **kwargs
            )
        
        # Step 2: Apply Aya enhancement if requested and available
        if enable_aya_enhancement and self.aya_enhancer:
            try:
                logger.info("✨ Applying Aya Arabic enhancement...")
                
                # Ensure Aya model is available
                model_available = await self.aya_enhancer.ensure_model_available()
                
                if model_available:
                    enhancement_start = time.time()
                    
                    # Extract text for enhancement
                    original_text = transcription_result.get('text', '')
                    
                    if original_text.strip():
                        # Enhance the text
                        enhanced_result = await self.aya_enhancer.enhance_arabic_text(original_text)
                        
                        # Also enhance segments if available
                        segments = transcription_result.get('segments', [])
                        enhanced_segments = []
                        
                        if segments:
                            enhanced_segments = await self.aya_enhancer.enhance_transcript_segments(
                                segments, chunk_size=5
                            )
                        
                        enhancement_time = time.time() - enhancement_start
                        
                        # Add enhancement data to result
                        transcription_result['aya_enhancement'] = {
                            'enabled': True,
                            'model': self.aya_enhancer.model_name,
                            'processing_time': enhancement_time,
                            'quality_score': enhanced_result.get('quality_score', 0),
                            'improvements': enhanced_result.get('improvements', []),
                            'enhanced_text': enhanced_result.get('enhanced_text', original_text),
                            'enhanced_segments': enhanced_segments
                        }
                        
                        # Optionally replace the main text with enhanced version
                        if getattr(self.config, 'AYA_REPLACE_ORIGINAL', True):
                            transcription_result['original_text'] = original_text
                            transcription_result['text'] = enhanced_result.get('enhanced_text', original_text)
                            if enhanced_segments:
                                transcription_result['original_segments'] = segments
                                transcription_result['segments'] = enhanced_segments
                        
                        logger.info(f"✅ Aya enhancement completed in {enhancement_time:.1f}s")
                        logger.info(f"📊 Quality score: {enhanced_result.get('quality_score', 0):.2f}")
                    else:
                        logger.warning("⚠️ No text available for Aya enhancement")
                        transcription_result['aya_enhancement'] = {
                            'enabled': False,
                            'error': 'No text to enhance'
                        }
                else:
                    logger.warning("❌ Aya model not available for enhancement")
                    transcription_result['aya_enhancement'] = {
                        'enabled': False,
                        'error': 'Aya model not available'
                    }
                    
            except Exception as e:
                logger.error(f"❌ Aya enhancement failed: {e}")
                transcription_result['aya_enhancement'] = {
                    'enabled': False,
                    'error': str(e)
                }
        elif enable_aya_enhancement and not self.aya_enhancer:
            transcription_result['aya_enhancement'] = {
                'enabled': False,
                'error': 'Aya enhancer not available (check installation and config)'
            }
        
        # Add total processing time
        total_time = time.time() - start_time
        if 'processing_time' not in transcription_result:
            transcription_result['processing_time'] = total_time
        else:
            transcription_result['total_processing_time'] = total_time
        
        return transcription_result

    async def _original_transcribe(
        self,
        audio_path: str,
        processing_mode: Optional[str] = None,
        model_size: str = 'medium',
        language: str = 'ar',
        job_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Transcribe audio using specified processing mode.
        
        Args:
            audio_path: Path to audio file
            processing_mode: 'local' or 'api' (defaults to config setting)
            model_size: Model size for local processing
            language: Language code
            job_id: Job ID for tracking time estimation accuracy
            **kwargs: Additional parameters
            
        Returns:
            Dict containing transcript and metadata including time estimates
        """
        try:
            # Determine processing mode
            mode = processing_mode or self.config.PROCESSING_MODE
            
            logger.info(f"Starting transcription using {mode} mode")
            
            # Get time estimate before processing
            time_estimate = await self.time_estimator.estimate_processing_time(
                audio_path=audio_path,
                model_size=model_size,
                processing_mode=mode,
                options=kwargs
            )
            
            # Record start time
            start_time = time.time()
            
            # Calculate timeout based on file size and estimated time
            file_size_mb = Path(audio_path).stat().st_size / (1024 * 1024)
            base_timeout = max(time_estimate['estimated_time_seconds'] * 3, 300)  # 3x estimate or 5 min minimum
            
            # Add extra time for very large files
            if file_size_mb > 100:
                base_timeout += file_size_mb * 2  # 2 seconds per MB for large files
                logger.info(f"Large file detected ({file_size_mb:.1f}MB), using extended timeout: {base_timeout:.0f}s")
            
            # Perform transcription with timeout
            if mode == 'local':
                result = await asyncio.wait_for(
                    self._transcribe_local(audio_path, model_size, language, **kwargs),
                    timeout=base_timeout
                )
            elif mode == 'api':
                result = await asyncio.wait_for(
                    self._transcribe_api(audio_path, language, **kwargs),
                    timeout=min(base_timeout, 1800)  # API max 30 minutes
                )
            else:
                raise ValueError(f"Invalid processing mode: {mode}")
                
            return result
                
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            raise

    async def _transcribe_local(
        self,
        audio_path: str,
        model_size: str = 'medium',
        language: str = 'ar',
        **kwargs
    ) -> Dict[str, Any]:
        """Transcribe using local Whisper model."""
        local_engine = await self.get_local_engine()
        return await local_engine.transcribe(
            audio_path=audio_path,
            model_size=model_size,
            language=language,
            **kwargs
        )
    
    async def _transcribe_api(
        self,
        audio_path: str,
        language: str = 'ar',
        **kwargs
    ) -> Dict[str, Any]:
        """Transcribe using API service."""
        api_engine = await self.get_api_engine()
        return await api_engine.transcribe_openai(
            audio_path=audio_path,
            language=language,
            **kwargs
        )

    # Additional utility methods for process management
    def cancel_current_process(self):
        """Cancel current transcription process."""
        if self.current_process and not self.current_process.done():
            self.current_process.cancel()
            self.is_processing = False
            logger.info("Current transcription process cancelled")

    def get_processing_status(self) -> Dict[str, Any]:
        """Get current processing status."""
        return {
            'is_processing': self.is_processing,
            'current_process_exists': self.current_process is not None,
            'process_done': self.current_process.done() if self.current_process else True,
            'aya_available': AYA_AVAILABLE,
            'aya_enabled': self.aya_enhancer is not None
        }
