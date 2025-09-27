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

logger = logging.getLogger(__name__)

class UnifiedTranscriptionService:
    """Unified service that can use either local models or cloud APIs with optional Aya enhancement."""
    
    def __init__(self):
        self.config = Config()
        self.local_engine = None
        self.api_engine = None
        self.time_estimator = TimeEstimationEngine()
        self.aya_enhancer = None
        self.enhanced_arabic_engine = None
        self.advanced_arabic_engine = None
        self.ultimate_arabic_engine = None
        
        # Initialize Aya enhancer
        if AYA_AVAILABLE:
            try:
                self.aya_enhancer = AyaArabicEnhancer()
                logger.info("✨ Aya Arabic enhancement initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Aya enhancer: {e}")
                self.aya_enhancer = None
        
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
                self.ultimate_arabic_engine = UltimateArabicTranscriptionEngine(device="cpu")
                if self.ultimate_arabic_engine.initialize_model():
                    logger.info("🔥 Ultimate Arabic transcription engine v3.0 initialized")
                else:
                    self.ultimate_arabic_engine = None
                    logger.error("Failed to initialize Ultimate Arabic engine v3.0 model")
            except Exception as e:
                logger.error(f"Failed to initialize Ultimate Arabic engine v3.0: {e}")
                self.ultimate_arabic_engine = None
    
    async def get_local_engine(self) -> TranscriptionEngine:
        """Get the local transcription engine."""
        if self.local_engine is None:
            self.local_engine = TranscriptionEngine(Config())
            await self.local_engine.initialize()
        return self.local_engine
    
    async def get_api_engine(self) -> APITranscriptionEngine:
        """Get the API transcription engine."""
        if self.api_engine is None:
            self.api_engine = APITranscriptionEngine()
        return self.api_engine
    
    def get_estimated_time(self, audio_duration: float, processing_mode: str = 'local', model_size: str = 'small') -> Dict[str, Any]:
        """Get estimated processing time."""
        return self.time_estimator.estimate_time(
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
        model_size: str = 'small',
        **kwargs
    ) -> Dict[str, Any]:
        """
        Transcribe audio with the best available engine for the language.
        
        For Arabic language:
        Priority: Ultimate Arabic v3.0 > Advanced Arabic v2.0 > Enhanced Arabic v1.0 > Standard Whisper
        """
        start_time = time.time()
        
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
            # Ultimate Arabic transcription - highest quality
            transcription_result = self.ultimate_arabic_engine.transcribe(audio_path)
            
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
        
        # Apply Aya enhancement if requested and available
        if enable_aya_enhancement and self.aya_enhancer:
            try:
                logger.info("🎨 Applying Aya Arabic enhancement...")
                enhanced_result = await self.aya_enhancer.enhance_arabic_transcript(
                    transcription_result,
                    audio_path=audio_path
                )
                
                if enhanced_result and enhanced_result.get('success'):
                    transcription_result = enhanced_result
                    logger.info("✨ Aya enhancement completed successfully")
                else:
                    logger.warning("Aya enhancement failed, using original result")
                    
            except Exception as e:
                logger.error(f"Aya enhancement error: {e}")
        
        # Add processing metadata
        processing_time = time.time() - start_time
        if 'metadata' not in transcription_result:
            transcription_result['metadata'] = {}
        transcription_result['metadata']['total_processing_time'] = processing_time
        
        # Add engine information
        if use_ultimate_arabic:
            transcription_result['metadata']['engine'] = 'ultimate_arabic_v3'
            transcription_result['metadata']['quality_level'] = 'maximum'
        elif use_advanced_arabic:
            transcription_result['metadata']['engine'] = 'advanced_arabic_v2'
            transcription_result['metadata']['quality_level'] = 'high'
        elif use_enhanced_arabic:
            transcription_result['metadata']['engine'] = 'enhanced_arabic_v1'
            transcription_result['metadata']['quality_level'] = 'enhanced'
        else:
            transcription_result['metadata']['engine'] = 'standard_whisper'
            transcription_result['metadata']['quality_level'] = 'standard'
        
        return transcription_result
    
    def get_available_engines(self) -> Dict[str, bool]:
        """Get status of available engines."""
        return {
            'standard': True,
            'enhanced_arabic': self.enhanced_arabic_engine is not None,
            'advanced_arabic': self.advanced_arabic_engine is not None,
            'ultimate_arabic': self.ultimate_arabic_engine is not None,
            'aya_enhancement': self.aya_enhancer is not None
        }
    
    def refresh_api_engine(self):
        """Refresh API engine configuration"""
        if self.api_engine:
            self.api_engine = None
        """Get comprehensive engine statistics."""
        engines = self.get_available_engines()
        
        # Determine best Arabic engine
        best_arabic_engine = 'standard'
        if engines['ultimate_arabic']:
            best_arabic_engine = 'ultimate_arabic_v3'
        elif engines['advanced_arabic']:
            best_arabic_engine = 'advanced_arabic_v2'
        elif engines['enhanced_arabic']:
            best_arabic_engine = 'enhanced_arabic_v1'
        
        return {
            'available_engines': engines,
            'best_arabic_engine': best_arabic_engine,
            'aya_enhancement': engines['aya_enhancement'],
            'total_engines': sum(engines.values()),
            'arabic_optimization_level': 'maximum' if engines['ultimate_arabic'] else 
                                       'high' if engines['advanced_arabic'] else
                                       'enhanced' if engines['enhanced_arabic'] else 'standard'
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        health_status = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'status': 'healthy',
            'engines': self.get_available_engines(),
            'memory_usage': f"{psutil.virtual_memory().percent}%",
            'disk_usage': f"{psutil.disk_usage('/').percent}%"
        }
        
        # Test local engine
        try:
            local_engine = await self.get_local_engine()
            if local_engine:
                health_status['local_engine'] = 'available'
            else:
                health_status['local_engine'] = 'unavailable'
                health_status['status'] = 'degraded'
        except Exception as e:
            health_status['local_engine'] = f'error: {str(e)}'
            health_status['status'] = 'degraded'
        
        return health_status
    
    async def shutdown(self):
        """Clean shutdown of all engines."""
        logger.info("Shutting down transcription service...")
        
        # Cleanup engines
        if self.local_engine:
            await self.local_engine.cleanup()
        
        if hasattr(self.ultimate_arabic_engine, 'cleanup'):
            try:
                await self.ultimate_arabic_engine.cleanup()
            except:
                pass
        
        logger.info("Transcription service shutdown complete")

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
