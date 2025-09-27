"""
Transcription engine using faster-whisper for Arabic STT Platform
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import torch
from faster_whisper import WhisperModel
import time
import psutil
import gc

from config import Config

logger = logging.getLogger(__name__)

class TranscriptionEngine:
    """Handles speech-to-text transcription using Whisper."""
    
    def __init__(self):
        self.config = Config()
        self.model = None
        self.current_model_size = None
        self.device = self._determine_device()
        
    def _determine_device(self) -> str:
        """Determine the best device for inference."""
        device = self.config.WHISPER_DEVICE
        
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
                logger.info("CUDA available, using GPU acceleration")
            else:
                device = 'cpu'
                logger.info("CUDA not available, using CPU")
        
        return device
    
    async def load_model(self, model_size: str = 'medium') -> None:
        """Load or switch Whisper model."""
        try:
            if self.model is None or self.current_model_size != model_size:
                logger.info(f"Loading Whisper model: {model_size}")
                
                # Unload previous model if exists
                if self.model is not None:
                    del self.model
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                # Load new model
                compute_type = "float16" if self.device == "cuda" else "int8"
                
                self.model = WhisperModel(
                    model_size,
                    device=self.device,
                    compute_type=compute_type,
                    download_root=str(self.config.MODELS_FOLDER)
                )
                
                self.current_model_size = model_size
                logger.info(f"Model {model_size} loaded successfully on {self.device}")
                
        except Exception as e:
            logger.error(f"Failed to load model {model_size}: {str(e)}")
            raise
    
    async def transcribe(
        self,
        audio_path: str,
        model_size: str = 'medium',
        language: str = 'ar',
        **kwargs
    ) -> Dict[str, Any]:
        """
        Transcribe audio file to text.
        
        Args:
            audio_path: Path to audio file
            model_size: Whisper model size
            language: Language code ('ar' for Arabic, 'auto' for auto-detect)
            **kwargs: Additional parameters
            
        Returns:
            Dict containing transcript and metadata
        """
        try:
            # Load model if needed
            await self.load_model(model_size)
            
            logger.info(f"Starting transcription of {audio_path}")
            
            # Memory monitoring for large files
            audio_path_obj = Path(audio_path)
            file_size_mb = audio_path_obj.stat().st_size / (1024 * 1024)
            
            if file_size_mb > 100:  # Large file (100MB+)
                logger.info(f"Large file detected ({file_size_mb:.1f}MB), optimizing memory usage")
                gc.collect()  # Force garbage collection
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            start_time = time.time()
            memory_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            # Prepare transcription parameters with optimizations for large files
            transcribe_params = {
                'beam_size': kwargs.get('beam_size', 3 if file_size_mb > 100 else 5),  # Reduce for large files
                'best_of': kwargs.get('best_of', 3 if file_size_mb > 100 else 5),
                'temperature': kwargs.get('temperature', [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
                'compression_ratio_threshold': kwargs.get('compression_ratio_threshold', 2.4),
                'log_prob_threshold': kwargs.get('log_prob_threshold', -1.0),
                'no_speech_threshold': kwargs.get('no_speech_threshold', 0.6),
                'condition_on_previous_text': kwargs.get('condition_on_previous_text', True),
                'initial_prompt': self._get_arabic_prompt() if language == 'ar' else None,
                'word_timestamps': True,
                'vad_filter': True,
                'vad_parameters': {
                    'min_silence_duration_ms': 1000 if file_size_mb > 100 else 500,  # Longer silence for large files
                    'speech_pad_ms': 200 if file_size_mb > 100 else 400
                }
            }
            
            # Set language
            if language != 'auto':
                transcribe_params['language'] = language
            
            # Run transcription
            segments_generator, info = self.model.transcribe(
                audio_path,
                **transcribe_params
            )
            
            # Process segments
            segments = []
            full_text = ""
            
            for segment in segments_generator:
                segment_data = {
                    'id': segment.id,
                    'start': segment.start,
                    'end': segment.end,
                    'text': segment.text.strip(),
                    'avg_logprob': segment.avg_logprob,
                    'compression_ratio': segment.compression_ratio,
                    'no_speech_prob': segment.no_speech_prob,
                    'words': []
                }
                
                # Add word-level timestamps
                if hasattr(segment, 'words') and segment.words:
                    for word in segment.words:
                        word_data = {
                            'start': word.start,
                            'end': word.end,
                            'word': word.word,
                            'probability': word.probability
                        }
                        segment_data['words'].append(word_data)
                
                segments.append(segment_data)
                full_text += segment.text.strip() + " "
            
            processing_time = time.time() - start_time
            memory_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            memory_used = memory_after - memory_before
            
            # Memory cleanup for large files
            if file_size_mb > 100:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info(f"Memory usage: {memory_used:.1f}MB peak during transcription")
            
            # Prepare result
            result = {
                'text': full_text.strip(),
                'segments': segments,
                'language': info.language,
                'language_probability': info.language_probability,
                'duration': info.duration,
                'processing_time': processing_time,
                'model_size': model_size,
                'device': self.device,
                'file_size_mb': file_size_mb,
                'memory_usage_mb': memory_used,
                'transcript_metadata': {
                    'avg_logprob': sum(s['avg_logprob'] for s in segments) / len(segments) if segments else 0,
                    'compression_ratio': sum(s['compression_ratio'] for s in segments) / len(segments) if segments else 0,
                    'no_speech_prob': sum(s['no_speech_prob'] for s in segments) / len(segments) if segments else 0,
                    'total_segments': len(segments),
                    'total_words': sum(len(s['words']) for s in segments)
                }
            }
            
            logger.info(f"Transcription completed in {processing_time:.2f}s")
            logger.info(f"Language: {info.language} (confidence: {info.language_probability:.2f})")
            logger.info(f"Duration: {info.duration:.2f}s, Segments: {len(segments)}")
            if file_size_mb > 50:
                logger.info(f"File size: {file_size_mb:.1f}MB, Memory used: {memory_used:.1f}MB")
            
            return result
            
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            raise
    
    def _get_arabic_prompt(self) -> str:
        """Get initial prompt optimized for Arabic transcription."""
        # This prompt helps Whisper understand the context for Arabic
        return (
            "هذا تسجيل صوتي باللغة العربية. "
            "الرجاء كتابة النص بدقة مع علامات الترقيم المناسبة."
        )
    
    async def get_supported_languages(self) -> Dict[str, str]:
        """Get list of supported languages."""
        # Load a small model to get language info
        if self.model is None:
            await self.load_model('tiny')
        
        # Whisper supported languages (subset relevant to Arabic regions)
        languages = {
            'ar': 'Arabic',
            'en': 'English',
            'fr': 'French',
            'es': 'Spanish',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'ru': 'Russian',
            'tr': 'Turkish',
            'fa': 'Persian',
            'ur': 'Urdu',
            'he': 'Hebrew'
        }
        
        return languages
    
    def get_model_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the currently loaded model."""
        if self.model is None:
            return None
        
        return {
            'model_size': self.current_model_size,
            'device': self.device,
            'compute_type': getattr(self.model, 'compute_type', 'unknown'),
            'model_path': str(self.config.MODELS_FOLDER)
        }
    
    async def validate_transcription_quality(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and assess transcription quality."""
        try:
            segments = result.get('segments', [])
            metadata = result.get('transcript_metadata', {})
            
            quality_metrics = {
                'overall_score': 0.0,
                'language_confidence': result.get('language_probability', 0.0),
                'avg_segment_confidence': metadata.get('avg_logprob', 0.0),
                'speech_ratio': 1.0 - metadata.get('no_speech_prob', 0.0),
                'compression_quality': metadata.get('compression_ratio', 0.0),
                'word_count': metadata.get('total_words', 0),
                'segment_count': metadata.get('total_segments', 0),
                'warnings': []
            }
            
            # Calculate overall quality score
            confidence_score = max(0, min(1, (quality_metrics['avg_segment_confidence'] + 4) / 4))
            language_score = quality_metrics['language_confidence']
            speech_score = quality_metrics['speech_ratio']
            
            quality_metrics['overall_score'] = (
                confidence_score * 0.5 +
                language_score * 0.3 +
                speech_score * 0.2
            )
            
            # Generate warnings
            if quality_metrics['language_confidence'] < 0.8:
                quality_metrics['warnings'].append("Low language detection confidence")
            
            if quality_metrics['avg_segment_confidence'] < -0.8:
                quality_metrics['warnings'].append("Low transcription confidence")
            
            if quality_metrics['compression_quality'] > 3.0:
                quality_metrics['warnings'].append("Possible repetitive content detected")
            
            if quality_metrics['word_count'] < 10:
                quality_metrics['warnings'].append("Very short transcript - check audio quality")
            
            return quality_metrics
            
        except Exception as e:
            logger.error(f"Quality validation failed: {str(e)}")
            return {'overall_score': 0.0, 'error': str(e)}
    
    def cleanup(self):
        """Cleanup resources."""
        try:
            if hasattr(self, 'model') and self.model is not None:
                del self.model
                self.model = None
                self.current_model_size = None
                
                # Clear GPU cache if using CUDA
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                logger.info("Transcription engine cleaned up")
                
        except Exception as e:
            logger.error(f"Cleanup error: {str(e)}")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()
