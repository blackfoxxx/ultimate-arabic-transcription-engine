"""
Enhanced Arabic Transcription Engine
Optimized specifically for accurate Arabic speech-to-text transcription
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
import librosa
import numpy as np

from config import Config

logger = logging.getLogger(__name__)

class EnhancedArabicTranscriptionEngine:
    """Enhanced transcription engine optimized specifically for Arabic."""
    
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
                logger.info("🚀 CUDA available, using GPU acceleration for Arabic transcription")
            else:
                device = 'cpu'
                logger.info("💻 Using CPU for Arabic transcription")
        
        return device
    
    def _get_enhanced_arabic_prompt(self) -> str:
        """Get comprehensive Arabic prompt for better transcription accuracy."""
        return (
            "هذا تسجيل صوتي باللغة العربية الفصحى والعامية. "
            "يرجى كتابة النص بدقة عالية مع مراعاة: "
            "الكلمات العربية الصحيحة، علامات الترقيم، الأرقام بالعربية، "
            "أسماء الأعلام والأماكن. النص يحتوي على محادثة طبيعية باللغة العربية."
        )
    
    def _preprocess_audio_for_arabic(self, audio_path: str) -> str:
        """Preprocess audio specifically for Arabic transcription."""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=16000)  # Whisper expects 16kHz
            
            # Audio enhancement for Arabic speech
            # 1. Normalize audio levels
            y = librosa.util.normalize(y)
            
            # 2. Reduce noise (helps with Arabic consonants)
            y = librosa.effects.preemphasis(y, coef=0.97)
            
            # 3. Trim silence (important for Arabic speech patterns)
            y, _ = librosa.effects.trim(y, top_db=20)
            
            # Save preprocessed audio
            output_path = audio_path.replace('.', '_arabic_enhanced.')
            try:
                import soundfile as sf
                sf.write(output_path, y, sr)
                return output_path
            except ImportError:
                logger.warning("soundfile not available, using librosa for audio saving")
                import librosa
                librosa.output.write_wav(output_path, y, sr)
                return output_path
            
        except Exception as e:
            logger.warning(f"Audio preprocessing failed, using original: {e}")
            return audio_path
    
    async def load_model(self, model_size: str = None) -> None:
        """Load Whisper model optimized for Arabic."""
        # Use config default if no model_size provided
        if model_size is None:
            model_size = self.config.ENHANCED_ARABIC_MODEL_SIZE
            
        try:
            if self.model is None or self.current_model_size != model_size:
                logger.info(f"🔄 Loading Arabic-optimized Whisper model: {model_size}")
                
                # Unload previous model
                if self.model is not None:
                    del self.model
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                # Optimize compute type for Arabic
                if self.device == "cuda":
                    compute_type = "float16"  # Better for Arabic with GPU
                else:
                    compute_type = "int8"  # Efficient for CPU
                
                # Load model with Arabic optimizations
                self.model = WhisperModel(
                    model_size,
                    device=self.device,
                    compute_type=compute_type,
                    download_root=str(self.config.MODELS_FOLDER),
                    cpu_threads=min(8, psutil.cpu_count()),  # Optimize for Arabic processing
                    num_workers=1  # Better for Arabic long-form audio
                )
                
                self.current_model_size = model_size
                logger.info(f"✅ Arabic-optimized model {model_size} loaded on {self.device}")
                
        except Exception as e:
            logger.error(f"❌ Failed to load Arabic model {model_size}: {str(e)}")
            raise
    
    async def transcribe_arabic(
        self,
        audio_path: str,
        model_size: str = None,
        enable_preprocessing: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Enhanced Arabic transcription with optimizations.
        
        Args:
            audio_path: Path to audio file
            model_size: Whisper model size (recommend 'medium' or 'large-v2' for Arabic)
            enable_preprocessing: Enable Arabic audio preprocessing
            **kwargs: Additional parameters
            
        Returns:
            Dict containing enhanced transcript and metadata
        """
        try:
            # Load model
            await self.load_model(model_size)
            
            logger.info(f"🎤 Starting enhanced Arabic transcription of {Path(audio_path).name}")
            
            # Preprocess audio for Arabic if enabled
            processed_audio_path = audio_path
            if enable_preprocessing:
                logger.info("🔧 Preprocessing audio for Arabic speech patterns...")
                processed_audio_path = self._preprocess_audio_for_arabic(audio_path)
            
            # File size analysis
            file_size_mb = Path(processed_audio_path).stat().st_size / (1024 * 1024)
            logger.info(f"📊 Audio file: {file_size_mb:.1f}MB")
            
            # Memory optimization for large files
            if file_size_mb > 50:
                logger.info("💾 Large file detected, optimizing memory usage...")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            start_time = time.time()
            
            # Arabic-optimized transcription parameters
            arabic_params = {
                # Core parameters optimized for Arabic
                'beam_size': 5,  # Higher beam size for better Arabic accuracy
                'best_of': 5,    # Multiple candidates for Arabic word selection
                'patience': 2.0,  # More patience for Arabic speech patterns
                
                # Temperature progression for Arabic
                'temperature': [0.0, 0.1, 0.3, 0.5, 0.7],  # More conservative for Arabic
                
                # Threshold tuning for Arabic
                'compression_ratio_threshold': 2.0,  # Lower threshold for Arabic complexity
                'log_prob_threshold': -0.8,         # Higher confidence threshold
                'no_speech_threshold': 0.5,         # Lower threshold for Arabic speech detection
                
                # Arabic context settings
                'condition_on_previous_text': True,  # Important for Arabic context
                'initial_prompt': self._get_enhanced_arabic_prompt(),
                'language': 'ar',  # Explicitly set Arabic
                
                # Timestamp settings optimized for Arabic
                'word_timestamps': True,
                'prepend_punctuations': "\"'([{-",
                'append_punctuations': "\"'.،؟!)}]-",
                
                # VAD settings for Arabic speech
                'vad_filter': True,
                'vad_parameters': {
                    'threshold': 0.4,  # Adjusted for Arabic speech patterns
                    'min_speech_duration_ms': 100,
                    'min_silence_duration_ms': 300,  # Arabic has different pause patterns
                    'speech_pad_ms': 300
                },
                
                # Hallucination prevention for Arabic
                'hallucination_silence_threshold': 0.8,
                # Remove problematic clip_timestamps for now
                
                # Override any provided parameters
                **kwargs
            }
            
            logger.info("🚀 Running Arabic-optimized transcription...")
            
            # Execute transcription
            segments_generator, info = self.model.transcribe(
                processed_audio_path,
                **arabic_params
            )
            
            # Process segments with Arabic-specific enhancements
            segments = []
            full_text = ""
            word_count = 0
            
            for segment in segments_generator:
                # Post-process Arabic text
                arabic_text = self._post_process_arabic_text(segment.text.strip())
                
                segment_data = {
                    'id': segment.id,
                    'start': segment.start,
                    'end': segment.end,
                    'text': arabic_text,
                    'original_text': segment.text.strip(),  # Keep original for comparison
                    'avg_logprob': segment.avg_logprob,
                    'compression_ratio': segment.compression_ratio,
                    'no_speech_prob': segment.no_speech_prob,
                    'confidence': self._calculate_arabic_confidence(segment),
                    'words': []
                }
                
                # Process word-level timestamps with Arabic enhancements
                if hasattr(segment, 'words') and segment.words:
                    for word in segment.words:
                        word_data = {
                            'start': word.start,
                            'end': word.end,
                            'word': self._post_process_arabic_word(word.word),
                            'probability': word.probability
                        }
                        segment_data['words'].append(word_data)
                        word_count += 1
                
                segments.append(segment_data)
                full_text += arabic_text + " "
            
            # Clean up preprocessed audio if it was created
            if enable_preprocessing and processed_audio_path != audio_path:
                try:
                    Path(processed_audio_path).unlink()
                except:
                    pass
            
            processing_time = time.time() - start_time
            
            # Calculate Arabic-specific quality metrics
            arabic_quality = self._assess_arabic_transcription_quality(segments, full_text.strip())
            
            result = {
                'text': full_text.strip(),
                'segments': segments,
                'language': info.language,
                'language_probability': info.language_probability,
                'duration': info.duration,
                'processing_time': processing_time,
                'model_size': model_size,
                'device': self.device,
                'preprocessing_enabled': enable_preprocessing,
                'arabic_quality_metrics': arabic_quality,
                'metadata': {
                    'total_segments': len(segments),
                    'total_words': word_count,
                    'avg_confidence': sum(s['confidence'] for s in segments) / len(segments) if segments else 0,
                    'file_size_mb': file_size_mb,
                    'transcription_speed': info.duration / processing_time if processing_time > 0 else 0
                }
            }
            
            logger.info(f"✅ Arabic transcription completed in {processing_time:.2f}s")
            logger.info(f"📝 Language: {info.language} (confidence: {info.language_probability:.3f})")
            logger.info(f"⏱️ Duration: {info.duration:.2f}s, Segments: {len(segments)}, Words: {word_count}")
            logger.info(f"🎯 Arabic quality score: {arabic_quality.get('overall_score', 0):.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Arabic transcription failed: {str(e)}")
            raise
    
    def _post_process_arabic_text(self, text: str) -> str:
        """Post-process Arabic text for better accuracy."""
        if not text:
            return text
        
        # Basic Arabic text cleaning
        # Remove extra spaces
        text = ' '.join(text.split())
        
        # Fix common Arabic transcription issues
        # Add more specific Arabic corrections here based on common Whisper errors
        corrections = {
            # Common Whisper Arabic mistakes
            ' ا ': ' ',  # Remove standalone alif
            '  ': ' ',   # Multiple spaces
        }
        
        for mistake, correction in corrections.items():
            text = text.replace(mistake, correction)
        
        return text.strip()
    
    def _post_process_arabic_word(self, word: str) -> str:
        """Post-process individual Arabic words."""
        return word.strip()
    
    def _calculate_arabic_confidence(self, segment) -> float:
        """Calculate confidence score adjusted for Arabic characteristics."""
        # Base confidence from log probability
        base_confidence = max(0.0, min(1.0, (segment.avg_logprob + 1.0)))
        
        # Adjust for compression ratio (Arabic should have moderate compression)
        compression_penalty = 0.0
        if segment.compression_ratio > 3.0:  # Too high compression might indicate errors
            compression_penalty = (segment.compression_ratio - 3.0) * 0.1
        elif segment.compression_ratio < 1.5:  # Too low might indicate missed content
            compression_penalty = (1.5 - segment.compression_ratio) * 0.1
        
        # Adjust for no-speech probability
        speech_confidence = 1.0 - segment.no_speech_prob
        
        # Combined confidence
        final_confidence = max(0.0, min(1.0, base_confidence * speech_confidence - compression_penalty))
        
        return final_confidence
    
    def _assess_arabic_transcription_quality(self, segments: List[Dict], full_text: str) -> Dict[str, Any]:
        """Assess the quality of Arabic transcription."""
        if not segments or not full_text:
            return {'overall_score': 0.0, 'details': 'No content to assess'}
        
        # Basic quality metrics for Arabic
        total_confidence = sum(s.get('confidence', 0) for s in segments)
        avg_confidence = total_confidence / len(segments)
        
        # Arabic character ratio (good transcription should be mostly Arabic)
        arabic_chars = sum(1 for char in full_text if '\u0600' <= char <= '\u06FF')
        total_chars = len(full_text.replace(' ', ''))
        arabic_ratio = arabic_chars / total_chars if total_chars > 0 else 0
        
        # Segment length distribution (good Arabic should have reasonable segment lengths)
        segment_lengths = [len(s.get('text', '')) for s in segments]
        avg_segment_length = sum(segment_lengths) / len(segment_lengths) if segment_lengths else 0
        
        # Overall quality score
        overall_score = (avg_confidence * 0.5 + arabic_ratio * 0.3 + min(avg_segment_length / 50, 1.0) * 0.2)
        
        return {
            'overall_score': overall_score,
            'avg_confidence': avg_confidence,
            'arabic_character_ratio': arabic_ratio,
            'avg_segment_length': avg_segment_length,
            'total_arabic_characters': arabic_chars,
            'recommendation': 'good' if overall_score > 0.7 else 'moderate' if overall_score > 0.4 else 'poor'
        }
