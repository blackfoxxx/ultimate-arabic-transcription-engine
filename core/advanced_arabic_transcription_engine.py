"""
Advanced Arabic Transcription Engine v2.0
Completely redesigned for superior Arabic speech-to-text accuracy
"""

import asyncio
import logging
import torch
from faster_whisper import WhisperModel
from pathlib import Path
import librosa
import numpy as np
import soundfile as sf
import re
from typing import Dict, List, Any, Optional, Tuple
import time
import gc

logger = logging.getLogger(__name__)

class AdvancedArabicTranscriptionEngine:
    """Advanced Arabic transcription engine with superior accuracy optimizations."""
    
    def __init__(self):
        self.model = None
        self.current_model_size = None
        self.device = self._determine_device()
        
        # Advanced Arabic-specific parameters
        self.arabic_optimized_params = {
            # Whisper decoding parameters optimized for Arabic
            "beam_size": 10,  # Increased for better Arabic beam search
            "patience": 3.0,  # More patience for Arabic word formation
            "length_penalty": 1.2,  # Encourage longer, more complete Arabic words
            "temperature": [0.0, 0.1, 0.2, 0.4, 0.6],  # Conservative temperature progression
            "compression_ratio_threshold": 2.4,  # Adjusted for Arabic text density
            "log_prob_threshold": -0.8,  # More lenient for Arabic phonemes
            "no_speech_threshold": 0.6,  # Adjusted for Arabic content detection
            "condition_on_previous_text": True,  # Important for Arabic context
            "initial_prompt": None,  # Will be set dynamically
            "prefix": None,
            "suppress_blank": True,
            "suppress_tokens": [-1],  # Suppress problematic tokens
            "without_timestamps": False,
            "max_initial_timestamp": 1.0,
            "word_timestamps": True,  # Enable for better Arabic word boundaries
            "prepend_punctuations": "\"'([{-",
            "append_punctuations": "\"'.،,!?:)]}-",
            "vad_filter": True,
            "vad_parameters": {
                "threshold": 0.4,  # Adjusted for Arabic speech patterns
                "min_speech_duration_ms": 250,
                "max_speech_duration_s": 30,
                "min_silence_duration_ms": 500,  # Longer silence detection
                "speech_pad_ms": 400
            }
        }
        
        # Comprehensive Arabic prompts for different contexts
        self.arabic_prompts = {
            "formal": (
                "هذا تسجيل صوتي باللغة العربية الفصحى. النص يحتوي على محادثة رسمية أو خطاب أو "
                "مقابلة إعلامية. يرجى كتابة النص بدقة عالية مع مراعاة القواعد النحوية والإملائية "
                "الصحيحة، استخدام علامات الترقيم المناسبة، كتابة الأرقام بالعربية، والحفاظ على "
                "أسماء الأعلام والأماكن كما وردت. النص باللغة العربية فقط."
            ),
            "dialect": (
                "هذا تسجيل صوتي باللهجة العربية المحكية والعامية. قد يحتوي على مصطلحات محلية "
                "وتعبيرات عامية. يرجى كتابة النص كما يُنطق بدقة مع مراعاة اللهجة المحلية، "
                "استخدام الكلمات العربية الصحيحة، تجنب الكلمات الأجنبية قدر الإمكان، والحفاظ "
                "على السياق الطبيعي للمحادثة. النص باللغة العربية."
            ),
            "mixed": (
                "هذا تسجيل صوتي باللغة العربية قد يحتوي على خليط من الفصحى والعامية. "
                "يرجى كتابة النص بأفضل شكل ممكن مع التركيز على الوضوح والدقة، استخدام "
                "الكلمات العربية الصحيحة، تجنب الخلط مع لغات أخرى، والحفاظ على معنى "
                "المحادثة. النص يجب أن يكون باللغة العربية فقط."
            ),
            "news": (
                "هذا تسجيل إخباري أو تقرير إعلامي باللغة العربية. يحتوي على أخبار ومعلومات "
                "وقد يتضمن أسماء أعلام وأماكن ومصطلحات إعلامية. يرجى كتابة النص بدقة عالية "
                "مع مراعاة الأسلوب الإعلامي، الحفاظ على أسماء الأشخاص والأماكن، استخدام "
                "الأرقام والتواريخ بالشكل الصحيح. النص باللغة العربية الفصحى."
            )
        }
        
        # Arabic-specific post-processing patterns
        self.arabic_cleanup_patterns = [
            # Remove random English words mixed in Arabic text
            (r'\b[a-zA-Z]{1,3}\b(?=\s*[\u0600-\u06FF])', ''),
            (r'(?<=[\u0600-\u06FF])\s*[a-zA-Z]{1,4}\s*(?=[\u0600-\u06FF])', ' '),
            # Fix common Arabic transcription errors
            (r'([أإآا])\s+([لل])\s+([ل])', r'\1\2\3'),  # Fix fragmented "الل"
            (r'(\s|^)ا\s+([لنمتبكسف])', r'\1ا\2'),  # Fix fragmented Arabic prefixes
            (r'([تن])\s+([اأإآ])', r'\1\2'),  # Fix fragmented Arabic combinations
            # Remove excessive repetition
            (r'(\b[\u0600-\u06FF]+\b)\s+\1\s+\1', r'\1'),
            # Clean up punctuation
            (r'\s+([،؛؟!.])', r'\1'),
            (r'([،؛؟!.])\s*([،؛؟!.])', r'\1'),
            # Fix word boundaries
            (r'([ال])\s+([بتثجحخدذرزسشصضطظعغفقكلمنهويى])', r'\1\2'),
        ]

    def _determine_device(self) -> str:
        """Determine optimal device for Arabic transcription."""
        if torch.cuda.is_available():
            logger.info("🚀 Using GPU acceleration for Arabic transcription")
            return "cuda"
        else:
            # Use CPU for better compatibility with faster-whisper
            logger.info("💻 Using CPU for Arabic transcription (MPS not yet supported by faster-whisper)")
            return "cpu"
    
    def _select_optimal_prompt(self, audio_duration: float) -> str:
        """Select the most appropriate Arabic prompt based on audio characteristics."""
        if audio_duration < 60:  # Short audio, likely dialogue
            return self.arabic_prompts["dialect"]
        elif audio_duration < 300:  # Medium audio, mixed content
            return self.arabic_prompts["mixed"]
        elif audio_duration < 1800:  # Long audio, likely formal content
            return self.arabic_prompts["formal"]
        else:  # Very long audio, likely news/lecture
            return self.arabic_prompts["news"]
    
    def _advanced_audio_preprocessing(self, audio_path: str) -> str:
        """Advanced audio preprocessing optimized for Arabic speech patterns."""
        try:
            logger.info("🔧 Applying advanced Arabic audio preprocessing...")
            
            # Load audio with optimal sample rate for Arabic phonemes
            audio, sr = librosa.load(audio_path, sr=16000)
            
            if len(audio) == 0:
                return audio_path
            
            # Advanced noise reduction for Arabic speech
            audio = self._arabic_noise_reduction(audio, sr)
            
            # Normalize audio specifically for Arabic dynamic range
            audio = librosa.util.normalize(audio, norm=np.inf, axis=None)
            
            # Apply Arabic-specific filtering
            audio = self._arabic_speech_enhancement(audio, sr)
            
            # Trim silence with Arabic speech pattern awareness
            audio, _ = librosa.effects.trim(
                audio, 
                top_db=25,  # More conservative for Arabic consonants
                frame_length=2048,
                hop_length=512
            )
            
            # Pad audio to ensure complete processing
            if len(audio) < sr:  # Less than 1 second
                audio = np.pad(audio, (0, sr - len(audio)), mode='constant')
            
            # Save preprocessed audio
            output_path = audio_path.replace('.', '_arabic_processed.')
            sf.write(output_path, audio, sr, format='WAV')
            
            logger.info(f"✅ Arabic audio preprocessing complete: {output_path}")
            return output_path
            
        except Exception as e:
            logger.warning(f"Arabic preprocessing failed: {e}")
            return audio_path
    
    def _arabic_noise_reduction(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply noise reduction optimized for Arabic speech frequencies."""
        try:
            # Calculate spectral centroid to identify speech regions
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            
            # Apply spectral gating for Arabic consonants (higher frequencies)
            stft = librosa.stft(audio)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Enhance frequencies important for Arabic phonemes (300-3000 Hz)
            freq_bins = librosa.fft_frequencies(sr=sr, n_fft=2048)
            arabic_freq_mask = (freq_bins >= 300) & (freq_bins <= 3000)
            
            # Apply gentle enhancement to Arabic frequency range
            magnitude[arabic_freq_mask] *= 1.1
            
            # Reconstruct audio
            enhanced_stft = magnitude * np.exp(1j * phase)
            enhanced_audio = librosa.istft(enhanced_stft)
            
            return enhanced_audio
            
        except Exception as e:
            logger.warning(f"Arabic noise reduction failed: {e}")
            return audio
    
    def _arabic_speech_enhancement(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Enhance audio specifically for Arabic speech patterns."""
        try:
            # Apply high-pass filter to remove low-frequency noise
            audio = librosa.effects.preemphasis(audio, coef=0.97)
            
            # Dynamic range compression for Arabic speech
            audio = np.sign(audio) * np.power(np.abs(audio), 0.8)
            
            # Gentle low-pass filter to remove high-frequency artifacts
            audio = librosa.effects.preemphasis(audio, coef=-0.1)
            
            return audio
            
        except Exception as e:
            logger.warning(f"Arabic speech enhancement failed: {e}")
            return audio
    
    def _arabic_text_post_processing(self, text: str) -> str:
        """Advanced post-processing for Arabic text cleanup."""
        if not text or not text.strip():
            return text
        
        # Apply Arabic-specific cleanup patterns
        cleaned_text = text
        for pattern, replacement in self.arabic_cleanup_patterns:
            cleaned_text = re.sub(pattern, replacement, cleaned_text)
        
        # Remove excessive whitespace
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        
        # Ensure text starts and ends properly
        cleaned_text = cleaned_text.strip()
        
        return cleaned_text
    
    def _calculate_arabic_quality_metrics(self, text: str, segments: List[Dict]) -> Dict[str, Any]:
        """Calculate advanced quality metrics for Arabic text."""
        if not text:
            return {
                "arabic_char_ratio": 0.0,
                "quality_score": 0.0,
                "confidence_avg": 0.0,
                "arabic_word_count": 0,
                "coherence_score": 0.0
            }
        
        # Arabic character analysis
        arabic_chars = sum(1 for char in text if '\u0600' <= char <= '\u06FF')
        total_chars = len([c for c in text if c.isalpha()])
        arabic_char_ratio = (arabic_chars / total_chars * 100) if total_chars > 0 else 0
        
        # Arabic word count
        arabic_words = len([word for word in text.split() if any('\u0600' <= c <= '\u06FF' for c in word)])
        
        # Average confidence from segments
        confidences = []
        for segment in segments:
            if 'words' in segment:
                for word in segment['words']:
                    if 'probability' in word:
                        confidences.append(word['probability'])
            elif 'avg_logprob' in segment:
                # Convert log probability to probability
                confidences.append(np.exp(segment['avg_logprob']))
        
        confidence_avg = np.mean(confidences) if confidences else 0.0
        
        # Coherence score based on text structure
        coherence_score = self._calculate_coherence_score(text)
        
        # Overall quality score
        quality_components = [
            arabic_char_ratio / 100 * 0.4,  # 40% weight for Arabic ratio
            confidence_avg * 0.3,  # 30% weight for confidence
            coherence_score * 0.3   # 30% weight for coherence
        ]
        quality_score = sum(quality_components)
        
        return {
            "arabic_char_ratio": round(arabic_char_ratio, 2),
            "quality_score": round(quality_score, 3),
            "confidence_avg": round(confidence_avg, 3),
            "arabic_word_count": arabic_words,
            "coherence_score": round(coherence_score, 3)
        }
    
    def _calculate_coherence_score(self, text: str) -> float:
        """Calculate text coherence score for Arabic text."""
        try:
            # Basic coherence indicators
            sentences = re.split(r'[.!?؟]', text)
            if len(sentences) < 2:
                return 0.5
            
            # Check for consistent language (Arabic vs mixed)
            arabic_sentences = [s for s in sentences if any('\u0600' <= c <= '\u06FF' for c in s)]
            language_consistency = len(arabic_sentences) / len(sentences) if sentences else 0
            
            # Check for reasonable sentence length distribution
            sentence_lengths = [len(s.split()) for s in sentences if s.strip()]
            if not sentence_lengths:
                return 0.3
            
            avg_length = np.mean(sentence_lengths)
            length_variance = np.var(sentence_lengths)
            
            # Reasonable Arabic sentences are typically 3-20 words
            length_score = 1.0 if 3 <= avg_length <= 20 else max(0.2, 1.0 - abs(avg_length - 10) / 20)
            
            # Combine scores
            coherence = (language_consistency * 0.6) + (length_score * 0.4)
            return min(1.0, max(0.0, coherence))
            
        except Exception:
            return 0.5
    
    async def load_model(self, model_size: str = 'large-v2') -> None:
        """Load Whisper model with Arabic optimizations."""
        try:
            if self.model is None or self.current_model_size != model_size:
                logger.info(f"🔄 Loading advanced Arabic-optimized Whisper: {model_size}")
                
                # Clear previous model
                if self.model is not None:
                    del self.model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                
                # Determine optimal compute type
                if self.device == "cuda":
                    compute_type = "float16"
                else:
                    compute_type = "int8"  # CPU optimized
                
                # Load model with Arabic-optimized settings
                self.model = WhisperModel(
                    model_size,
                    device=self.device,
                    compute_type=compute_type,
                    cpu_threads=4,
                    num_workers=1,
                    download_root=None,
                    local_files_only=False
                )
                
                self.current_model_size = model_size
                logger.info(f"✅ Advanced Arabic Whisper model loaded: {model_size}")
                
        except Exception as e:
            logger.error(f"Failed to load Arabic-optimized model: {e}")
            raise
    
    async def transcribe_arabic_advanced(
        self, 
        audio_path: str,
        model_size: str = 'large-v2',
        enable_preprocessing: bool = True
    ) -> Dict[str, Any]:
        """Advanced Arabic transcription with superior accuracy."""
        
        start_time = time.time()
        
        try:
            logger.info(f"🎯 Starting advanced Arabic transcription: {audio_path}")
            
            # Load optimal model
            await self.load_model(model_size)
            
            # Advanced preprocessing
            processed_audio_path = audio_path
            if enable_preprocessing:
                processed_audio_path = self._advanced_audio_preprocessing(audio_path)
            
            # Get audio duration for prompt selection
            try:
                audio_info = librosa.get_duration(path=processed_audio_path)
            except:
                audio_info = 0
            
            # Select optimal Arabic prompt
            arabic_prompt = self._select_optimal_prompt(audio_info)
            
            # Update transcription parameters
            transcribe_params = self.arabic_optimized_params.copy()
            transcribe_params["initial_prompt"] = arabic_prompt
            
            logger.info(f"🔧 Using Arabic prompt: {arabic_prompt[:50]}...")
            logger.info(f"🔧 Transcription parameters: beam_size={transcribe_params['beam_size']}, patience={transcribe_params['patience']}")
            
            # Perform advanced Arabic transcription
            segments, info = self.model.transcribe(
                processed_audio_path,
                language="ar",
                task="transcribe",
                **transcribe_params
            )
            
            # Process results
            segments_list = list(segments)
            full_text = " ".join([segment.text for segment in segments_list])
            
            # Advanced Arabic text post-processing
            cleaned_text = self._arabic_text_post_processing(full_text)
            
            # Calculate advanced quality metrics
            quality_metrics = self._calculate_arabic_quality_metrics(cleaned_text, segments_list)
            
            processing_time = time.time() - start_time
            
            # Build comprehensive result
            result = {
                "text": cleaned_text,
                "language": info.language,
                "language_probability": round(info.language_probability, 3),
                "duration": round(info.duration, 2),
                "segments": [
                    {
                        "start": round(s.start, 2),
                        "end": round(s.end, 2),
                        "text": s.text,
                        "avg_logprob": round(s.avg_logprob, 3),
                        "no_speech_prob": round(s.no_speech_prob, 3)
                    }
                    for s in segments_list
                ],
                "processing_time": round(processing_time, 2),
                "model_size": model_size,
                "preprocessing_applied": enable_preprocessing,
                **quality_metrics
            }
            
            logger.info(f"✅ Advanced Arabic transcription complete in {processing_time:.1f}s")
            logger.info(f"📊 Quality metrics: Arabic ratio={quality_metrics['arabic_char_ratio']:.1f}%, Quality score={quality_metrics['quality_score']:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Advanced Arabic transcription failed: {e}")
            return {
                "text": "",
                "error": str(e),
                "processing_time": time.time() - start_time,
                "quality_score": 0.0
            }
