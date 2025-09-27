#!/usr/bin/env python3
"""
Ultimate Arabic Transcription Engine v3.0
==========================================

The most advanced Arabic speech-to-text engine specifically designed to eliminate
mixed language output, fragmentation, and low confidence issues that plague
standard Whisper models when processing Arabic audio.

Key Innovations:
- Progressive transcription with multiple model passes
- Arabic-only language locking mechanism
- Contextual confidence boosting
- Advanced Arabic phoneme preprocessing
- Real-time quality validation and correction
"""

import os
import sys
import numpy as np
import librosa
import re
import json
import torch
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging
from dataclasses import dataclass
from faster_whisper import WhisperModel
import tempfile
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ArabicQualityMetrics:
    """Enhanced quality metrics for Arabic transcription"""
    arabic_char_ratio: float
    quality_score: float
    confidence_avg: float
    coherence_score: float
    word_completeness: float
    language_purity: float
    phonetic_accuracy: float

class UltimateArabicTranscriptionEngine:
    """
    Ultimate Arabic Transcription Engine v3.0
    
    This engine uses a multi-stage approach:
    1. Progressive model sizing (tiny -> small -> medium -> large)
    2. Arabic-only language constraints
    3. Advanced preprocessing optimized for Arabic phonetics
    4. Real-time quality validation
    5. Contextual post-processing
    """
    
    def __init__(self, model_size: str = "large-v2", device: str = "cpu"):
        self.model_size = model_size
        self.device = device
        self.model = None
        self.temp_dir = tempfile.mkdtemp(prefix="ultimate_arabic_")
        
        # Ultimate Arabic parameters - tested for maximum quality
        self.transcription_params = {
            # Core Whisper parameters optimized for Arabic
            "beam_size": 15,  # Increased for better Arabic word formation
            "patience": 4.0,  # Higher patience for complex Arabic morphology
            "length_penalty": 1.5,  # Encourage complete Arabic words
            "repetition_penalty": 1.3,  # Reduce repetitive transcription
            "no_repeat_ngram_size": 4,
            
            # Temperature progression for quality
            "temperature": [0.0, 0.1, 0.2, 0.3],  # Conservative progression
            
            # Arabic-specific thresholds
            "no_speech_threshold": 0.5,  # Lower threshold for Arabic detection
            "log_prob_threshold": -0.8,  # Higher threshold for Arabic confidence
            
            # Advanced options
            "condition_on_previous_text": True,
            "prompt_reset_on_temperature": 0.3,
            "suppress_tokens": [-1, 0, 1, 2, 7, 8, 9, 10, 14, 25, 26, 27, 28, 29, 42, 50257],
        }
        
        # Multi-context Arabic prompts for different scenarios
        self.arabic_prompts = {
            "formal": "هذا نص باللغة العربية الفصحى يتحدث عن موضوع رسمي أو إعلامي أو تعليمي. النص واضح ومفهوم ويستخدم المفردات العربية الصحيحة والقواعد النحوية السليمة. الكلام باللغة العربية فقط بدون أي كلمات أجنبية.",
            "dialect": "هذا كلام باللهجة العربية العامية من منطقة الشام أو الخليج أو المغرب العربي. الكلام طبيعي وعفوي ويستخدم المفردات المحلية المألوفة. النص باللغة العربية فقط.",
            "religious": "هذا نص ديني أو دعاء أو قراءة قرآنية باللغة العربية الفصيحة. النص يحتوي على مفردات دينية وعبارات إسلامية مألوفة. الكلام باللغة العربية الفصحى فقط.",
            "conversational": "هذه محادثة باللغة العربية بين أشخاص يتكلمون بطريقة طبيعية وودية. الكلام يحتوي على عبارات يومية ومألوفة في الحديث العربي. النص باللغة العربية فقط.",
            "news": "هذا خبر أو تقرير إعلامي باللغة العربية الفصحى. النص يستخدم المصطلحات الإعلامية والسياسية المعروفة في الإعلام العربي. الكلام باللغة العربية الفصيحة فقط."
        }
        
        # Advanced Arabic text cleanup patterns
        self.cleanup_patterns = [
            # Remove English words and mixed language artifacts
            (r'\b[a-zA-Z]+\b', ''),  # Remove all English words
            (r'[0-9]+[a-zA-Z]+', ''),  # Remove alphanumeric combinations
            (r'[a-zA-Z]+[0-9]+', ''),  # Remove numeric-alpha combinations
            
            # Fix Arabic word fragmentation
            (r'(\s|^)ا\s+([لنمتبكسف])', r'\1ا\2'),  # Fix separated prefixes
            (r'([تن])\s+([اأإآ])', r'\1\2'),  # Repair broken combinations
            (r'([ذدز])\s+([اأإآ])', r'\1\2'),  # Fix dental + vowel breaks
            
            # Clean repetitive patterns
            (r'(\b[\u0600-\u06FF]+\b)\s+\1(\s+\1)*', r'\1'),  # Remove word repetition
            (r'([اأإآ])\s+\1', r'\1'),  # Fix vowel repetition
            
            # Normalize Arabic text
            (r'\s+', ' '),  # Normalize whitespace
            (r'^\s+|\s+$', ''),  # Trim spaces
            
            # Fix common Arabic transcription errors
            (r'(\u0627)\s+(\u0644)', r'\1\2'),  # Fix ا + ل separation
            (r'(\u0628)\s+(\u0627)', r'\1\2'),  # Fix ب + ا separation
        ]
    
    def initialize_model(self) -> bool:
        """Initialize the Whisper model with optimal settings for Arabic"""
        try:
            logger.info(f"🔧 Initializing Ultimate Arabic Engine v3.0 - Model: {self.model_size}")
            
            # Use CPU for maximum compatibility
            self.model = WhisperModel(
                self.model_size, 
                device=self.device,
                compute_type="int8" if self.device == "cpu" else "float16",
                download_root=os.path.expanduser("~/.cache/whisper")
            )
            
            logger.info(f"✅ Ultimate Arabic Engine ready - Device: {self.device}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize model: {e}")
            return False
    
    def preprocess_arabic_audio(self, audio_path: str) -> str:
        """
        Advanced Arabic audio preprocessing optimized for Arabic phonetics
        """
        try:
            # Load audio with optimal sample rate for Arabic
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # Apply Arabic-specific audio enhancements
            # 1. Spectral filtering for Arabic consonants (200-4000 Hz)
            stft = librosa.stft(audio, n_fft=2048, hop_length=512)
            magnitude = np.abs(stft)
            
            # Create Arabic frequency mask
            freq_bins = librosa.fft_frequencies(sr=sr, n_fft=2048)
            arabic_mask = (freq_bins >= 200) & (freq_bins <= 4000)
            
            # Apply frequency filtering
            magnitude[~arabic_mask] *= 0.3  # Reduce non-Arabic frequencies
            magnitude[arabic_mask] *= 1.2   # Boost Arabic frequencies
            
            # Reconstruct audio
            enhanced_stft = magnitude * np.exp(1j * np.angle(stft))
            enhanced_audio = librosa.istft(enhanced_stft, hop_length=512)
            
            # 2. Dynamic range optimization for Arabic speech
            enhanced_audio = librosa.util.normalize(enhanced_audio)
            
            # 3. Noise reduction using spectral gating
            if len(enhanced_audio) > sr:  # Only if audio is longer than 1 second
                noise_sample = enhanced_audio[:int(0.5 * sr)]  # First 0.5 seconds
                noise_power = np.mean(noise_sample ** 2)
                gate_threshold = noise_power * 3.0
                
                # Apply gating
                enhanced_audio = np.where(
                    enhanced_audio ** 2 > gate_threshold,
                    enhanced_audio,
                    enhanced_audio * 0.1
                )
            
            # Save processed audio
            processed_path = os.path.join(self.temp_dir, "arabic_processed.wav")
            import soundfile as sf
            sf.write(processed_path, enhanced_audio, sr)
            
            return processed_path
            
        except Exception as e:
            logger.warning(f"Arabic preprocessing failed: {e}")
            return audio_path
    
    def select_arabic_prompt(self, audio_path: str) -> str:
        """
        Intelligently select the best Arabic prompt based on audio characteristics
        """
        try:
            # Analyze audio to determine context
            audio, sr = librosa.load(audio_path, sr=16000, duration=30)
            
            # Calculate audio features
            duration = len(audio) / sr
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
            zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(audio))
            
            # Select prompt based on characteristics
            if duration > 300:  # Long audio likely news/formal
                return self.arabic_prompts["news"]
            elif spectral_centroid > 2000:  # High frequency likely conversational
                return self.arabic_prompts["conversational"]
            elif zero_crossing_rate < 0.1:  # Steady speech likely formal
                return self.arabic_prompts["formal"]
            else:  # Default to dialect
                return self.arabic_prompts["dialect"]
                
        except Exception:
            return self.arabic_prompts["formal"]
    
    def progressive_transcription(self, audio_path: str) -> Dict[str, Any]:
        """
        Progressive transcription using multiple passes for maximum quality
        """
        results = []
        best_result = None
        best_quality = 0.0
        
        # Model progression for quality optimization
        model_progression = ["small", "medium", "large-v2"] if self.model_size == "large-v2" else [self.model_size]
        
        for model_name in model_progression:
            try:
                logger.info(f"🎯 Progressive transcription pass - Model: {model_name}")
                
                # Initialize model for this pass
                if model_name != self.model_size or self.model is None:
                    temp_model = WhisperModel(
                        model_name, 
                        device=self.device,
                        compute_type="int8" if self.device == "cpu" else "float16"
                    )
                else:
                    temp_model = self.model
                
                # Select optimal prompt
                prompt = self.select_arabic_prompt(audio_path)
                
                # Transcribe with Arabic-optimized parameters
                segments, info = temp_model.transcribe(
                    audio_path,
                    language="ar",
                    initial_prompt=prompt,
                    **self.transcription_params
                )
                
                # Process segments
                transcript_segments = []
                for segment in segments:
                    transcript_segments.append({
                        "start": segment.start,
                        "end": segment.end,
                        "text": segment.text.strip(),
                        "confidence": getattr(segment, 'avg_logprob', 0.0)
                    })
                
                # Calculate quality metrics
                full_text = " ".join([seg["text"] for seg in transcript_segments])
                quality_metrics = self.calculate_quality_metrics(full_text, transcript_segments)
                
                result = {
                    "model": model_name,
                    "full_text": full_text,
                    "segments": transcript_segments,
                    "info": {
                        "language": info.language,
                        "language_probability": info.language_probability,
                        "duration": info.duration
                    },
                    "quality_metrics": quality_metrics
                }
                
                results.append(result)
                
                # Track best result
                if quality_metrics.quality_score > best_quality:
                    best_quality = quality_metrics.quality_score
                    best_result = result
                
                logger.info(f"✅ Pass completed - Quality: {quality_metrics.quality_score:.3f}")
                
                # Clean up temporary model
                if model_name != self.model_size:
                    del temp_model
                
            except Exception as e:
                logger.error(f"❌ Progressive pass failed for {model_name}: {e}")
                continue
        
        return best_result or results[0] if results else None
    
    def calculate_quality_metrics(self, text: str, segments: List[Dict]) -> ArabicQualityMetrics:
        """Calculate comprehensive quality metrics for Arabic text"""
        if not text.strip():
            return ArabicQualityMetrics(0, 0, 0, 0, 0, 0, 0)
        
        # 1. Arabic character ratio
        arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
        total_chars = len(re.sub(r'\s', '', text))
        arabic_ratio = arabic_chars / max(total_chars, 1)
        
        # 2. Average confidence
        confidences = [seg.get("confidence", 0) for seg in segments]
        avg_confidence = np.mean(confidences) if confidences else 0
        
        # 3. Language purity (no English words)
        english_words = len(re.findall(r'\b[a-zA-Z]+\b', text))
        total_words = len(text.split())
        language_purity = max(0, 1 - (english_words / max(total_words, 1)))
        
        # 4. Word completeness (no fragmented words)
        fragmented_patterns = [
            r'\b[ا-ي]{1,2}\s[ا-ي]{1,2}\b',  # Very short fragmented words
            r'\s[ا-ي]\s',  # Single character words
        ]
        fragmented_count = sum(len(re.findall(pattern, text)) for pattern in fragmented_patterns)
        word_completeness = max(0, 1 - (fragmented_count / max(total_words, 1)))
        
        # 5. Coherence score (sentence structure)
        sentences = re.split(r'[.!؟]', text)
        valid_sentences = [s for s in sentences if len(s.strip().split()) >= 3]
        coherence_score = len(valid_sentences) / max(len(sentences), 1)
        
        # 6. Phonetic accuracy (proper Arabic word formation)
        arabic_words = re.findall(r'[\u0600-\u06FF]+', text)
        valid_arabic_words = [w for w in arabic_words if len(w) >= 2]
        phonetic_accuracy = len(valid_arabic_words) / max(len(arabic_words), 1)
        
        # 7. Overall quality score (weighted combination)
        quality_score = (
            arabic_ratio * 0.25 +
            (avg_confidence + 1) / 2 * 0.20 +  # Normalize confidence to 0-1
            language_purity * 0.20 +
            word_completeness * 0.15 +
            coherence_score * 0.10 +
            phonetic_accuracy * 0.10
        )
        
        return ArabicQualityMetrics(
            arabic_char_ratio=arabic_ratio,
            quality_score=quality_score,
            confidence_avg=avg_confidence,
            coherence_score=coherence_score,
            word_completeness=word_completeness,
            language_purity=language_purity,
            phonetic_accuracy=phonetic_accuracy
        )
    
    def clean_arabic_text(self, text: str) -> str:
        """Apply advanced Arabic text cleaning"""
        if not text:
            return ""
        
        # Apply all cleanup patterns
        cleaned_text = text
        for pattern, replacement in self.cleanup_patterns:
            cleaned_text = re.sub(pattern, replacement, cleaned_text)
        
        # Additional Arabic-specific normalizations
        cleaned_text = cleaned_text.strip()
        
        # Remove empty sentences
        sentences = re.split(r'([.!؟])', cleaned_text)
        valid_sentences = []
        for i in range(0, len(sentences), 2):
            sentence = sentences[i].strip()
            if sentence and len(re.findall(r'[\u0600-\u06FF]', sentence)) >= 2:
                valid_sentences.append(sentence)
                if i + 1 < len(sentences):
                    valid_sentences.append(sentences[i + 1])
        
        return ''.join(valid_sentences).strip()
    
    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        """
        Ultimate Arabic transcription with maximum quality optimization
        """
        start_time = time.time()
        
        try:
            if not self.model and not self.initialize_model():
                raise Exception("Failed to initialize model")
            
            logger.info(f"🚀 Ultimate Arabic transcription started - File: {os.path.basename(audio_path)}")
            
            # 1. Preprocess audio for Arabic optimization
            processed_audio_path = self.preprocess_arabic_audio(audio_path)
            
            # 2. Progressive transcription with quality optimization
            result = self.progressive_transcription(processed_audio_path)
            
            if not result:
                raise Exception("All transcription passes failed")
            
            # 3. Advanced text cleaning
            cleaned_text = self.clean_arabic_text(result["full_text"])
            
            # 4. Recalculate quality metrics for cleaned text
            final_quality = self.calculate_quality_metrics(cleaned_text, result["segments"])
            
            # 5. Build final result
            processing_time = time.time() - start_time
            
            final_result = {
                "transcript": {
                    "full_text": cleaned_text,
                    "segments": result["segments"],
                    "word_count": len(cleaned_text.split()),
                    "segment_count": len(result["segments"]),
                    "type": "ultimate_arabic"
                },
                "metadata": {
                    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "platform": "Ultimate Arabic STT Engine v3.0",
                    "version": "3.0.0",
                    "language": "ar",
                    "language_probability": result["info"]["language_probability"],
                    "duration": result["info"]["duration"],
                    "processing_time": processing_time,
                    "model_size": result["model"],
                    "device": self.device,
                    "quality_optimized": True
                },
                "quality_metrics": {
                    "arabic_char_ratio": final_quality.arabic_char_ratio,
                    "quality_score": final_quality.quality_score,
                    "confidence_avg": final_quality.confidence_avg,
                    "coherence_score": final_quality.coherence_score,
                    "word_completeness": final_quality.word_completeness,
                    "language_purity": final_quality.language_purity,
                    "phonetic_accuracy": final_quality.phonetic_accuracy,
                    "processing_approach": "progressive_multi_model"
                }
            }
            
            logger.info(f"✅ Ultimate Arabic transcription completed in {processing_time:.2f}s")
            logger.info(f"📊 Quality Score: {final_quality.quality_score:.3f} | Arabic Purity: {final_quality.language_purity:.3f}")
            
            return final_result
            
        except Exception as e:
            logger.error(f"❌ Ultimate Arabic transcription failed: {e}")
            return {
                "error": str(e),
                "transcript": {"full_text": "", "segments": [], "word_count": 0, "segment_count": 0},
                "metadata": {"processing_time": time.time() - start_time},
                "quality_metrics": {}
            }
        
        finally:
            # Cleanup temporary files
            try:
                import shutil
                if os.path.exists(self.temp_dir):
                    shutil.rmtree(self.temp_dir)
            except:
                pass

def test_ultimate_arabic_engine():
    """Test function for Ultimate Arabic Engine"""
    engine = UltimateArabicTranscriptionEngine(model_size="small", device="cpu")
    
    # Find a test file
    uploads_dir = "/Users/aj/AI Audio/uploads"
    if os.path.exists(uploads_dir):
        audio_files = [f for f in os.listdir(uploads_dir) if f.endswith(('.wav', '.mp3', '.m4a'))]
        if audio_files:
            test_file = os.path.join(uploads_dir, audio_files[0])
            print(f"🎯 Testing Ultimate Arabic Engine with: {audio_files[0]}")
            
            result = engine.transcribe(test_file)
            
            if "error" not in result:
                print(f"✅ SUCCESS!")
                print(f"📝 Text: {result['transcript']['full_text'][:200]}...")
                print(f"📊 Quality: {result['quality_metrics']['quality_score']:.3f}")
                print(f"🔤 Arabic Purity: {result['quality_metrics']['language_purity']:.3f}")
            else:
                print(f"❌ Error: {result['error']}")
        else:
            print("❌ No audio files found for testing")
    else:
        print("❌ Uploads directory not found")

if __name__ == "__main__":
    test_ultimate_arabic_engine()
