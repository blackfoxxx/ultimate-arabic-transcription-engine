"""
Voice Analysis Engine for Arabic Transcription System
Provides sentiment analysis, narrative checking, and attention detection from voice patterns.
"""

import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call
import opensmile
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import ShortTermFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import logging
import os
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

class VoiceAnalysisEngine:
    """
    Advanced voice analysis engine for extracting psychological and emotional features from speech.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the voice analysis engine.
        
        Args:
            model_path: Path to pre-trained models directory
        """
        self.logger = logging.getLogger(__name__)
        self.model_path = model_path or "models/voice_analysis"
        
        # Initialize openSMILE for comprehensive feature extraction
        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.ComParE_2016,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        
        # Initialize scalers and models
        self.scaler = StandardScaler()
        self.sentiment_model = None
        self.attention_model = None
        self.narrative_model = None
        
        # Feature extraction parameters
        self.sample_rate = 16000
        self.frame_length = 0.025  # 25ms
        self.frame_shift = 0.010   # 10ms
        
        self._load_models()
        
    def _load_models(self):
        """Load pre-trained models if available."""
        try:
            if os.path.exists(os.path.join(self.model_path, 'sentiment_model.joblib')):
                self.sentiment_model = joblib.load(os.path.join(self.model_path, 'sentiment_model.joblib'))
                self.logger.info("Loaded pre-trained sentiment model")
                
            if os.path.exists(os.path.join(self.model_path, 'attention_model.joblib')):
                self.attention_model = joblib.load(os.path.join(self.model_path, 'attention_model.joblib'))
                self.logger.info("Loaded pre-trained attention model")
                
            if os.path.exists(os.path.join(self.model_path, 'narrative_model.joblib')):
                self.narrative_model = joblib.load(os.path.join(self.model_path, 'narrative_model.joblib'))
                self.logger.info("Loaded pre-trained narrative model")
                
            if os.path.exists(os.path.join(self.model_path, 'scaler.joblib')):
                self.scaler = joblib.load(os.path.join(self.model_path, 'scaler.joblib'))
                self.logger.info("Loaded feature scaler")
                
        except Exception as e:
            self.logger.warning(f"Could not load pre-trained models: {e}")
            self._initialize_default_models()
    
    def _initialize_default_models(self):
        """Initialize default models for basic analysis."""
        self.sentiment_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.attention_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.narrative_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.logger.info("Initialized default models")
    
    def extract_praat_features(self, audio_path: str) -> Dict[str, float]:
        """
        Extract acoustic features using Praat-Parselmouth.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary of acoustic features
        """
        try:
            # Load sound with Praat
            sound = parselmouth.Sound(audio_path)
            
            # Extract pitch
            pitch = call(sound, "To Pitch", 0.0, 75, 600)
            mean_f0 = call(pitch, "Get mean", 0, 0, "Hertz")
            std_f0 = call(pitch, "Get standard deviation", 0, 0, "Hertz")
            
            # Extract formants
            formant = call(sound, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50)
            f1_mean = call(formant, "Get mean", 1, 0, 0, "Hertz")
            f2_mean = call(formant, "Get mean", 2, 0, 0, "Hertz")
            f3_mean = call(formant, "Get mean", 3, 0, 0, "Hertz")
            
            # Extract intensity
            intensity = call(sound, "To Intensity", 75, 0, "yes")
            mean_intensity = call(intensity, "Get mean", 0, 0, "energy")
            
            # Extract voice quality measures
            harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
            mean_hnr = call(harmonicity, "Get mean", 0, 0)
            
            # Calculate jitter and shimmer
            point_process = call(sound, "To PointProcess (periodic, cc)", 75, 600)
            jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            shimmer = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            
            # Speaking rate (syllables per second approximation)
            duration = call(sound, "Get total duration")
            pulses = call(point_process, "Get number of points")
            speaking_rate = pulses / duration if duration > 0 else 0
            
            features = {
                'mean_f0': mean_f0 if not np.isnan(mean_f0) else 0,
                'std_f0': std_f0 if not np.isnan(std_f0) else 0,
                'f1_mean': f1_mean if not np.isnan(f1_mean) else 0,
                'f2_mean': f2_mean if not np.isnan(f2_mean) else 0,
                'f3_mean': f3_mean if not np.isnan(f3_mean) else 0,
                'mean_intensity': mean_intensity if not np.isnan(mean_intensity) else 0,
                'mean_hnr': mean_hnr if not np.isnan(mean_hnr) else 0,
                'jitter': jitter if not np.isnan(jitter) else 0,
                'shimmer': shimmer if not np.isnan(shimmer) else 0,
                'speaking_rate': speaking_rate,
                'duration': duration
            }
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting Praat features: {e}")
            return {}
    
    def extract_opensmile_features(self, audio_path: str) -> np.ndarray:
        """
        Extract comprehensive features using openSMILE.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Feature vector
        """
        try:
            features = self.smile.process_file(audio_path)
            return features.values.flatten()
        except Exception as e:
            self.logger.error(f"Error extracting openSMILE features: {e}")
            return np.array([])
    
    def extract_pyaudio_features(self, audio_path: str) -> np.ndarray:
        """
        Extract features using pyAudioAnalysis.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Feature vector
        """
        try:
            # Load audio
            sampling_rate, signal = audioBasicIO.read_audio_file(audio_path)
            
            # Extract short-term features
            features, _ = ShortTermFeatures.feature_extraction(
                signal, sampling_rate, 
                0.050 * sampling_rate,  # 50ms window
                0.025 * sampling_rate   # 25ms step
            )
            
            # Calculate statistics over time
            feature_stats = []
            for i in range(features.shape[0]):
                feature_stats.extend([
                    np.mean(features[i, :]),
                    np.std(features[i, :]),
                    np.max(features[i, :]),
                    np.min(features[i, :])
                ])
            
            return np.array(feature_stats)
            
        except Exception as e:
            self.logger.error(f"Error extracting pyAudioAnalysis features: {e}")
            return np.array([])
    
    def extract_comprehensive_features(self, audio_path: str) -> Dict[str, Any]:
        """
        Extract comprehensive voice features from multiple sources.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary containing all extracted features
        """
        features = {}
        
        # Extract Praat features
        praat_features = self.extract_praat_features(audio_path)
        features['praat'] = praat_features
        
        # Extract openSMILE features
        opensmile_features = self.extract_opensmile_features(audio_path)
        features['opensmile'] = opensmile_features
        
        # Extract pyAudioAnalysis features
        pyaudio_features = self.extract_pyaudio_features(audio_path)
        features['pyaudio'] = pyaudio_features
        
        # Combine all features into a single vector
        combined_features = []
        
        # Add Praat features
        if praat_features:
            combined_features.extend(list(praat_features.values()))
        
        # Add openSMILE features
        if len(opensmile_features) > 0:
            combined_features.extend(opensmile_features.tolist())
        
        # Add pyAudioAnalysis features
        if len(pyaudio_features) > 0:
            combined_features.extend(pyaudio_features.tolist())
        
        features['combined'] = np.array(combined_features)
        
        return features
    
    def analyze_sentiment(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze sentiment from voice features.
        
        Args:
            features: Extracted voice features
            
        Returns:
            Sentiment analysis results
        """
        try:
            praat_features = features.get('praat', {})
            
            # Rule-based sentiment analysis using acoustic features
            sentiment_score = 0.0
            confidence = 0.0
            
            if praat_features:
                # Higher pitch often indicates positive emotions or stress
                f0_mean = praat_features.get('mean_f0', 0)
                f0_std = praat_features.get('std_f0', 0)
                intensity = praat_features.get('mean_intensity', 0)
                speaking_rate = praat_features.get('speaking_rate', 0)
                
                # Normalize features for analysis
                if f0_mean > 150:  # Higher pitch
                    sentiment_score += 0.2
                if f0_std > 20:    # High pitch variation
                    sentiment_score += 0.1
                if intensity > 60: # Higher intensity
                    sentiment_score += 0.15
                if speaking_rate > 4: # Fast speaking
                    sentiment_score += 0.1
                
                # Voice quality indicators
                jitter = praat_features.get('jitter', 0)
                shimmer = praat_features.get('shimmer', 0)
                hnr = praat_features.get('mean_hnr', 0)
                
                if jitter > 0.01:  # High jitter indicates stress/negative emotion
                    sentiment_score -= 0.1
                if shimmer > 0.1:  # High shimmer indicates stress
                    sentiment_score -= 0.1
                if hnr < 10:       # Low HNR indicates voice strain
                    sentiment_score -= 0.15
                
                confidence = min(1.0, abs(sentiment_score) * 2)
                sentiment_score = max(-1.0, min(1.0, sentiment_score))
            
            # Determine sentiment label
            if sentiment_score > 0.2:
                sentiment_label = "positive"
            elif sentiment_score < -0.2:
                sentiment_label = "negative"
            else:
                sentiment_label = "neutral"
            
            return {
                'sentiment_score': sentiment_score,
                'sentiment_label': sentiment_label,
                'confidence': confidence,
                'features_used': list(praat_features.keys()) if praat_features else []
            }
            
        except Exception as e:
            self.logger.error(f"Error in sentiment analysis: {e}")
            return {
                'sentiment_score': 0.0,
                'sentiment_label': 'neutral',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def analyze_attention_stress(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze attention level and stress indicators from voice.
        
        Args:
            features: Extracted voice features
            
        Returns:
            Attention and stress analysis results
        """
        try:
            praat_features = features.get('praat', {})
            
            attention_score = 0.5  # Baseline
            stress_score = 0.0
            
            if praat_features:
                # Attention indicators
                speaking_rate = praat_features.get('speaking_rate', 0)
                f0_std = praat_features.get('std_f0', 0)
                intensity = praat_features.get('mean_intensity', 0)
                
                # High attention: moderate speaking rate, good pitch control
                if 3 <= speaking_rate <= 6:
                    attention_score += 0.2
                if 15 <= f0_std <= 30:  # Good pitch variation
                    attention_score += 0.15
                if intensity > 55:      # Clear voice
                    attention_score += 0.1
                
                # Stress indicators
                jitter = praat_features.get('jitter', 0)
                shimmer = praat_features.get('shimmer', 0)
                hnr = praat_features.get('mean_hnr', 0)
                f0_mean = praat_features.get('mean_f0', 0)
                
                if jitter > 0.015:      # High jitter = stress
                    stress_score += 0.3
                if shimmer > 0.12:      # High shimmer = stress
                    stress_score += 0.25
                if hnr < 8:             # Low HNR = vocal strain
                    stress_score += 0.2
                if f0_mean > 200 or f0_mean < 80:  # Extreme pitch
                    stress_score += 0.15
                if speaking_rate > 7 or speaking_rate < 2:  # Extreme rate
                    stress_score += 0.1
                
                # Normalize scores
                attention_score = max(0.0, min(1.0, attention_score))
                stress_score = max(0.0, min(1.0, stress_score))
            
            # Determine attention level
            if attention_score > 0.7:
                attention_level = "high"
            elif attention_score > 0.4:
                attention_level = "moderate"
            else:
                attention_level = "low"
            
            # Determine stress level
            if stress_score > 0.6:
                stress_level = "high"
            elif stress_score > 0.3:
                stress_level = "moderate"
            else:
                stress_level = "low"
            
            return {
                'attention_score': attention_score,
                'attention_level': attention_level,
                'stress_score': stress_score,
                'stress_level': stress_level,
                'confidence': min(1.0, (attention_score + stress_score) / 2),
                'indicators': {
                    'speaking_rate': praat_features.get('speaking_rate', 0),
                    'pitch_variation': praat_features.get('std_f0', 0),
                    'voice_quality': {
                        'jitter': praat_features.get('jitter', 0),
                        'shimmer': praat_features.get('shimmer', 0),
                        'hnr': praat_features.get('mean_hnr', 0)
                    }
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in attention/stress analysis: {e}")
            return {
                'attention_score': 0.5,
                'attention_level': 'unknown',
                'stress_score': 0.0,
                'stress_level': 'unknown',
                'error': str(e)
            }
    
    def analyze_narrative_consistency(self, features: Dict[str, Any], segments: List[Dict] = None) -> Dict[str, Any]:
        """
        Analyze narrative consistency and potential deception indicators.
        
        Args:
            features: Extracted voice features
            segments: Optional list of speech segments for temporal analysis
            
        Returns:
            Narrative consistency analysis results
        """
        try:
            praat_features = features.get('praat', {})
            
            consistency_score = 0.5  # Baseline
            deception_risk = 0.0
            
            if praat_features:
                # Consistency indicators
                f0_std = praat_features.get('std_f0', 0)
                speaking_rate = praat_features.get('speaking_rate', 0)
                jitter = praat_features.get('jitter', 0)
                shimmer = praat_features.get('shimmer', 0)
                
                # Consistent narrative: stable prosodic features
                if 10 <= f0_std <= 25:     # Moderate pitch variation
                    consistency_score += 0.2
                if 3 <= speaking_rate <= 5: # Steady speaking rate
                    consistency_score += 0.15
                if jitter < 0.01:          # Low jitter = stable voice
                    consistency_score += 0.1
                if shimmer < 0.08:         # Low shimmer = stable voice
                    consistency_score += 0.1
                
                # Deception risk indicators (based on research literature)
                if f0_std > 35:            # High pitch variation
                    deception_risk += 0.2
                if speaking_rate < 2.5 or speaking_rate > 6.5:  # Extreme rates
                    deception_risk += 0.25
                if jitter > 0.02:          # High jitter
                    deception_risk += 0.15
                if shimmer > 0.15:         # High shimmer
                    deception_risk += 0.15
                
                # Voice breaks and hesitations (approximated)
                hnr = praat_features.get('mean_hnr', 0)
                if hnr < 6:                # Very low HNR = voice breaks
                    deception_risk += 0.2
                
                # Normalize scores
                consistency_score = max(0.0, min(1.0, consistency_score))
                deception_risk = max(0.0, min(1.0, deception_risk))
            
            # Determine consistency level
            if consistency_score > 0.7:
                consistency_level = "high"
            elif consistency_score > 0.4:
                consistency_level = "moderate"
            else:
                consistency_level = "low"
            
            # Determine deception risk level
            if deception_risk > 0.6:
                risk_level = "high"
            elif deception_risk > 0.3:
                risk_level = "moderate"
            else:
                risk_level = "low"
            
            return {
                'consistency_score': consistency_score,
                'consistency_level': consistency_level,
                'deception_risk': deception_risk,
                'risk_level': risk_level,
                'confidence': min(1.0, consistency_score * 0.7),  # Lower confidence for deception detection
                'warning': "Deception detection is experimental and should not be used for critical decisions",
                'prosodic_stability': {
                    'pitch_variation': praat_features.get('std_f0', 0),
                    'speaking_rate': praat_features.get('speaking_rate', 0),
                    'voice_quality_stability': {
                        'jitter': praat_features.get('jitter', 0),
                        'shimmer': praat_features.get('shimmer', 0)
                    }
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in narrative consistency analysis: {e}")
            return {
                'consistency_score': 0.5,
                'consistency_level': 'unknown',
                'deception_risk': 0.0,
                'risk_level': 'unknown',
                'error': str(e)
            }
    
    async def analyze_audio(self, audio_path: str, language: str = 'ar') -> Dict[str, Any]:
        """
        Async wrapper for comprehensive voice analysis.
        
        Args:
            audio_path: Path to audio file
            language: Language code (default: 'ar' for Arabic)
            
        Returns:
            Complete voice analysis results
        """
        return self.analyze_voice_comprehensive(audio_path)
    
    def analyze_voice_comprehensive(self, audio_path: str) -> Dict[str, Any]:
        """
        Perform comprehensive voice analysis including sentiment, attention, and narrative consistency.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Complete voice analysis results
        """
        try:
            self.logger.info(f"Starting comprehensive voice analysis for: {audio_path}")
            
            # Extract all features
            features = self.extract_comprehensive_features(audio_path)
            
            # Perform all analyses
            sentiment_results = self.analyze_sentiment(features)
            attention_results = self.analyze_attention_stress(features)
            narrative_results = self.analyze_narrative_consistency(features)
            
            # Compile comprehensive results
            results = {
                'audio_file': audio_path,
                'analysis_timestamp': np.datetime64('now').astype(str),
                'sentiment_analysis': sentiment_results,
                'attention_stress_analysis': attention_results,
                'narrative_consistency': narrative_results,
                'acoustic_features': features.get('praat', {}),
                'feature_extraction_success': {
                    'praat': bool(features.get('praat')),
                    'opensmile': len(features.get('opensmile', [])) > 0,
                    'pyaudio': len(features.get('pyaudio', [])) > 0
                },
                'overall_confidence': np.mean([
                    sentiment_results.get('confidence', 0),
                    attention_results.get('confidence', 0),
                    narrative_results.get('confidence', 0)
                ]),
                'warnings': [
                    "Voice-based psychological analysis is experimental",
                    "Results should be interpreted by qualified professionals",
                    "Cultural and individual variations may affect accuracy"
                ]
            }
            
            self.logger.info("Comprehensive voice analysis completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive voice analysis: {e}")
            return {
                'audio_file': audio_path,
                'error': str(e),
                'analysis_timestamp': np.datetime64('now').astype(str)
            }
    
    def save_models(self, model_path: str = None):
        """Save trained models to disk."""
        save_path = model_path or self.model_path
        os.makedirs(save_path, exist_ok=True)
        
        try:
            if self.sentiment_model:
                joblib.dump(self.sentiment_model, os.path.join(save_path, 'sentiment_model.joblib'))
            if self.attention_model:
                joblib.dump(self.attention_model, os.path.join(save_path, 'attention_model.joblib'))
            if self.narrative_model:
                joblib.dump(self.narrative_model, os.path.join(save_path, 'narrative_model.joblib'))
            if self.scaler:
                joblib.dump(self.scaler, os.path.join(save_path, 'scaler.joblib'))
                
            self.logger.info(f"Models saved to {save_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving models: {e}")