"""
Advanced Speaker Diarization and Voice Separation for Arabic STT Platform
Handles multi-speaker audio processing and voice isolation
"""

import asyncio
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import librosa
import soundfile as sf
from dataclasses import dataclass
import json
import subprocess
import tempfile
import os

from config import Config
from utils.logger import setup_logger

logger = setup_logger(__name__)

@dataclass
class Speaker:
    """Represents a speaker in the audio."""
    id: str
    label: str
    segments: List[Dict[str, Any]]
    characteristics: Dict[str, Any]

@dataclass 
class SpeakerSegment:
    """Represents a segment of audio belonging to a speaker."""
    start_time: float
    end_time: float
    speaker_id: str
    confidence: float
    audio_path: Optional[str] = None

class SpeakerDiarizationEngine:
    """Advanced speaker diarization and voice separation engine."""
    
    def __init__(self):
        self.config = Config()
        self.sample_rate = 16000
        self.min_segment_duration = 1.0  # Minimum segment duration in seconds
        self.max_speakers = 10  # Maximum number of speakers to detect
        
    async def process_multi_speaker_audio(
        self,
        audio_path: str,
        job_id: str,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process multi-speaker audio with diarization and separation.
        
        Args:
            audio_path: Path to input audio file
            job_id: Unique job identifier
            options: Processing options
            
        Returns:
            Dict containing speaker information and separated audio files
        """
        try:
            options = options or {}
            
            logger.info(f"Starting multi-speaker processing for job {job_id}")
            
            # Step 1: Load and preprocess audio
            audio_data, sr = await self._load_and_preprocess_audio(audio_path)
            
            # Step 2: Perform speaker diarization
            diarization_result = await self._perform_diarization(audio_data, sr, options)
            
            # Step 3: Separate speakers into individual audio files
            separated_audio_files = await self._separate_speakers(
                audio_data, sr, diarization_result, job_id
            )
            
            # Step 4: Enhance individual speaker audio
            enhanced_files = await self._enhance_speaker_audio(
                separated_audio_files, job_id, options
            )
            
            # Step 5: Generate speaker profiles
            speaker_profiles = await self._generate_speaker_profiles(
                enhanced_files, diarization_result
            )
            
            result = {
                'job_id': job_id,
                'total_speakers': len(speaker_profiles),
                'total_duration': len(audio_data) / sr,
                'speakers': speaker_profiles,
                'separated_files': enhanced_files,
                'diarization_segments': diarization_result['segments'],
                'processing_options': options
            }
            
            logger.info(f"Multi-speaker processing completed for job {job_id}")
            logger.info(f"Detected {len(speaker_profiles)} speakers")
            
            return result
            
        except Exception as e:
            logger.error(f"Multi-speaker processing failed for job {job_id}: {str(e)}")
            raise
    
    async def _load_and_preprocess_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Load and preprocess audio for diarization."""
        try:
            # Load audio file
            audio_data, original_sr = librosa.load(audio_path, sr=None)
            
            # Resample to target sample rate if needed
            if original_sr != self.sample_rate:
                audio_data = librosa.resample(
                    audio_data, 
                    orig_sr=original_sr, 
                    target_sr=self.sample_rate
                )
            
            # Normalize audio
            audio_data = librosa.util.normalize(audio_data)
            
            # Apply basic noise reduction
            audio_data = await self._basic_noise_reduction(audio_data)
            
            logger.info(f"Audio loaded and preprocessed: {len(audio_data)/self.sample_rate:.2f}s")
            
            return audio_data, self.sample_rate
            
        except Exception as e:
            logger.error(f"Audio preprocessing failed: {str(e)}")
            raise
    
    async def _basic_noise_reduction(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply basic noise reduction to audio."""
        try:
            # Spectral subtraction for noise reduction
            stft = librosa.stft(audio_data)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Estimate noise from the first 0.5 seconds
            noise_frames = int(0.5 * self.sample_rate / 512)  # 512 is hop length
            noise_spectrum = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
            
            # Spectral subtraction
            alpha = 2.0  # Over-subtraction factor
            beta = 0.001  # Spectral floor
            
            clean_magnitude = magnitude - alpha * noise_spectrum
            clean_magnitude = np.maximum(clean_magnitude, beta * magnitude)
            
            # Reconstruct audio
            clean_stft = clean_magnitude * np.exp(1j * phase)
            clean_audio = librosa.istft(clean_stft)
            
            return clean_audio
            
        except Exception as e:
            logger.warning(f"Basic noise reduction failed: {str(e)}")
            return audio_data  # Return original audio if noise reduction fails
    
    async def _perform_diarization(
        self, 
        audio_data: np.ndarray, 
        sr: int, 
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform speaker diarization using multiple methods.
        
        This implementation uses spectral clustering and voice activity detection.
        For production, consider integrating pyannote.audio or similar libraries.
        """
        try:
            logger.info("Starting speaker diarization")
            
            # Extract features for diarization
            features = await self._extract_speaker_features(audio_data, sr)
            
            # Perform voice activity detection
            vad_segments = await self._voice_activity_detection(audio_data, sr)
            
            # Cluster speakers using spectral clustering
            speaker_clusters = await self._cluster_speakers(features, vad_segments, options)
            
            # Refine speaker boundaries
            refined_segments = await self._refine_speaker_boundaries(
                speaker_clusters, audio_data, sr
            )
            
            result = {
                'segments': refined_segments,
                'num_speakers': len(set(seg['speaker_id'] for seg in refined_segments)),
                'total_speech_duration': sum(
                    seg['end_time'] - seg['start_time'] for seg in refined_segments
                ),
                'features': features
            }
            
            logger.info(f"Diarization completed: {result['num_speakers']} speakers detected")
            
            return result
            
        except Exception as e:
            logger.error(f"Speaker diarization failed: {str(e)}")
            raise
    
    async def _extract_speaker_features(
        self, 
        audio_data: np.ndarray, 
        sr: int
    ) -> np.ndarray:
        """Extract features for speaker identification."""
        try:
            # Use MFCC features for speaker identification
            mfccs = librosa.feature.mfcc(
                y=audio_data,
                sr=sr,
                n_mfcc=13,
                hop_length=512,
                n_fft=2048
            )
            
            # Add spectral centroid and rolloff for additional speaker characteristics
            spectral_centroids = librosa.feature.spectral_centroid(
                y=audio_data, sr=sr, hop_length=512
            )
            
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=audio_data, sr=sr, hop_length=512
            )
            
            # Combine features
            features = np.vstack([mfccs, spectral_centroids, spectral_rolloff])
            
            # Normalize features
            features = (features - np.mean(features, axis=1, keepdims=True)) / (
                np.std(features, axis=1, keepdims=True) + 1e-8
            )
            
            return features.T  # Shape: (time_frames, features)
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {str(e)}")
            raise
    
    async def _voice_activity_detection(
        self, 
        audio_data: np.ndarray, 
        sr: int
    ) -> List[Dict[str, float]]:
        """Detect voice activity segments."""
        try:
            # Calculate energy-based VAD
            frame_length = int(0.025 * sr)  # 25ms frames
            hop_length = int(0.010 * sr)   # 10ms hop
            
            # Calculate short-time energy
            energy = []
            for i in range(0, len(audio_data) - frame_length, hop_length):
                frame = audio_data[i:i + frame_length]
                energy.append(np.sum(frame ** 2))
            
            energy = np.array(energy)
            
            # Adaptive threshold based on energy statistics
            energy_mean = np.mean(energy)
            energy_std = np.std(energy)
            threshold = energy_mean + 0.5 * energy_std
            
            # Find speech segments
            speech_frames = energy > threshold
            
            # Convert frame indices to time segments
            vad_segments = []
            in_speech = False
            start_time = 0
            
            for i, is_speech in enumerate(speech_frames):
                current_time = i * hop_length / sr
                
                if is_speech and not in_speech:
                    # Start of speech segment
                    start_time = current_time
                    in_speech = True
                elif not is_speech and in_speech:
                    # End of speech segment
                    if current_time - start_time >= self.min_segment_duration:
                        vad_segments.append({
                            'start_time': start_time,
                            'end_time': current_time
                        })
                    in_speech = False
            
            # Handle case where audio ends during speech
            if in_speech:
                end_time = len(audio_data) / sr
                if end_time - start_time >= self.min_segment_duration:
                    vad_segments.append({
                        'start_time': start_time,
                        'end_time': end_time
                    })
            
            logger.info(f"VAD detected {len(vad_segments)} speech segments")
            
            return vad_segments
            
        except Exception as e:
            logger.error(f"Voice activity detection failed: {str(e)}")
            return []
    
    async def _cluster_speakers(
        self, 
        features: np.ndarray, 
        vad_segments: List[Dict[str, float]],
        options: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Cluster speech segments by speaker."""
        try:
            from sklearn.cluster import SpectralClustering
            from sklearn.preprocessing import StandardScaler
            
            if len(vad_segments) == 0:
                return []
            
            # Extract features for each VAD segment
            segment_features = []
            valid_segments = []
            
            hop_length = 512
            frame_duration = hop_length / self.sample_rate
            
            for segment in vad_segments:
                start_frame = int(segment['start_time'] / frame_duration)
                end_frame = int(segment['end_time'] / frame_duration)
                
                if end_frame <= start_frame or end_frame >= len(features):
                    continue
                
                # Average features over the segment
                segment_feat = np.mean(features[start_frame:end_frame], axis=0)
                segment_features.append(segment_feat)
                valid_segments.append(segment)
            
            if len(segment_features) < 2:
                # Only one or no segments, assign all to speaker 0
                return [{
                    **segment,
                    'speaker_id': 'speaker_0',
                    'confidence': 1.0
                } for segment in valid_segments]
            
            # Standardize features
            scaler = StandardScaler()
            segment_features = scaler.fit_transform(np.array(segment_features))
            
            # Determine number of speakers
            max_speakers = min(
                options.get('max_speakers', self.max_speakers),
                len(segment_features)
            )
            
            # Use simple k-means for small number of segments, spectral clustering for larger
            if len(segment_features) <= 5:
                from sklearn.cluster import KMeans
                n_speakers = min(2, len(segment_features))
                clustering = KMeans(n_clusters=n_speakers, random_state=42, n_init=10)
            else:
                # Estimate number of speakers using silhouette analysis
                from sklearn.metrics import silhouette_score
                best_score = -1
                best_n = 2
                
                for n in range(2, min(max_speakers + 1, len(segment_features))):
                    clustering = SpectralClustering(
                        n_clusters=n, 
                        random_state=42,
                        affinity='rbf'
                    )
                    labels = clustering.fit_predict(segment_features)
                    score = silhouette_score(segment_features, labels)
                    
                    if score > best_score:
                        best_score = score
                        best_n = n
                
                clustering = SpectralClustering(
                    n_clusters=best_n,
                    random_state=42,
                    affinity='rbf'
                )
            
            # Perform clustering
            speaker_labels = clustering.fit_predict(segment_features)
            
            # Assign speaker IDs and confidence scores
            clustered_segments = []
            for i, (segment, label) in enumerate(zip(valid_segments, speaker_labels)):
                clustered_segments.append({
                    **segment,
                    'speaker_id': f'speaker_{label}',
                    'confidence': 0.8  # TODO: Calculate actual confidence
                })
            
            logger.info(f"Speaker clustering completed: {len(set(speaker_labels))} speakers")
            
            return clustered_segments
            
        except ImportError:
            logger.warning("scikit-learn not available, using simple segmentation")
            # Fallback: alternate between two speakers
            clustered_segments = []
            for i, segment in enumerate(vad_segments):
                clustered_segments.append({
                    **segment,
                    'speaker_id': f'speaker_{i % 2}',
                    'confidence': 0.5
                })
            return clustered_segments
            
        except Exception as e:
            logger.error(f"Speaker clustering failed: {str(e)}")
            # Fallback: assign all to one speaker
            return [{
                **segment,
                'speaker_id': 'speaker_0',
                'confidence': 0.3
            } for segment in vad_segments]
    
    async def _refine_speaker_boundaries(
        self,
        speaker_segments: List[Dict[str, Any]],
        audio_data: np.ndarray,
        sr: int
    ) -> List[Dict[str, Any]]:
        """Refine speaker change boundaries using audio analysis."""
        try:
            # Sort segments by start time
            segments = sorted(speaker_segments, key=lambda x: x['start_time'])
            
            refined_segments = []
            
            for i, segment in enumerate(segments):
                start_time = segment['start_time']
                end_time = segment['end_time']
                speaker_id = segment['speaker_id']
                
                # Check for overlaps with next segment
                if i < len(segments) - 1:
                    next_segment = segments[i + 1]
                    
                    # If there's a gap, extend current segment
                    if next_segment['start_time'] > end_time:
                        gap_duration = next_segment['start_time'] - end_time
                        if gap_duration < 1.0:  # Fill gaps shorter than 1 second
                            end_time = next_segment['start_time']
                    
                    # If there's overlap, find optimal boundary
                    elif next_segment['start_time'] < end_time:
                        optimal_boundary = await self._find_optimal_boundary(
                            audio_data, sr, next_segment['start_time'], end_time
                        )
                        end_time = optimal_boundary
                
                refined_segments.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'speaker_id': speaker_id,
                    'confidence': segment['confidence'],
                    'duration': end_time - start_time
                })
            
            return refined_segments
            
        except Exception as e:
            logger.error(f"Boundary refinement failed: {str(e)}")
            return speaker_segments
    
    async def _find_optimal_boundary(
        self,
        audio_data: np.ndarray,
        sr: int,
        start_time: float,
        end_time: float
    ) -> float:
        """Find optimal speaker change boundary in overlapping region."""
        try:
            # Convert times to sample indices
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            
            # Find minimum energy point as boundary
            segment = audio_data[start_sample:end_sample]
            
            if len(segment) == 0:
                return (start_time + end_time) / 2
            
            # Calculate energy in small windows
            window_size = int(0.1 * sr)  # 100ms windows
            min_energy = float('inf')
            optimal_sample = len(segment) // 2
            
            for i in range(0, len(segment) - window_size, window_size // 2):
                window = segment[i:i + window_size]
                energy = np.sum(window ** 2)
                
                if energy < min_energy:
                    min_energy = energy
                    optimal_sample = i + window_size // 2
            
            optimal_time = start_time + (optimal_sample / sr)
            
            return min(max(optimal_time, start_time), end_time)
            
        except Exception as e:
            logger.error(f"Optimal boundary detection failed: {str(e)}")
            return (start_time + end_time) / 2
    
    async def _separate_speakers(
        self,
        audio_data: np.ndarray,
        sr: int,
        diarization_result: Dict[str, Any],
        job_id: str
    ) -> Dict[str, str]:
        """Separate speakers into individual audio files."""
        try:
            segments = diarization_result['segments']
            separated_files = {}
            
            # Group segments by speaker
            speaker_segments = {}
            for segment in segments:
                speaker_id = segment['speaker_id']
                if speaker_id not in speaker_segments:
                    speaker_segments[speaker_id] = []
                speaker_segments[speaker_id].append(segment)
            
            # Create separated audio for each speaker
            for speaker_id, speaker_segs in speaker_segments.items():
                # Concatenate all segments for this speaker
                speaker_audio_segments = []
                
                for segment in speaker_segs:
                    start_sample = int(segment['start_time'] * sr)
                    end_sample = int(segment['end_time'] * sr)
                    
                    if start_sample < len(audio_data) and end_sample <= len(audio_data):
                        segment_audio = audio_data[start_sample:end_sample]
                        speaker_audio_segments.append(segment_audio)
                
                if speaker_audio_segments:
                    # Concatenate segments with small gaps
                    gap_samples = int(0.1 * sr)  # 100ms gap between segments
                    gap = np.zeros(gap_samples)
                    
                    speaker_audio = speaker_audio_segments[0]
                    for segment_audio in speaker_audio_segments[1:]:
                        speaker_audio = np.concatenate([speaker_audio, gap, segment_audio])
                    
                    # Save separated audio file
                    output_dir = Path(self.config.TEMP_FOLDER) / job_id / 'separated'
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    output_path = output_dir / f'{speaker_id}.wav'
                    sf.write(str(output_path), speaker_audio, sr)
                    
                    separated_files[speaker_id] = str(output_path)
            
            logger.info(f"Speaker separation completed: {len(separated_files)} speakers")
            
            return separated_files
            
        except Exception as e:
            logger.error(f"Speaker separation failed: {str(e)}")
            raise
    
    async def _enhance_speaker_audio(
        self,
        separated_files: Dict[str, str],
        job_id: str,
        options: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Enhance separated speaker audio files."""
        try:
            enhanced_files = {}
            
            for speaker_id, audio_path in separated_files.items():
                # Load speaker audio
                audio_data, sr = librosa.load(audio_path, sr=self.sample_rate)
                
                # Apply enhancement
                enhanced_audio = await self._enhance_single_speaker(
                    audio_data, sr, options
                )
                
                # Save enhanced audio
                enhanced_dir = Path(self.config.TEMP_FOLDER) / job_id / 'enhanced'
                enhanced_dir.mkdir(parents=True, exist_ok=True)
                
                enhanced_path = enhanced_dir / f'{speaker_id}_enhanced.wav'
                sf.write(str(enhanced_path), enhanced_audio, sr)
                
                enhanced_files[speaker_id] = {
                    'original_path': audio_path,
                    'enhanced_path': str(enhanced_path),
                    'duration': len(enhanced_audio) / sr,
                    'sample_rate': sr
                }
            
            logger.info(f"Audio enhancement completed for {len(enhanced_files)} speakers")
            
            return enhanced_files
            
        except Exception as e:
            logger.error(f"Audio enhancement failed: {str(e)}")
            # Return original files if enhancement fails
            return {
                speaker_id: {
                    'original_path': path,
                    'enhanced_path': path,
                    'duration': 0,
                    'sample_rate': self.sample_rate
                }
                for speaker_id, path in separated_files.items()
            }
    
    async def _enhance_single_speaker(
        self,
        audio_data: np.ndarray,
        sr: int,
        options: Dict[str, Any]
    ) -> np.ndarray:
        """Apply enhancement to single speaker audio."""
        try:
            enhanced_audio = audio_data.copy()
            
            # 1. Noise reduction (improved spectral subtraction)
            enhanced_audio = await self._advanced_noise_reduction(enhanced_audio, sr)
            
            # 2. Normalize audio levels
            enhanced_audio = librosa.util.normalize(enhanced_audio)
            
            # 3. Apply gentle high-pass filter to remove low-frequency noise
            from scipy import signal
            
            # Validate sample rate and frequency for filtering
            if sr >= 1000:  # Only apply if sample rate is reasonable
                nyquist = sr / 2
                hp_freq_hz = 80.0
                hp_freq_norm = hp_freq_hz / nyquist
                
                # Apply filter only if normalized frequency is valid
                if hp_freq_norm > 0.01 and hp_freq_norm < 0.95:
                    try:
                        sos = signal.butter(3, hp_freq_norm, 'hp', output='sos')  # Reduced order, normalized freq
                        enhanced_audio = signal.sosfilt(sos, enhanced_audio)
                        logger.debug(f"High-pass filter applied at {hp_freq_hz}Hz")
                    except Exception as e:
                        logger.warning(f"High-pass filter failed: {str(e)}")
                else:
                    logger.debug(f"Skipping high-pass filter - invalid normalized frequency: {hp_freq_norm:.3f}")
            else:
                logger.debug(f"Skipping high-pass filter - sample rate too low: {sr}Hz")
            
            # Ensure array format after filtering
            enhanced_audio = np.asarray(enhanced_audio)
            
            # 4. Dynamic range compression
            enhanced_audio = await self._apply_compression(enhanced_audio)
            
            return enhanced_audio
            
        except Exception as e:
            logger.warning(f"Single speaker enhancement failed: {str(e)}")
            return audio_data
    
    async def _advanced_noise_reduction(
        self,
        audio_data: np.ndarray,
        sr: int
    ) -> np.ndarray:
        """Apply advanced noise reduction techniques."""
        try:
            # Wiener filtering for noise reduction
            stft = librosa.stft(audio_data, hop_length=512, n_fft=2048)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Estimate noise from quieter portions
            energy = np.sum(magnitude ** 2, axis=0)
            noise_threshold = np.percentile(energy, 20)
            noise_frames = energy < noise_threshold
            
            if np.any(noise_frames):
                noise_spectrum = np.mean(magnitude[:, noise_frames], axis=1, keepdims=True)
                
                # Wiener filter
                signal_power = magnitude ** 2
                noise_power = noise_spectrum ** 2
                wiener_gain = signal_power / (signal_power + noise_power + 1e-10)
                
                # Apply gain with minimum threshold
                wiener_gain = np.maximum(wiener_gain, 0.1)
                clean_magnitude = magnitude * wiener_gain
            else:
                clean_magnitude = magnitude
            
            # Reconstruct audio
            clean_stft = clean_magnitude * np.exp(1j * phase)
            clean_audio = librosa.istft(clean_stft, hop_length=512)
            
            return clean_audio
            
        except Exception as e:
            logger.warning(f"Advanced noise reduction failed: {str(e)}")
            return audio_data
    
    async def _apply_compression(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply dynamic range compression to audio."""
        try:
            # Simple RMS-based compression
            window_size = 1024
            compressed_audio = audio_data.copy()
            
            for i in range(0, len(audio_data) - window_size, window_size // 2):
                window = audio_data[i:i + window_size]
                rms = np.sqrt(np.mean(window ** 2))
                
                if rms > 0.1:  # Apply compression to louder parts
                    compression_ratio = 0.8
                    target_rms = 0.1 + (rms - 0.1) * compression_ratio
                    gain = target_rms / (rms + 1e-10)
                    compressed_audio[i:i + window_size] = window * gain
            
            return compressed_audio
            
        except Exception as e:
            logger.warning(f"Audio compression failed: {str(e)}")
            return audio_data
    
    async def _generate_speaker_profiles(
        self,
        enhanced_files: Dict[str, Dict[str, Any]],
        diarization_result: Dict[str, Any]
    ) -> List[Speaker]:
        """Generate detailed speaker profiles."""
        try:
            speakers = []
            segments = diarization_result['segments']
            
            # Group segments by speaker
            speaker_segments_map = {}
            for segment in segments:
                speaker_id = segment['speaker_id']
                if speaker_id not in speaker_segments_map:
                    speaker_segments_map[speaker_id] = []
                speaker_segments_map[speaker_id].append(segment)
            
            for speaker_id, speaker_segments in speaker_segments_map.items():
                # Calculate speaker statistics
                total_duration = sum(
                    seg['end_time'] - seg['start_time'] 
                    for seg in speaker_segments
                )
                
                avg_confidence = np.mean([seg['confidence'] for seg in speaker_segments])
                
                # Analyze audio characteristics if enhanced file exists
                characteristics = {}
                if speaker_id in enhanced_files:
                    try:
                        audio_path = enhanced_files[speaker_id]['enhanced_path']
                        characteristics = await self._analyze_speaker_characteristics(audio_path)
                    except Exception as e:
                        logger.warning(f"Failed to analyze speaker {speaker_id}: {e}")
                
                speaker = Speaker(
                    id=speaker_id,
                    label=f"Speaker {speaker_id.split('_')[1]}",
                    segments=speaker_segments,
                    characteristics={
                        'total_duration': total_duration,
                        'segment_count': len(speaker_segments),
                        'avg_confidence': avg_confidence,
                        'audio_file': enhanced_files.get(speaker_id, {}),
                        **characteristics
                    }
                )
                
                speakers.append(speaker)
            
            # Sort speakers by total speaking time
            speakers.sort(key=lambda s: s.characteristics['total_duration'], reverse=True)
            
            return speakers
            
        except Exception as e:
            logger.error(f"Speaker profile generation failed: {str(e)}")
            return []
    
    async def _analyze_speaker_characteristics(self, audio_path: str) -> Dict[str, Any]:
        """Analyze acoustic characteristics of a speaker."""
        try:
            audio_data, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Calculate fundamental frequency (pitch)
            f0 = librosa.yin(audio_data, fmin=50, fmax=400)
            f0_clean = f0[f0 > 0]  # Remove unvoiced frames
            
            characteristics = {
                'avg_pitch_hz': float(np.mean(f0_clean)) if len(f0_clean) > 0 else 0.0,
                'pitch_std_hz': float(np.std(f0_clean)) if len(f0_clean) > 0 else 0.0,
                'pitch_range_hz': float(np.max(f0_clean) - np.min(f0_clean)) if len(f0_clean) > 0 else 0.0,
            }
            
            # Calculate spectral characteristics
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sr)[0]
            characteristics['avg_spectral_centroid'] = float(np.mean(spectral_centroids))
            
            # Calculate speaking rate (approximate)
            onset_frames = librosa.onset.onset_detect(y=audio_data, sr=sr)
            speaking_rate = len(onset_frames) / (len(audio_data) / sr) * 60  # onsets per minute
            characteristics['speaking_rate_opm'] = float(speaking_rate)
            
            # Energy characteristics
            rms_energy = librosa.feature.rms(y=audio_data)[0]
            characteristics['avg_energy'] = float(np.mean(rms_energy))
            characteristics['energy_std'] = float(np.std(rms_energy))
            
            return characteristics
            
        except Exception as e:
            logger.warning(f"Speaker characteristics analysis failed: {str(e)}")
            return {}
