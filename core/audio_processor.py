"""
Audio processing module for Arabic STT Platform
Handles audio extraction from video and noise reduction
"""

import asyncio
import subprocess
import os
import tempfile
import logging
from pathlib import Path
from typing import Optional, Union, Dict, Any, List
import librosa
import soundfile as sf
import numpy as np

from config import Config
from .speaker_diarization import SpeakerDiarizationEngine

logger = logging.getLogger(__name__)

class AudioProcessor:
    """Handles audio extraction and enhancement."""
    
    def __init__(self):
        self.config = Config()
        self.rnnoise_available = self._check_rnnoise_availability()
        self.diarization_engine = SpeakerDiarizationEngine()
        
    def _check_rnnoise_availability(self) -> bool:
        """Check if RNNoise is available."""
        try:
            # Check if RNNoise binary is available
            result = subprocess.run(['which', 'rnnoise_demo'], 
                                  capture_output=True, text=True)
            return result.returncode == 0
        except Exception:
            logger.warning("RNNoise not found, will use traditional filters")
            return False
    
    async def extract_audio(self, video_path: str, job_id: str) -> str:
        """Extract audio from video file using FFmpeg."""
        try:
            output_path = self.config.TEMP_FOLDER / f"{job_id}_extracted.wav"
            
            # FFmpeg command for audio extraction
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-vn',  # No video
                '-acodec', 'pcm_s16le',  # PCM 16-bit
                '-ar', '16000',  # 16kHz sample rate (optimal for Whisper)
                '-ac', '1',  # Mono
                '-y',  # Overwrite output
                str(output_path)
            ]
            
            logger.info(f"Extracting audio: {' '.join(cmd)}")
            
            # Run FFmpeg
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown FFmpeg error"
                raise Exception(f"FFmpeg failed: {error_msg}")
            
            if not output_path.exists():
                raise Exception("Audio extraction failed - output file not created")
                
            logger.info(f"Audio extracted successfully to {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Audio extraction failed: {str(e)}")
            raise
    
    async def reduce_noise(self, audio_path: str, job_id: str, method: str = 'auto') -> str:
        """Apply noise reduction to audio."""
        try:
            if method == 'none':
                return audio_path
            
            # Determine which method to use
            if method == 'auto':
                method = 'rnnoise' if self.rnnoise_available else 'traditional'
            
            if method == 'rnnoise' and self.rnnoise_available:
                return await self._apply_rnnoise(audio_path, job_id)
            else:
                return await self._apply_traditional_filters(audio_path, job_id)
                
        except Exception as e:
            logger.error(f"Noise reduction failed: {str(e)}")
            # Return original file if noise reduction fails
            return audio_path
    
    async def _apply_rnnoise(self, audio_path: str, job_id: str) -> str:
        """Apply RNNoise for advanced noise reduction."""
        try:
            output_path = self.config.TEMP_FOLDER / f"{job_id}_denoised_rnn.wav"
            
            # Convert to raw PCM for RNNoise
            raw_input = self.config.TEMP_FOLDER / f"{job_id}_input.raw"
            raw_output = self.config.TEMP_FOLDER / f"{job_id}_output.raw"
            
            # Convert to raw PCM 16kHz mono
            cmd_to_raw = [
                'ffmpeg',
                '-i', audio_path,
                '-f', 's16le',
                '-ar', '48000',  # RNNoise expects 48kHz
                '-ac', '1',
                '-y',
                str(raw_input)
            ]
            
            process = await asyncio.create_subprocess_exec(*cmd_to_raw)
            await process.communicate()
            
            # Apply RNNoise
            cmd_rnnoise = [
                'rnnoise_demo',
                str(raw_input),
                str(raw_output)
            ]
            
            process = await asyncio.create_subprocess_exec(*cmd_rnnoise)
            await process.communicate()
            
            # Convert back to WAV
            cmd_to_wav = [
                'ffmpeg',
                '-f', 's16le',
                '-ar', '48000',
                '-ac', '1',
                '-i', str(raw_output),
                '-ar', '16000',  # Convert back to 16kHz for Whisper
                '-y',
                str(output_path)
            ]
            
            process = await asyncio.create_subprocess_exec(*cmd_to_wav)
            await process.communicate()
            
            # Cleanup temporary files
            for temp_file in [raw_input, raw_output]:
                if temp_file.exists():
                    temp_file.unlink()
            
            logger.info(f"RNNoise applied successfully to {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"RNNoise processing failed: {str(e)}")
            raise
    
    async def _apply_traditional_filters(self, audio_path: str, job_id: str) -> str:
        """Apply traditional audio filters for noise reduction with memory optimization."""
        try:
            output_path = self.config.TEMP_FOLDER / f"{job_id}_denoised_trad.wav"
            
            # Check file size and determine processing method
            file_size = Path(audio_path).stat().st_size
            
            if file_size > 50 * 1024 * 1024:  # 50MB threshold
                logger.info(f"Large file detected ({file_size / (1024*1024):.1f}MB), using chunked processing")
                return await self._apply_filters_chunked(audio_path, job_id, output_path)
            else:
                # Load audio with librosa for smaller files
                audio, sr_float = librosa.load(audio_path, sr=16000, mono=True)
                sr = int(sr_float)  # Convert to int
                
                # Apply noise reduction techniques
                audio = await self._spectral_subtraction(audio, sr)
                audio = await self._apply_bandpass_filter(audio, sr)
                audio = await self._normalize_audio(audio)
                
                # Save processed audio
                sf.write(str(output_path), audio, sr)
                
                logger.info(f"Traditional filters applied to {output_path}")
                return str(output_path)
            
        except Exception as e:
            logger.error(f"Traditional filtering failed: {str(e)}")
            raise
    
    async def _spectral_subtraction(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply spectral subtraction for noise reduction."""
        # Estimate noise from first 0.5 seconds
        noise_duration = int(0.5 * sr)
        noise_sample = audio[:noise_duration]
        
        # Compute STFT
        stft = librosa.stft(audio)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Estimate noise spectrum
        noise_stft = librosa.stft(noise_sample)
        noise_magnitude = np.mean(np.abs(noise_stft), axis=1, keepdims=True)
        
        # Spectral subtraction
        alpha = 2.0  # Over-subtraction factor
        enhanced_magnitude = magnitude - alpha * noise_magnitude
        
        # Apply spectral floor
        enhanced_magnitude = np.maximum(enhanced_magnitude, 0.1 * magnitude)
        
        # Reconstruct audio
        enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
        enhanced_audio = librosa.istft(enhanced_stft)
        
        return enhanced_audio
    
    async def _apply_bandpass_filter(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply bandpass filter to remove frequencies outside speech range."""
        try:
            from scipy.signal import butter, sosfiltfilt
            
            # Ensure minimum sample rate for filtering
            if sr < 1000:
                logger.warning(f"Sample rate too low for filtering: {sr}Hz")
                return audio
            
            # Human speech frequency range: 80Hz - 8kHz
            low_freq = 80.0
            high_freq = min(8000.0, sr * 0.4)  # Cap at 40% of Nyquist
            
            # Normalize frequencies
            nyquist = sr / 2
            low = low_freq / nyquist
            high = high_freq / nyquist
            
            # Robust frequency validation
            min_freq = 0.01  # 1% of Nyquist
            max_freq = 0.95  # 95% of Nyquist
            
            low = max(min_freq, min(max_freq, low))
            high = max(low + 0.05, min(max_freq, high))  # Ensure at least 5% bandwidth
            
            # Final validation - if still invalid, skip filtering
            if low >= high or high >= 1.0 or low <= 0 or high - low < 0.01:
                logger.warning(f"Skipping bandpass filter - invalid frequencies: low={low:.3f}, high={high:.3f}, sr={sr}")
                return audio
            
            # Design bandpass filter using SOS format for stability
            sos = butter(4, [low, high], btype='band', output='sos')  # Reduced order for stability
            
            # Apply filter
            filtered_audio = sosfiltfilt(sos, audio)
            
            logger.debug(f"Bandpass filter applied: {low_freq}Hz-{high_freq}Hz (normalized: {low:.3f}-{high:.3f})")
            return filtered_audio
            
        except Exception as e:
            logger.error(f"Bandpass filtering failed: {str(e)}")
            return audio  # Return original audio if filtering fails
    
    async def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio amplitude."""
        # Remove DC offset
        audio = audio - np.mean(audio)
        
        # Normalize to [-0.8, 0.8] to prevent clipping
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio * (0.8 / max_val)
        
        return audio

    async def _apply_filters_chunked(self, audio_path: str, job_id: str, output_path: Path) -> str:
        """Apply filters to large audio files using chunking to manage memory."""
        try:
            # Get audio info without loading the whole file
            info = sf.info(audio_path)
            sr = 16000  # Target sample rate
            duration = info.duration
            chunk_duration = 30.0  # Process 30-second chunks
            
            logger.info(f"Processing {duration:.1f}s audio file in chunks of {chunk_duration}s")
            
            # Process file in chunks and write directly to output file
            with sf.SoundFile(str(output_path), 'w', samplerate=sr, channels=1, format='WAV') as output_file:
                
                for start_time in np.arange(0, duration, chunk_duration):
                    end_time = min(start_time + chunk_duration, duration)
                    
                    logger.debug(f"Processing chunk {start_time:.1f}s - {end_time:.1f}s")
                    
                    # Read chunk using soundfile
                    with sf.SoundFile(audio_path) as f:
                        f.seek(int(start_time * f.samplerate))
                        chunk_samples = int((end_time - start_time) * f.samplerate)
                        audio_chunk = f.read(chunk_samples)
                        
                        # Convert to mono if stereo
                        if len(audio_chunk.shape) > 1:
                            audio_chunk = np.mean(audio_chunk, axis=1)
                    
                    # Resample to target rate if needed
                    if f.samplerate != sr:
                        audio_chunk = librosa.resample(audio_chunk, orig_sr=f.samplerate, target_sr=sr)
                    
                    # Apply filters to chunk
                    audio_chunk = await self._spectral_subtraction(audio_chunk, sr)
                    audio_chunk = await self._apply_bandpass_filter(audio_chunk, sr)
                    audio_chunk = await self._normalize_audio(audio_chunk)
                    
                    # Write chunk directly to output file (streaming)
                    output_file.write(audio_chunk)
                    
                    # Force garbage collection to free memory
                    del audio_chunk
                    if start_time % 300 == 0:  # Every 5 minutes of audio
                        import gc
                        gc.collect()
            
            logger.info(f"Chunked processing completed: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Chunked filtering failed: {str(e)}")
            raise
    
    def get_audio_info(self, audio_path: str) -> dict:
        """Get audio file information without loading entire file."""
        try:
            # Use soundfile for efficient metadata reading
            info = sf.info(audio_path)
            
            return {
                'duration': info.duration,
                'sample_rate': info.samplerate,
                'channels': info.channels,
                'format': Path(audio_path).suffix,
                'file_size': Path(audio_path).stat().st_size
            }
        except Exception as e:
            logger.error(f"Failed to get audio info: {str(e)}")
            # Fallback to librosa for compatibility
            try:
                audio, sr = librosa.load(audio_path, sr=None)
                duration = len(audio) / sr
                
                return {
                    'duration': duration,
                    'sample_rate': sr,
                    'channels': 1 if audio.ndim == 1 else audio.shape[0],
                    'format': Path(audio_path).suffix,
                    'file_size': Path(audio_path).stat().st_size
                }
            except Exception as fallback_e:
                logger.error(f"Fallback audio info failed: {str(fallback_e)}")
                return {}
    
    async def validate_audio_file(self, file_path: str) -> bool:
        """Validate that file is a proper audio/video file."""
        try:
            # Try to get basic info with FFmpeg
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_streams',
                file_path
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                import json
                info = json.loads(stdout.decode())
                # Check if file has audio streams
                audio_streams = [s for s in info.get('streams', []) 
                               if s.get('codec_type') == 'audio']
                return len(audio_streams) > 0
            
            return False
            
        except Exception as e:
            logger.error(f"Audio validation failed: {str(e)}")
            return False

    async def process_multi_speaker_audio(
        self,
        audio_path: str,
        job_id: str,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process audio with multiple speakers.
        
        Args:
            audio_path: Path to input audio file
            job_id: Unique job identifier
            options: Processing options including:
                - enable_diarization: Enable speaker separation
                - max_speakers: Maximum number of speakers to detect
                - enhance_audio: Apply audio enhancement
                - noise_reduction: Type of noise reduction
                
        Returns:
            Dict containing speaker information and separated files
        """
        try:
            options = options or {}
            
            logger.info(f"Starting multi-speaker audio processing for job {job_id}")
            
            # First apply standard noise reduction if requested
            processed_audio_path = audio_path
            if options.get('noise_reduction', 'auto') != 'none':
                logger.info("Applying noise reduction before speaker separation")
                processed_audio_path = await self.reduce_noise(
                    audio_path, job_id, options.get('noise_reduction', 'auto')
                )
            
            # Check if diarization is enabled
            if not options.get('enable_diarization', True):
                logger.info("Speaker diarization disabled, processing as single speaker")
                return {
                    'job_id': job_id,
                    'total_speakers': 1,
                    'speakers': [{
                        'id': 'speaker_0',
                        'label': 'Speaker 1',
                        'segments': [],
                        'characteristics': {
                            'audio_file': {
                                'enhanced_path': processed_audio_path,
                                'original_path': audio_path
                            }
                        }
                    }],
                    'processing_type': 'single_speaker'
                }
            
            # Perform speaker diarization and separation
            diarization_result = await self.diarization_engine.process_multi_speaker_audio(
                processed_audio_path, job_id, options
            )
            
            # Add processing metadata
            diarization_result['processing_type'] = 'multi_speaker'
            diarization_result['original_audio_path'] = audio_path
            diarization_result['noise_reduced_path'] = processed_audio_path
            
            logger.info(f"Multi-speaker processing completed for job {job_id}")
            logger.info(f"Detected {diarization_result['total_speakers']} speakers")
            
            return diarization_result
            
        except Exception as e:
            logger.error(f"Multi-speaker processing failed for job {job_id}: {str(e)}")
            # Fallback to single speaker processing
            return {
                'job_id': job_id,
                'total_speakers': 1,
                'speakers': [{
                    'id': 'speaker_0',
                    'label': 'Speaker 1',
                    'segments': [],
                    'characteristics': {
                        'audio_file': {
                            'enhanced_path': audio_path,
                            'original_path': audio_path
                        }
                    }
                }],
                'processing_type': 'fallback_single_speaker',
                'error': str(e)
            }
    
    async def extract_audio_with_enhancement(
        self,
        video_path: str,
        job_id: str,
        enhancement_options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Extract audio from video with advanced enhancement options.
        
        Args:
            video_path: Path to input video file
            job_id: Unique job identifier
            enhancement_options: Enhancement options
            
        Returns:
            Dict containing extracted audio info and enhanced versions
        """
        try:
            enhancement_options = enhancement_options or {}
            
            logger.info(f"Starting enhanced audio extraction for job {job_id}")
            
            # Extract audio
            audio_path = await self.extract_audio(video_path, job_id)
            
            # Check if multiple audio tracks exist
            audio_tracks = await self._detect_multiple_audio_tracks(video_path)
            
            result = {
                'job_id': job_id,
                'original_video': video_path,
                'extracted_audio': audio_path,
                'audio_tracks': audio_tracks,
                'enhanced_files': {}
            }
            
            # Process each audio track if multiple exist
            if len(audio_tracks) > 1:
                logger.info(f"Multiple audio tracks detected: {len(audio_tracks)}")
                
                for i, track in enumerate(audio_tracks):
                    track_path = await self._extract_specific_audio_track(
                        video_path, job_id, track['index']
                    )
                    
                    # Apply enhancement to each track
                    enhanced_path = await self._apply_advanced_enhancement(
                        track_path, job_id, f"track_{i}", enhancement_options
                    )
                    
                    result['enhanced_files'][f'track_{i}'] = {
                        'original': track_path,
                        'enhanced': enhanced_path,
                        'info': track
                    }
            else:
                # Single audio track processing
                enhanced_path = await self._apply_advanced_enhancement(
                    audio_path, job_id, "main", enhancement_options
                )
                
                result['enhanced_files']['main'] = {
                    'original': audio_path,
                    'enhanced': enhanced_path,
                    'info': audio_tracks[0] if audio_tracks else {}
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Enhanced audio extraction failed for job {job_id}: {str(e)}")
            raise
    
    async def _detect_multiple_audio_tracks(self, video_path: str) -> List[Dict[str, Any]]:
        """Detect multiple audio tracks in video file."""
        try:
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_streams',
                '-select_streams', 'a',  # Audio streams only
                video_path
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                import json
                probe_data = json.loads(stdout.decode())
                
                audio_tracks = []
                for stream in probe_data.get('streams', []):
                    if stream.get('codec_type') == 'audio':
                        audio_tracks.append({
                            'index': stream.get('index'),
                            'codec': stream.get('codec_name'),
                            'channels': stream.get('channels'),
                            'sample_rate': stream.get('sample_rate'),
                            'duration': stream.get('duration'),
                            'language': stream.get('tags', {}).get('language', 'unknown')
                        })
                
                return audio_tracks
            
            return []
            
        except Exception as e:
            logger.warning(f"Audio track detection failed: {str(e)}")
            return []
    
    async def _extract_specific_audio_track(
        self,
        video_path: str,
        job_id: str,
        track_index: int
    ) -> str:
        """Extract specific audio track from video."""
        try:
            output_path = self.config.TEMP_FOLDER / f"{job_id}_track_{track_index}.wav"
            
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-map', f'0:a:{track_index}',  # Select specific audio track
                '-vn',
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                '-y',
                str(output_path)
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0 and output_path.exists():
                logger.info(f"Audio track {track_index} extracted successfully")
                return str(output_path)
            else:
                raise Exception(f"FFmpeg extraction failed: {stderr.decode()}")
                
        except Exception as e:
            logger.error(f"Specific audio track extraction failed: {str(e)}")
            raise
    
    async def _apply_advanced_enhancement(
        self,
        audio_path: str,
        job_id: str,
        track_name: str,
        options: Dict[str, Any]
    ) -> str:
        """Apply advanced audio enhancement techniques."""
        try:
            # Load audio
            audio_data, sr = librosa.load(audio_path, sr=16000)
            
            enhanced_audio = audio_data.copy()
            
            # 1. Advanced noise reduction
            if options.get('advanced_noise_reduction', True):
                enhanced_audio = await self._spectral_gating_noise_reduction(enhanced_audio, sr)
            
            # 2. Voice enhancement
            if options.get('voice_enhancement', True):
                enhanced_audio = await self._voice_enhancement(enhanced_audio, sr)
            
            # 3. Dynamic range optimization
            if options.get('dynamic_range_optimization', True):
                enhanced_audio = await self._optimize_dynamic_range(enhanced_audio, sr)
            
            # 4. Frequency response correction
            if options.get('frequency_correction', True):
                enhanced_audio = await self._frequency_response_correction(enhanced_audio, sr)
            
            # Save enhanced audio
            output_dir = Path(self.config.TEMP_FOLDER) / job_id / 'enhanced'
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_path = output_dir / f'{track_name}_enhanced.wav'
            sf.write(str(output_path), enhanced_audio, sr)
            
            logger.info(f"Advanced enhancement completed for {track_name}")
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Advanced enhancement failed for {track_name}: {str(e)}")
            return audio_path  # Return original if enhancement fails
    
    async def _spectral_gating_noise_reduction(
        self,
        audio_data: np.ndarray,
        sr: int
    ) -> np.ndarray:
        """Apply spectral gating noise reduction."""
        try:
            # Multi-band spectral gating
            stft = librosa.stft(audio_data, hop_length=512, n_fft=2048)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Frequency band analysis
            freq_bins = magnitude.shape[0]
            band_size = freq_bins // 8  # 8 frequency bands
            
            enhanced_magnitude = magnitude.copy()
            
            for band in range(8):
                start_bin = band * band_size
                end_bin = min((band + 1) * band_size, freq_bins)
                
                band_magnitude = magnitude[start_bin:end_bin, :]
                
                # Estimate noise floor for this band
                band_energy = np.mean(band_magnitude ** 2, axis=0)
                noise_floor = np.percentile(band_energy, 30)
                
                # Apply adaptive gating
                for t in range(band_magnitude.shape[1]):
                    current_energy = np.mean(band_magnitude[:, t] ** 2)
                    
                    if current_energy < noise_floor * 2:
                        # Likely noise, apply reduction
                        gate_factor = max(0.1, current_energy / (noise_floor * 2))
                        enhanced_magnitude[start_bin:end_bin, t] *= gate_factor
            
            # Reconstruct audio
            enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
            enhanced_audio = librosa.istft(enhanced_stft, hop_length=512)
            
            return enhanced_audio
            
        except Exception as e:
            logger.warning(f"Spectral gating failed: {str(e)}")
            return audio_data
    
    async def _voice_enhancement(self, audio_data: np.ndarray, sr: int) -> np.ndarray:
        """Enhance voice characteristics in audio."""
        try:
            # Ensure minimum sample rate for filtering
            if sr < 1000:
                logger.warning(f"Sample rate too low for voice enhancement: {sr}Hz")
                return audio_data
                
            # Focus on speech frequency range (80Hz - 8kHz)
            from scipy import signal
            
            # Calculate normalized frequencies with robust validation
            nyquist = sr / 2
            
            # High-pass filter to remove low-frequency noise
            hp_freq_hz = 80.0
            hp_freq = hp_freq_hz / nyquist
            
            # Validate and apply high-pass filter
            if hp_freq > 0.01 and hp_freq < 0.95:
                try:
                    sos_hp = signal.butter(3, hp_freq, 'hp', output='sos')  # Reduced order
                    enhanced_audio = signal.sosfilt(sos_hp, audio_data)
                    logger.debug(f"High-pass filter applied at {hp_freq_hz}Hz")
                except Exception as e:
                    logger.warning(f"High-pass filter failed: {str(e)}")
                    enhanced_audio = audio_data.copy()
            else:
                enhanced_audio = audio_data.copy()
            
            # Low-pass filter to remove high-frequency noise  
            lp_freq_hz = min(8000.0, nyquist * 0.8)  # Cap at 80% of Nyquist
            lp_freq = lp_freq_hz / nyquist
            
            # Validate and apply low-pass filter
            if lp_freq < 0.95 and lp_freq > 0.05:
                try:
                    sos_lp = signal.butter(3, lp_freq, 'lp', output='sos')  # Reduced order
                    enhanced_audio = signal.sosfilt(sos_lp, enhanced_audio)
                    logger.debug(f"Low-pass filter applied at {lp_freq_hz}Hz")
                except Exception as e:
                    logger.warning(f"Low-pass filter failed: {str(e)}")
            
            # Enhance formant frequencies (voice characteristics) - only if sample rate is sufficient
            if sr >= 8000:
                try:
                    stft = librosa.stft(enhanced_audio)
                    magnitude = np.abs(stft)
                    phase = np.angle(stft)
                    
                    # Boost formant regions (300-3400 Hz for speech)
                    freqs = librosa.fft_frequencies(sr=sr, n_fft=stft.shape[0]*2-1)
                    formant_mask = (freqs >= 300) & (freqs <= min(3400, nyquist * 0.8))
                    
                    # Apply gentle boost to formant regions
                    if np.any(formant_mask):
                        formant_boost = np.ones_like(magnitude)
                        formant_boost[formant_mask, :] *= 1.05  # Reduced boost for stability
                        enhanced_magnitude = magnitude * formant_boost
                        
                        # Reconstruct audio
                        enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
                        enhanced_audio = librosa.istft(enhanced_stft)
                        logger.debug("Formant enhancement applied")
                    
                except Exception as e:
                    logger.warning(f"Formant enhancement failed: {str(e)}")
            
            return enhanced_audio
            
        except Exception as e:
            logger.warning(f"Voice enhancement failed: {str(e)}")
            return audio_data
    
    async def _optimize_dynamic_range(self, audio_data: np.ndarray, sr: int) -> np.ndarray:
        """Optimize dynamic range for better transcription."""
        try:
            # Multiband compression
            enhanced_audio = audio_data.copy()
            
            # Analyze RMS levels in windows
            window_size = int(0.1 * sr)  # 100ms windows
            overlap = window_size // 2
            
            for i in range(0, len(audio_data) - window_size, overlap):
                window = audio_data[i:i + window_size]
                rms = np.sqrt(np.mean(window ** 2))
                
                if rms > 0:
                    # Apply gentle compression for loud parts
                    if rms > 0.3:
                        compression_ratio = 0.7
                        target_rms = 0.3 + (rms - 0.3) * compression_ratio
                        gain = target_rms / rms
                    # Apply gentle expansion for quiet parts
                    elif rms < 0.05:
                        expansion_ratio = 1.2
                        target_rms = rms * expansion_ratio
                        gain = min(target_rms / rms, 2.0)  # Limit gain
                    else:
                        gain = 1.0
                    
                    enhanced_audio[i:i + window_size] = window * gain
            
            # Final normalization
            enhanced_audio = librosa.util.normalize(enhanced_audio, norm=np.inf)
            
            return enhanced_audio
            
        except Exception as e:
            logger.warning(f"Dynamic range optimization failed: {str(e)}")
            return audio_data
    
    async def _frequency_response_correction(
        self,
        audio_data: np.ndarray,
        sr: int
    ) -> np.ndarray:
        """Correct frequency response for optimal speech recognition."""
        try:
            # Apply EQ curve optimized for speech recognition
            stft = librosa.stft(audio_data)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            freqs = librosa.fft_frequencies(sr=sr, n_fft=stft.shape[0]*2-1)
            
            # Create EQ curve
            eq_curve = np.ones_like(freqs)
            
            # Gentle boost for consonants (2-4 kHz)
            consonant_mask = (freqs >= 2000) & (freqs <= 4000)
            eq_curve[consonant_mask] *= 1.05
            
            # Slight reduction of low frequencies (< 200 Hz)
            low_freq_mask = freqs < 200
            eq_curve[low_freq_mask] *= 0.95
            
            # Apply EQ curve
            eq_curve = eq_curve.reshape(-1, 1)
            enhanced_magnitude = magnitude * eq_curve
            
            # Reconstruct audio
            enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
            enhanced_audio = librosa.istft(enhanced_stft)
            
            return enhanced_audio
            
        except Exception as e:
            logger.warning(f"Frequency response correction failed: {str(e)}")
            return audio_data
