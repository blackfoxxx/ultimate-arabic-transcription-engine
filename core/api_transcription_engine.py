"""
API-based transcription engine for Arabic STT Platform
Supports OpenAI Whisper API and other cloud services
"""

import asyncio
import logging
import aiohttp
import aiofiles
from typing import Dict, List, Any, Optional
from pathlib import Path
import time
import json

from config import Config

logger = logging.getLogger(__name__)

class APITranscriptionEngine:
    """Handles speech-to-text transcription using cloud APIs."""
    
    def __init__(self):
        self.config = Config()
        self.session = None
        # OpenAI file size limit (25MB)
        self.OPENAI_MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB in bytes
        
    async def _get_session(self):
        """Get or create aiohttp session."""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=300)  # 5 minutes timeout
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session
    
    def _check_file_size(self, file_path: str) -> Dict[str, Any]:
        """Check if file size is within API limits."""
        try:
            file_size = Path(file_path).stat().st_size
            size_mb = file_size / (1024 * 1024)
            
            return {
                'size_bytes': file_size,
                'size_mb': round(size_mb, 2),
                'exceeds_limit': file_size > self.OPENAI_MAX_FILE_SIZE,
                'limit_mb': 25
            }
        except Exception as e:
            logger.error(f"Failed to check file size for {file_path}: {e}")
            return {
                'size_bytes': 0,
                'size_mb': 0,
                'exceeds_limit': True,
                'error': str(e)
            }
    
    async def transcribe_openai(
        self,
        audio_path: str,
        model: str = 'whisper-1',
        language: str = 'ar',
        **kwargs
    ) -> Dict[str, Any]:
        """
        Transcribe audio using OpenAI Whisper API with retry logic.
        
        Args:
            audio_path: Path to audio file
            model: OpenAI model name
            language: Language code
            **kwargs: Additional parameters
            
        Returns:
            Dict containing transcript and metadata
        """
        try:
            if not self.config.OPENAI_API_KEY:
                raise ValueError("OpenAI API key not configured")
            
            # Check file size
            file_size_info = self._check_file_size(audio_path)
            if file_size_info['exceeds_limit']:
                raise ValueError(f"File size exceeds OpenAI API limit of {file_size_info['limit_mb']}MB. "
                                 f"File size: {file_size_info['size_mb']}MB")
            
            logger.info(f"Starting OpenAI transcription of {audio_path}")
            start_time = time.time()
            
            # Retry configuration
            max_retries = 3
            retry_delay = 1  # seconds
            
            for attempt in range(max_retries):
                try:
                    session = await self._get_session()
                    
                    # Prepare request data
                    headers = {
                        'Authorization': f'Bearer {self.config.OPENAI_API_KEY}'
                    }
                    
                    # Read audio file
                    async with aiofiles.open(audio_path, 'rb') as audio_file:
                        audio_data = await audio_file.read()
                    
                    # Prepare form data
                    data = aiohttp.FormData()
                    data.add_field('file', audio_data, 
                                  filename=Path(audio_path).name,
                                  content_type='audio/mpeg')
                    data.add_field('model', model)
                    
                    if language != 'auto':
                        data.add_field('language', language)
                    
                    # Add response format for detailed output
                    response_format = kwargs.get('response_format', 'verbose_json')
                    data.add_field('response_format', response_format)
                    
                    # Add optional parameters
                    if 'temperature' in kwargs:
                        data.add_field('temperature', str(kwargs['temperature']))
                    
                    if 'prompt' in kwargs:
                        data.add_field('prompt', kwargs['prompt'])
                    elif language == 'ar':
                        # Use Arabic-specific prompt
                        arabic_prompt = self._get_arabic_prompt()
                        data.add_field('prompt', arabic_prompt)
                    
                    # Make API request
                    url = f"{self.config.OPENAI_BASE_URL}/audio/transcriptions"
                    
                    if attempt > 0:
                        logger.info(f"Retry attempt {attempt + 1}/{max_retries} for OpenAI API")
                    
                    logger.info(f"Making request to OpenAI API: {url}")
                    
                    async with session.post(url, headers=headers, data=data) as response:
                        response_text = await response.text()
                        
                        if response.status != 200:
                            logger.error(f"OpenAI API error {response.status}: {response_text}")
                            
                            # Determine if we should retry
                            should_retry = (
                                response.status in [500, 502, 503, 504] and  # Server errors
                                attempt < max_retries - 1  # Not the last attempt
                            )
                            
                            if should_retry:
                                logger.warning(f"Retrying in {retry_delay} seconds due to server error...")
                                await asyncio.sleep(retry_delay)
                                retry_delay *= 2  # Exponential backoff
                                continue
                            
                            # Provide more specific error messages
                            if response.status == 500:
                                raise Exception(f"OpenAI API server error (500): The OpenAI service is experiencing internal issues. This is typically a temporary problem on OpenAI's end. Response: {response_text}")
                            elif response.status == 503:
                                raise Exception(f"OpenAI API service unavailable (503): The service is temporarily overloaded or down for maintenance. Response: {response_text}")
                            elif response.status == 429:
                                raise Exception(f"OpenAI API rate limit exceeded (429): Too many requests. Please wait before trying again. Response: {response_text}")
                            elif response.status == 401:
                                raise Exception(f"OpenAI API authentication failed (401): Invalid API key. Response: {response_text}")
                            elif response.status == 400:
                                raise Exception(f"OpenAI API bad request (400): Invalid request parameters. Response: {response_text}")
                            else:
                                raise Exception(f"OpenAI API error {response.status}: {response_text}")
                        
                        try:
                            result = json.loads(response_text)
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse OpenAI API response: {response_text}")
                            raise Exception(f"Invalid JSON response from OpenAI API: {str(e)}")
                    
                    # Success - break out of retry loop
                    break
                    
                except Exception as e:
                    if attempt == max_retries - 1:  # Last attempt
                        raise
                    
                    # Check if it's worth retrying
                    error_str = str(e).lower()
                    if any(x in error_str for x in ['server error', '500', '502', '503', '504', 'timeout']):
                        logger.warning(f"Attempt {attempt + 1} failed with: {str(e)}. Retrying in {retry_delay} seconds...")
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    else:
                        # Don't retry for authentication, bad request, etc.
                        raise
            
            processing_time = time.time() - start_time
            
            # Convert OpenAI response to our standard format
            standardized_result = self._standardize_openai_response(
                result, processing_time, model, audio_path
            )
            
            logger.info(f"OpenAI transcription completed in {processing_time:.2f}s")
            
            return standardized_result
            
        except Exception as e:
            logger.error(f"OpenAI transcription failed: {str(e)}")
            raise
    
    def _standardize_openai_response(
        self, 
        openai_result: Dict[str, Any], 
        processing_time: float,
        model: str,
        audio_path: str
    ) -> Dict[str, Any]:
        """Convert OpenAI API response to standard format."""
        
        # Extract segments if available (verbose_json format)
        segments = []
        full_text = openai_result.get('text', '')
        
        if 'segments' in openai_result:
            for i, segment in enumerate(openai_result['segments']):
                segment_data = {
                    'id': i,
                    'start': segment.get('start', 0),
                    'end': segment.get('end', 0),
                    'text': segment.get('text', '').strip(),
                    'avg_logprob': segment.get('avg_logprob', 0),
                    'compression_ratio': segment.get('compression_ratio', 0),
                    'no_speech_prob': segment.get('no_speech_prob', 0),
                    'words': []
                }
                
                # Add word-level timestamps if available
                if 'words' in segment:
                    for word in segment['words']:
                        word_data = {
                            'start': word.get('start', 0),
                            'end': word.get('end', 0),
                            'word': word.get('word', ''),
                            'probability': 1.0 - word.get('no_speech_prob', 0)
                        }
                        segment_data['words'].append(word_data)
                
                segments.append(segment_data)
        
        # Calculate metadata
        duration = openai_result.get('duration', 0)
        if not duration and segments:
            duration = max(s['end'] for s in segments) if segments else 0
        
        # Prepare standardized result
        result = {
            'text': full_text,
            'segments': segments,
            'language': openai_result.get('language', 'ar'),
            'language_probability': 0.95,  # OpenAI doesn't provide this, assume high confidence
            'duration': duration,
            'processing_time': processing_time,
            'model_size': model,
            'device': 'cloud_api',
            'api_provider': 'openai',
            'transcript_metadata': {
                'avg_logprob': sum(s['avg_logprob'] for s in segments) / len(segments) if segments else 0,
                'compression_ratio': sum(s['compression_ratio'] for s in segments) / len(segments) if segments else 0,
                'no_speech_prob': sum(s['no_speech_prob'] for s in segments) / len(segments) if segments else 0,
                'total_segments': len(segments),
                'total_words': sum(len(s['words']) for s in segments)
            }
        }
        
        return result
    
    def _get_arabic_prompt(self) -> str:
        """Get initial prompt optimized for Arabic transcription."""
        return (
            "هذا تسجيل صوتي باللغة العربية. "
            "الرجاء كتابة النص بدقة مع علامات الترقيم المناسبة."
        )
    
    def estimate_cost(self, duration_seconds: float, provider: str = 'openai', model: str = 'whisper-1') -> Dict[str, Any]:
        """Estimate transcription cost."""
        try:
            if provider == 'openai' and model == 'whisper-1':
                minutes = duration_seconds / 60
                cost = minutes * 0.006  # $0.006 per minute
                
                return {
                    'duration_minutes': round(minutes, 2),
                    'estimated_cost_usd': round(cost, 4),
                    'provider': provider,
                    'model': model,
                    'rate_per_minute': 0.006
                }
        except Exception as e:
            logger.error(f"Cost estimation failed: {str(e)}")
        
        return {
            'duration_minutes': 0,
            'estimated_cost_usd': 0,
            'provider': provider,
            'model': model,
            'error': 'Cost estimation unavailable'
        }
    
    async def validate_api_credentials(self, provider: str = 'openai') -> bool:
        """Validate API credentials."""
        try:
            if provider == 'openai':
                if not self.config.OPENAI_API_KEY:
                    return False
                
                session = await self._get_session()
                headers = {
                    'Authorization': f'Bearer {self.config.OPENAI_API_KEY}'
                }
                
                # Test API access with models endpoint
                url = f"{self.config.OPENAI_BASE_URL}/models"
                async with session.get(url, headers=headers) as response:
                    return response.status == 200
            
            return False
            
        except Exception as e:
            logger.error(f"API validation failed for {provider}: {str(e)}")
            return False
    
    async def get_supported_models(self, provider: str = 'openai') -> Dict[str, str]:
        """Get list of supported models for a provider."""
        if provider == 'openai':
            return {
                'whisper-1': 'OpenAI Whisper v1 (Latest)'
            }
        else:
            return {}
    
    async def get_pricing_info(self, provider: str = 'openai') -> Dict[str, Any]:
        """Get pricing information for API usage."""
        if provider == 'openai':
            return {
                'whisper-1': {
                    'price_per_minute': 0.006,  # $0.006 per minute
                    'currency': 'USD',
                    'billing_unit': 'per minute'
                }
            }
        else:
            return {}
    
    async def cleanup(self):
        """Cleanup resources."""
        try:
            if self.session and not self.session.closed:
                await self.session.close()
                logger.info("API session closed")
        except Exception as e:
            logger.error(f"API cleanup error: {str(e)}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()
