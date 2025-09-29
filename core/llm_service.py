"""
LLM Service for Arabic STT Platform
Provides local LLM integration with multiple backend support
"""

import asyncio
import logging
import json
import aiohttp
import time
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from config import Config

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class LLMBackend(Enum):
    """Supported LLM backends."""
    OLLAMA = "ollama"
    TRANSFORMERS = "transformers"
    OPENAI_COMPATIBLE = "openai_compatible"

@dataclass
class LLMResponse:
    """Standard LLM response format."""
    text: str
    tokens_used: int
    processing_time: float
    model: str
    metadata: Dict[str, Any]

class LLMService:
    """Unified LLM service supporting multiple backends."""
    
    def __init__(self):
        self.config = Config()
        self.backend = LLMBackend.OLLAMA  # Default backend
        self.model = "llama3.2:3b"
        self.server_url = "http://localhost:11434"
        self.session = None
        self.arabic_model = "aya:8b"  # Specialized Arabic model
        self._session_lock = asyncio.Lock() if hasattr(asyncio, 'current_task') else None
        
        # Load configuration
        self._load_config()
        
    def _load_config(self):
        """Load LLM configuration from config."""
        # LLM settings from config or environment
        import os
        self.backend = LLMBackend(os.getenv('LLM_BACKEND', 'ollama'))
        self.model = os.getenv('LLM_MODEL', 'llama3.2:3b')
        self.server_url = os.getenv('LLM_SERVER_URL', 'http://localhost:11434')
        self.arabic_model = os.getenv('ARABIC_LLM_MODEL', 'aya:8b')
        self.max_tokens = int(os.getenv('LLM_MAX_TOKENS', '4096'))
        self.temperature = float(os.getenv('LLM_TEMPERATURE', '0.7'))
        
    async def initialize(self):
        """Initialize the LLM service."""
        try:
            if self.session is None:
                self.session = aiohttp.ClientSession()
            
            # Test connectivity and model availability
            health_status = await self.health_check()
            if health_status:
                logger.info(f"LLM service initialized with {self.backend.value} backend")
                return True
            else:
                logger.warning(f"LLM service initialized but health check failed - models may not be available")
                return True  # Still return True as service is initialized, just models might not be ready
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM service: {str(e)}")
            return False
    
    async def cleanup(self):
        """Cleanup LLM service resources."""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def _ensure_session(self):
        """Ensure we have a valid aiohttp session for the current event loop."""
        # Always create a new session for each request to avoid event loop issues
        if self.session and not self.session.closed:
            try:
                await self.session.close()
            except:
                pass
        
        try:
            # Create a new session for the current event loop
            self.session = aiohttp.ClientSession()
        except Exception as e:
            logger.error(f"Failed to create aiohttp session: {str(e)}")
            raise
    
    async def health_check(self) -> bool:
        """Check if the LLM service is healthy."""
        try:
            logger.debug(f"Checking Ollama health at {self.server_url}")
            await self._ensure_session()
            
            async with self.session.get(f"{self.server_url}/api/version", timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status == 200:
                    logger.debug(f"Ollama version check status: {response.status}")
                    result = True
                else:
                    logger.error(f"Ollama health check failed with status: {response.status}")
                    result = False
            
            # Close the session after use
            if self.session and not self.session.closed:
                await self.session.close()
                self.session = None
            
            logger.debug(f"Health check result: {result}")
            return result
        except Exception as e:
            logger.error(f"Ollama health check failed: {str(e)}")
            # Ensure session is closed on error
            if self.session and not self.session.closed:
                try:
                    await self.session.close()
                except:
                    pass
                self.session = None
            return False
    
    async def _ollama_health_check(self) -> bool:
        """Check Ollama server health."""
        try:
            logger.debug(f"Checking Ollama health at {self.server_url}")
            async with self.session.get(f"{self.server_url}/api/version") as response:
                logger.debug(f"Ollama version check status: {response.status}")
                if response.status == 200:
                    # Check if primary model is available
                    logger.debug(f"Checking primary model: {self.model}")
                    primary_available = await self._check_model_availability(self.model)
                    logger.debug(f"Primary model available: {primary_available}")
                    
                    # Check if Arabic model is available
                    logger.debug(f"Checking Arabic model: {self.arabic_model}")
                    arabic_available = await self._check_model_availability(self.arabic_model)
                    logger.debug(f"Arabic model available: {arabic_available}")
                    
                    # Return true if at least one model is available
                    result = primary_available or arabic_available
                    logger.debug(f"Health check result: {result}")
                    return result
                return False
        except Exception as e:
            logger.error(f"Ollama health check failed: {str(e)}")
            return False
    
    async def _transformers_health_check(self) -> bool:
        """Check Transformers backend health."""
        try:
            # Import transformers to check availability
            import transformers
            return True
        except ImportError:
            return False
    
    async def _check_model_availability(self, model: str) -> bool:
        """Check if specific model is available."""
        try:
            if self.backend == LLMBackend.OLLAMA:
                async with self.session.get(f"{self.server_url}/api/tags") as response:
                    if response.status == 200:
                        data = await response.json()
                        models = [m['name'] for m in data.get('models', [])]
                        logger.debug(f"Available models: {models}")
                        logger.debug(f"Checking for model: {model}")
                        return model in models
            return False
        except Exception as e:
            logger.error(f"Failed to check model availability for {model}: {str(e)}")
            return False
    
    async def generate_text(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None
    ) -> LLMResponse:
        """Generate text using the LLM."""
        try:
            start_time = time.time()
            
            # Use provided parameters or defaults
            model = model or self.model
            max_tokens = max_tokens or self.max_tokens
            temperature = temperature or self.temperature
            
            if self.backend == LLMBackend.OLLAMA:
                response = await self._ollama_generate(
                    prompt, model, max_tokens, temperature, system_prompt
                )
            elif self.backend == LLMBackend.TRANSFORMERS:
                response = await self._transformers_generate(
                    prompt, model, max_tokens, temperature, system_prompt
                )
            else:
                raise ValueError(f"Unsupported backend: {self.backend}")
            
            processing_time = time.time() - start_time
            
            return LLMResponse(
                text=response['text'],
                tokens_used=response.get('tokens_used', 0),
                processing_time=processing_time,
                model=model,
                metadata=response.get('metadata', {})
            )
            
        except Exception as e:
            logger.error(f"Text generation failed: {str(e)}")
            raise
    
    async def _ollama_generate(
        self,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float,
        system_prompt: Optional[str]
    ) -> Dict[str, Any]:
        """Generate text using Ollama backend."""
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature
                }
            }
            
            if system_prompt:
                payload["system"] = system_prompt
            
            async with self.session.post(
                f"{self.server_url}/api/generate",
                json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        'text': data.get('response', ''),
                        'tokens_used': data.get('eval_count', 0),
                        'metadata': {
                            'model': data.get('model', model),
                            'total_duration': data.get('total_duration', 0),
                            'load_duration': data.get('load_duration', 0),
                            'prompt_eval_count': data.get('prompt_eval_count', 0),
                            'eval_count': data.get('eval_count', 0)
                        }
                    }
                else:
                    error_text = await response.text()
                    raise Exception(f"Ollama API error: {response.status} - {error_text}")
                    
        except Exception as e:
            logger.error(f"Ollama generation failed: {str(e)}")
            raise
    
    async def _transformers_generate(
        self,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float,
        system_prompt: Optional[str]
    ) -> Dict[str, Any]:
        """Generate text using Transformers backend."""
        # This would be implemented for local transformers models
        # For now, raise not implemented
        raise NotImplementedError("Transformers backend not yet implemented")
    
    async def enhance_arabic_text(
        self,
        text: str,
        enhancement_type: str = "grammar"
    ) -> LLMResponse:
        """Enhance Arabic text with specialized processing."""
        try:
            # Use Arabic-specialized model if available
            model = self.arabic_model if await self._check_model_availability(self.arabic_model) else self.model
            
            system_prompt = self._get_arabic_enhancement_prompt(enhancement_type)
            
            # Create enhancement prompt
            if enhancement_type == "grammar":
                prompt = f"Correct the grammar and improve the clarity of this Arabic text while preserving its meaning:\n\n{text}"
            elif enhancement_type == "diacritization":
                prompt = f"Add appropriate diacritical marks (تشكيل) to this Arabic text:\n\n{text}"
            elif enhancement_type == "normalization":
                prompt = f"Normalize this Arabic text to Modern Standard Arabic while preserving the original meaning:\n\n{text}"
            else:
                prompt = f"Enhance this Arabic text:\n\n{text}"
            
            return await self.generate_text(
                prompt=prompt,
                model=model,
                system_prompt=system_prompt,
                temperature=0.3  # Lower temperature for text enhancement
            )
            
        except Exception as e:
            logger.error(f"Arabic text enhancement failed: {str(e)}")
            raise
    
    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of Arabic text."""
        try:
            system_prompt = """You are an expert in Arabic language sentiment analysis. 
            Analyze the sentiment of the given text and respond with a JSON object containing:
            - score: float between -1 (very negative) and 1 (very positive)
            - label: "positive", "negative", or "neutral"
            - confidence: float between 0 and 1
            - emotions: list of detected emotions
            
            Respond only with valid JSON, no additional text."""
            
            prompt = f"Analyze the sentiment of this Arabic text:\n\n{text}"
            
            response = await self.generate_text(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.1
            )
            
            # Parse JSON response
            try:
                sentiment_data = json.loads(response.text)
                return sentiment_data
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                return {
                    "score": 0.0,
                    "label": "neutral",
                    "confidence": 0.5,
                    "emotions": [],
                    "error": "Failed to parse sentiment analysis response"
                }
                
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {str(e)}")
            raise
    
    async def generate_summary(
        self,
        text: str,
        summary_type: str = "medium",
        language: str = "ar"
    ) -> Dict[str, str]:
        """Generate text summary in different lengths."""
        try:
            summaries = {}
            
            # Define summary prompts
            prompts = {
                "short": "Create a brief 2-3 sentence summary of this text:",
                "medium": "Create a comprehensive paragraph summary of this text:",
                "detailed": "Create a detailed multi-paragraph summary of this text:",
                "bullet_points": "Extract the key points from this text as bullet points:"
            }
            
            # Generate requested summary type or all types
            types_to_generate = [summary_type] if summary_type in prompts else prompts.keys()
            
            for s_type in types_to_generate:
                system_prompt = f"You are an expert summarizer. Provide clear, accurate summaries in {language}."
                prompt = f"{prompts[s_type]}\n\n{text}"
                
                response = await self.generate_text(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=0.3
                )
                
                summaries[s_type] = response.text
            
            return summaries
            
        except Exception as e:
            logger.error(f"Summary generation failed: {str(e)}")
            raise
    
    async def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities from Arabic text."""
        try:
            system_prompt = """You are an expert in Arabic named entity recognition.
            Extract entities from the text and respond with a JSON object containing:
            - persons: list of person names
            - locations: list of locations and places
            - organizations: list of organizations and institutions
            - dates: list of dates and time references
            - other: list of other important entities
            
            Respond only with valid JSON, no additional text."""
            
            prompt = f"Extract named entities from this Arabic text:\n\n{text}"
            
            response = await self.generate_text(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.1
            )
            
            # Parse JSON response
            try:
                entities = json.loads(response.text)
                return entities
            except json.JSONDecodeError:
                return {
                    "persons": [],
                    "locations": [],
                    "organizations": [],
                    "dates": [],
                    "other": [],
                    "error": "Failed to parse entity extraction response"
                }
                
        except Exception as e:
            logger.error(f"Entity extraction failed: {str(e)}")
            raise
    
    def _get_arabic_enhancement_prompt(self, enhancement_type: str) -> str:
        """Get system prompt for Arabic text enhancement."""
        prompts = {
            "grammar": "You are an expert in Arabic grammar and language. Your task is to correct grammatical errors, improve sentence structure, and enhance clarity while preserving the original meaning and style of the text.",
            "diacritization": "You are an expert in Arabic diacritization (تشكيل). Add appropriate diacritical marks to help with proper pronunciation and meaning disambiguation.",
            "normalization": "You are an expert in Arabic language normalization. Convert dialectal Arabic to Modern Standard Arabic while preserving the original meaning and intent."
        }
        return prompts.get(enhancement_type, prompts["grammar"])
    
    async def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models from the LLM service."""
        try:
            await self._ensure_session()
            
            async with self.session.get(f"{self.server_url}/api/tags", timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    data = await response.json()
                    models = []
                    for model in data.get('models', []):
                        models.append({
                            'name': model.get('name', ''),
                            'size': model.get('size', 0),
                            'modified_at': model.get('modified_at', ''),
                            'digest': model.get('digest', ''),
                            'details': model.get('details', {})
                        })
                    
                    # Close the session after use
                    if self.session and not self.session.closed:
                        await self.session.close()
                        self.session = None
                    
                    return models
                else:
                    logger.error(f"Failed to get models, status: {response.status}")
                    # Close the session on error
                    if self.session and not self.session.closed:
                        await self.session.close()
                        self.session = None
                    return []
        except Exception as e:
            logger.error(f"Failed to get available models: {str(e)}")
            # Ensure session is closed on error
            if self.session and not self.session.closed:
                try:
                    await self.session.close()
                except:
                    pass
                self.session = None
            return []
