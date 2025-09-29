"""
Multi-Model Transcription Service for Arabic STT Platform
Handles simultaneous transcription using multiple models and provides comparison results.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional
from pathlib import Path
import difflib
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from config import Config
from core.transcription_engine import TranscriptionEngine
from core.enhanced_arabic_transcription_engine import EnhancedArabicTranscriptionEngine
from core.advanced_arabic_transcription_engine import AdvancedArabicTranscriptionEngine
from core.ultimate_arabic_transcription_engine import UltimateArabicTranscriptionEngine
from core.api_transcription_engine_fixed import APITranscriptionEngine

logger = logging.getLogger(__name__)

class MultiModelTranscriptionService:
    """Service for running multiple transcription models simultaneously and comparing results."""
    
    def __init__(self):
        self.config = Config()
        self.engines = {}
        self.available_models = {
            'standard_whisper': {
                'name': 'Standard Whisper',
                'description': 'Basic Whisper model with standard settings',
                'engine_class': TranscriptionEngine,
                'model_sizes': ['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3'],
                'default_size': 'medium'
            },
            'enhanced_arabic': {
                'name': 'Enhanced Arabic v1.0',
                'description': 'Arabic-optimized Whisper with enhanced preprocessing',
                'engine_class': EnhancedArabicTranscriptionEngine,
                'model_sizes': ['small', 'medium', 'large', 'large-v2', 'large-v3'],
                'default_size': 'medium'
            },
            'advanced_arabic': {
                'name': 'Advanced Arabic v2.0',
                'description': 'High-performance Arabic transcription with advanced optimizations',
                'engine_class': AdvancedArabicTranscriptionEngine,
                'model_sizes': ['medium', 'large', 'large-v2', 'large-v3'],
                'default_size': 'large-v2'
            },
            'ultimate_arabic': {
                'name': 'Ultimate Arabic v3.0',
                'description': 'Latest generation Arabic transcription with superior quality',
                'engine_class': UltimateArabicTranscriptionEngine,
                'model_sizes': ['large-v2', 'large-v3'],
                'default_size': 'large-v2'
            },
            'openai_api': {
                'name': 'OpenAI Whisper API',
                'description': 'Cloud-based OpenAI Whisper API',
                'engine_class': APITranscriptionEngine,
                'model_sizes': ['whisper-1'],
                'default_size': 'whisper-1'
            }
        }
        
    async def get_available_models(self) -> Dict[str, Any]:
        """Get list of available transcription models."""
        models = {}
        for model_id, model_info in self.available_models.items():
            # Check if API key is available for OpenAI
            if model_id == 'openai_api' and not self.config.OPENAI_API_KEY:
                continue
                
            models[model_id] = {
                'name': model_info['name'],
                'description': model_info['description'],
                'model_sizes': model_info['model_sizes'],
                'default_size': model_info['default_size'],
                'available': True
            }
        return models
    
    async def transcribe_multi_model(
        self,
        audio_path: str,
        selected_models: List[Dict[str, str]],
        language: str = 'ar',
        **kwargs
    ) -> Dict[str, Any]:
        """
        Transcribe audio using multiple models simultaneously.
        
        Args:
            audio_path: Path to audio file
            selected_models: List of model configurations [{'model_id': 'standard_whisper', 'model_size': 'medium'}, ...]
            language: Language code
            **kwargs: Additional parameters
            
        Returns:
            Dict containing results from all models and comparison data
        """
        try:
            logger.info(f"Starting multi-model transcription with {len(selected_models)} models")
            start_time = time.time()
            
            # Validate audio file
            if not Path(audio_path).exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            # Initialize results structure
            results = {
                'audio_file': audio_path,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_models': len(selected_models),
                'language': language,
                'model_results': {},
                'comparison': {},
                'processing_times': {},
                'errors': {}
            }
            
            # Run transcriptions concurrently
            tasks = []
            for model_config in selected_models:
                task = self._transcribe_single_model(
                    audio_path, 
                    model_config['model_id'], 
                    model_config.get('model_size'), 
                    language, 
                    **kwargs
                )
                tasks.append((model_config['model_id'], task))
            
            # Wait for all transcriptions to complete
            for model_id, task in tasks:
                try:
                    model_start = time.time()
                    result = await task
                    processing_time = time.time() - model_start
                    
                    results['model_results'][model_id] = result
                    results['processing_times'][model_id] = processing_time
                    
                    logger.info(f"Model {model_id} completed in {processing_time:.2f}s")
                    
                except Exception as e:
                    logger.error(f"Error in model {model_id}: {str(e)}")
                    results['errors'][model_id] = str(e)
            
            # Generate comparison analysis
            if len(results['model_results']) > 1:
                results['comparison'] = self._generate_comparison(results['model_results'])
            
            total_time = time.time() - start_time
            results['total_processing_time'] = total_time
            
            logger.info(f"Multi-model transcription completed in {total_time:.2f}s")
            return results
            
        except Exception as e:
            logger.error(f"Multi-model transcription failed: {str(e)}")
            return {
                'audio_file': audio_path,
                'error': str(e),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
    
    async def _transcribe_single_model(
        self,
        audio_path: str,
        model_id: str,
        model_size: Optional[str],
        language: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Transcribe using a single model."""
        try:
            model_info = self.available_models.get(model_id)
            if not model_info:
                raise ValueError(f"Unknown model: {model_id}")
            
            # Use default size if not specified
            if not model_size:
                model_size = model_info['default_size']
            
            # Initialize engine
            engine_class = model_info['engine_class']
            
            if model_id == 'ultimate_arabic':
                # Ultimate Arabic engine has different initialization
                engine = engine_class(model_size=model_size)
                if not engine.initialize_model():
                    raise RuntimeError("Failed to initialize Ultimate Arabic engine")
                
                # Use the engine's transcribe method
                result = engine.transcribe(audio_path)
                
                return {
                    'engine': model_id,
                    'model_size': model_size,
                    'text': result.get('text', ''),
                    'segments': result.get('segments', []),
                    'confidence': result.get('confidence_score', 0.0),
                    'language': result.get('language', language),
                    'processing_time': result.get('processing_time', 0.0),
                    'quality_metrics': result.get('quality_metrics', {}),
                    'word_count': len(result.get('text', '').split()) if result.get('text') else 0
                }
                
            elif model_id == 'openai_api':
                # API engine
                engine = engine_class()
                result = await engine.transcribe_openai(
                    audio_path=audio_path,
                    model=model_size,
                    language=language,
                    **kwargs
                )
                
                return {
                    'engine': model_id,
                    'model_size': model_size,
                    'text': result.get('text', ''),
                    'segments': result.get('segments', []),
                    'confidence': result.get('confidence', 0.0),
                    'language': result.get('language', language),
                    'processing_time': result.get('processing_time', 0.0),
                    'word_count': len(result.get('text', '').split()) if result.get('text') else 0
                }
                
            else:
                # Standard engines (TranscriptionEngine, EnhancedArabicTranscriptionEngine, AdvancedArabicTranscriptionEngine)
                engine = engine_class()
                result = await engine.transcribe(
                    audio_path=audio_path,
                    model_size=model_size,
                    language=language,
                    **kwargs
                )
                
                return {
                    'engine': model_id,
                    'model_size': model_size,
                    'text': result.get('text', ''),
                    'segments': result.get('segments', []),
                    'confidence': result.get('confidence', 0.0),
                    'language': result.get('language', language),
                    'processing_time': result.get('processing_time', 0.0),
                    'word_count': len(result.get('text', '').split()) if result.get('text') else 0
                }
                
        except Exception as e:
            logger.error(f"Error transcribing with {model_id}: {str(e)}")
            raise
    
    def _generate_comparison(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comparison analysis between model results."""
        try:
            texts = {}
            confidences = {}
            word_counts = {}
            
            # Extract texts and metrics
            for model_id, result in model_results.items():
                texts[model_id] = result.get('text', '')
                confidences[model_id] = result.get('confidence', 0.0)
                word_counts[model_id] = result.get('word_count', 0)
            
            # Calculate text similarities
            similarities = {}
            model_ids = list(texts.keys())
            
            for i, model1 in enumerate(model_ids):
                for j, model2 in enumerate(model_ids[i+1:], i+1):
                    similarity = self._calculate_text_similarity(texts[model1], texts[model2])
                    similarities[f"{model1}_vs_{model2}"] = similarity
            
            # Find best performing model based on confidence and length
            best_model = max(model_results.keys(), 
                           key=lambda x: (confidences.get(x, 0) * 0.7 + 
                                        min(word_counts.get(x, 0) / 100, 1.0) * 0.3))
            
            # Generate text differences
            differences = {}
            if len(texts) >= 2:
                base_model = list(texts.keys())[0]
                base_text = texts[base_model]
                
                for model_id, text in texts.items():
                    if model_id != base_model:
                        diff = self._generate_text_diff(base_text, text)
                        differences[f"{base_model}_vs_{model_id}"] = diff
            
            return {
                'similarities': similarities,
                'best_model': best_model,
                'confidence_scores': confidences,
                'word_counts': word_counts,
                'text_differences': differences,
                'average_confidence': sum(confidences.values()) / len(confidences) if confidences else 0.0,
                'total_unique_words': len(set(' '.join(texts.values()).split())),
                'consensus_level': self._calculate_consensus_level(similarities)
            }
            
        except Exception as e:
            logger.error(f"Error generating comparison: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts using difflib."""
        try:
            if not text1 or not text2:
                return 0.0
            
            # Use sequence matcher for similarity
            matcher = difflib.SequenceMatcher(None, text1.lower(), text2.lower())
            return matcher.ratio()
            
        except Exception:
            return 0.0
    
    def _generate_text_diff(self, text1: str, text2: str) -> Dict[str, Any]:
        """Generate detailed text differences."""
        try:
            # Split into words for better comparison
            words1 = text1.split()
            words2 = text2.split()
            
            # Generate unified diff
            diff = list(difflib.unified_diff(
                words1, words2,
                fromfile='Model 1',
                tofile='Model 2',
                lineterm=''
            ))
            
            # Count changes
            additions = sum(1 for line in diff if line.startswith('+') and not line.startswith('+++'))
            deletions = sum(1 for line in diff if line.startswith('-') and not line.startswith('---'))
            
            return {
                'diff_lines': diff,
                'additions': additions,
                'deletions': deletions,
                'total_changes': additions + deletions,
                'similarity_ratio': self._calculate_text_similarity(text1, text2)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _calculate_consensus_level(self, similarities: Dict[str, float]) -> str:
        """Calculate consensus level based on similarities."""
        if not similarities:
            return 'unknown'
        
        avg_similarity = sum(similarities.values()) / len(similarities)
        
        if avg_similarity >= 0.9:
            return 'high'
        elif avg_similarity >= 0.7:
            return 'medium'
        elif avg_similarity >= 0.5:
            return 'low'
        else:
            return 'very_low'
    
    async def transcribe_with_multiple_models(
        self,
        audio_path: str,
        selected_models: List[Dict[str, str]],
        processing_mode: str = 'local',
        model_size: str = 'medium',
        language: str = 'ar',
        job_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Wrapper method for multi-model transcription that returns results in the expected format.
        This method is called from app.py and ensures compatibility with the existing job processing system.
        """
        try:
            logger.info(f"Starting multi-model transcription with {len(selected_models)} models")
            start_time = time.time()
            
            # Call the main multi-model transcription method
            multi_model_results = await self.transcribe_multi_model(
                audio_path=audio_path,
                selected_models=selected_models,
                language=language,
                **kwargs
            )
            
            # Extract the best result for primary display
            best_result = None
            best_model_id = None
            
            if 'model_results' in multi_model_results and multi_model_results['model_results']:
                # Get the best model from comparison if available
                if 'comparison' in multi_model_results and 'best_model' in multi_model_results['comparison']:
                    best_model_id = multi_model_results['comparison']['best_model']
                    best_result = multi_model_results['model_results'].get(best_model_id)
                else:
                    # Fallback to first result
                    best_model_id = list(multi_model_results['model_results'].keys())[0]
                    best_result = multi_model_results['model_results'][best_model_id]
            
            # Calculate total processing time
            total_processing_time = time.time() - start_time
            
            # Format result in the expected structure for job processing
            result = {
                'text': best_result.get('text', '') if best_result else '',
                'segments': best_result.get('segments', []) if best_result else [],
                'language': language,
                'confidence': best_result.get('confidence', 0.0) if best_result else 0.0,
                'processing_time': total_processing_time,
                'model_size': model_size,
                'processing_info': {
                    'processing_mode': 'multi_model',
                    'model_used': f"Multi-Model ({len(selected_models)} models)",
                    'best_model': best_model_id if best_model_id else 'Unknown',
                    'total_models': len(selected_models),
                    'processing_time': total_processing_time,
                    'language': language,
                    'model_size': model_size
                },
                'multi_model_results': multi_model_results,  # Include full multi-model data
                'success': True
            }
            
            logger.info(f"Multi-model transcription completed in {total_processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Multi-model transcription wrapper failed: {str(e)}")
            return {
                'error': str(e),
                'success': False,
                'processing_time': time.time() - start_time if 'start_time' in locals() else 0,
                'processing_info': {
                    'processing_mode': 'multi_model',
                    'model_used': 'Error',
                    'processing_time': time.time() - start_time if 'start_time' in locals() else 0,
                    'language': language,
                    'model_size': model_size
                }
            }

    def get_model_recommendations(self, audio_duration: float, file_size_mb: float) -> List[str]:
        """Get recommended models based on audio characteristics."""
        recommendations = []
        
        # For short audio (< 5 minutes)
        if audio_duration < 300:
            recommendations.extend(['standard_whisper', 'enhanced_arabic'])
        
        # For medium audio (5-30 minutes)
        elif audio_duration < 1800:
            recommendations.extend(['enhanced_arabic', 'advanced_arabic'])
        
        # For long audio (> 30 minutes)
        else:
            recommendations.extend(['advanced_arabic', 'ultimate_arabic'])
        
        # Add API option if available
        if self.config.OPENAI_API_KEY and file_size_mb < 25:  # OpenAI limit
            recommendations.append('openai_api')
        
        return recommendations[:3]  # Return top 3 recommendations