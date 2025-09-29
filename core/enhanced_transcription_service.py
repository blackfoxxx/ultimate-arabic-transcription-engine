"""
Enhanced Unified Transcription Service with LLM Integration
Combines STT with advanced text analysis and enhancement
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import time
import json

from config import Config
from core.unified_transcription_service import UnifiedTranscriptionService
from core.llm_service import LLMService
from core.text_enhancement import TextEnhancementEngine, EnhancementType
from core.text_analysis import TextAnalysisEngine, AnalysisType
from core.advanced_post_processor import AdvancedPostProcessor

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class EnhancedTranscriptionService:
    """Enhanced transcription service with LLM-powered text analysis and enhancement."""
    
    def __init__(self):
        self.config = Config()
        
        # Core transcription service
        self.transcription_service = UnifiedTranscriptionService()
        
        # LLM services
        self.llm_service = LLMService()
        self.text_enhancer = TextEnhancementEngine()
        self.text_analyzer = TextAnalysisEngine()
        self.post_processor = AdvancedPostProcessor()
        
        # Service state
        self.initialized = False
        self.llm_enabled = self.config.LLM_ENABLED
        
    async def initialize(self):
        """Initialize all services."""
        try:
            logger.info("Initializing Enhanced Transcription Service...")
            
            # Always initialize core transcription
            self.initialized = True
            
            # Initialize LLM services if enabled
            if self.llm_enabled:
                logger.info("Initializing LLM services...")
                
                await self.llm_service.initialize()
                await self.text_enhancer.initialize()
                await self.text_analyzer.initialize()
                await self.post_processor.initialize()
                
                logger.info("LLM services initialized successfully")
            else:
                logger.info("LLM services disabled - using basic transcription only")
                
            logger.info("Enhanced Transcription Service ready")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Enhanced Transcription Service: {str(e)}")
            return False
    
    async def cleanup(self):
        """Cleanup all services."""
        try:
            if self.llm_enabled:
                await self.llm_service.cleanup()
                await self.text_enhancer.cleanup()
                await self.text_analyzer.cleanup()
                await self.post_processor.cleanup()
            
            self.initialized = False
            logger.info("Enhanced Transcription Service cleaned up")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")
    
    async def transcribe_with_llm(
        self,
        audio_path: str,
        processing_mode: Optional[str] = None,
        model_size: str = 'medium',
        language: str = 'ar',
        enable_enhancement: bool = True,
        enable_analysis: bool = True,
        enhancement_options: Optional[Dict[str, Any]] = None,
        analysis_options: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Transcribe audio with optional LLM enhancement and analysis.
        
        Args:
            audio_path: Path to audio file
            processing_mode: 'local' or 'api'
            model_size: Whisper model size
            language: Language code
            enable_enhancement: Whether to enhance text with LLM
            enable_analysis: Whether to analyze text with LLM
            enhancement_options: Options for text enhancement
            analysis_options: Options for text analysis
            **kwargs: Additional parameters
            
        Returns:
            Dict containing transcription, enhancement, and analysis results
        """
        if not self.initialized:
            raise RuntimeError("Enhanced Transcription Service not initialized")
        
        start_time = time.time()
        
        try:
            logger.info("Starting enhanced transcription process...")
            
            # Step 1: Basic transcription
            logger.info("Performing speech-to-text transcription...")
            transcription_result = await self.transcription_service.transcribe(
                audio_path=audio_path,
                processing_mode=processing_mode,
                model_size=model_size,
                language=language,
                **kwargs
            )
            
            # Extract transcribed text - handle both old and new format
            transcribed_text = transcription_result.get('text', '')
            if not transcribed_text and 'transcript' in transcription_result:
                transcribed_text = transcription_result['transcript'].get('full_text', '')
            
            if not transcribed_text:
                logger.warning("No text was transcribed")
                return transcription_result
            
            # Build enhanced result starting with basic transcription
            enhanced_result = transcription_result.copy()
            enhanced_result['llm_processing'] = {
                'enabled': self.llm_enabled,
                'enhancement_applied': False,
                'analysis_applied': False
            }
            
            # Skip LLM processing if disabled
            if not self.llm_enabled:
                logger.info("LLM processing disabled - returning basic transcription")
                enhanced_result['llm_processing']['disabled_reason'] = 'LLM_ENABLED=False'
                return enhanced_result
            
            # Step 2: Text Enhancement (if enabled)
            if enable_enhancement:
                try:
                    logger.info("Enhancing transcribed text...")
                    enhancement_result = await self._enhance_text(
                        transcribed_text, enhancement_options or {}
                    )
                    enhanced_result['enhancement'] = enhancement_result
                    enhanced_result['llm_processing']['enhancement_applied'] = True
                    
                    # Use enhanced text for further processing
                    best_enhanced_text = self._get_best_enhanced_text(enhancement_result)
                    if best_enhanced_text:
                        transcribed_text = best_enhanced_text
                        enhanced_result['enhanced_text'] = best_enhanced_text
                    
                except Exception as e:
                    logger.error(f"Text enhancement failed: {str(e)}")
                    enhanced_result['enhancement'] = {'error': str(e)}
            
            # Step 3: Text Analysis (if enabled)
            if enable_analysis:
                try:
                    logger.info("Analyzing transcribed text...")
                    analysis_result = await self._analyze_text(
                        transcribed_text, analysis_options or {}
                    )
                    enhanced_result['analysis'] = analysis_result
                    enhanced_result['llm_processing']['analysis_applied'] = True
                    
                except Exception as e:
                    logger.error(f"Text analysis failed: {str(e)}")
                    enhanced_result['analysis'] = {'error': str(e)}
            
            # Step 4: Advanced Post-Processing (if enabled and LLM available)
            if self.llm_enabled and enhanced_result.get('enhanced_text'):
                try:
                    logger.info("Applying advanced post-processing...")
                    post_processed_result = await self.post_processor.process_transcription(
                        enhanced_result, options.get('post_processing', {})
                    )
                    enhanced_result = post_processed_result
                    enhanced_result['llm_processing']['post_processing_applied'] = True
                    
                except Exception as e:
                    logger.error(f"Advanced post-processing failed: {str(e)}")
                    enhanced_result['post_processing'] = {'error': str(e)}
            
            # Add processing metadata
            total_time = time.time() - start_time
            enhanced_result['processing_time'] = {
                'total': total_time,
                'transcription': transcription_result.get('processing_time', 0),
                'llm_processing': total_time - transcription_result.get('processing_time', 0)
            }
            
            logger.info(f"Enhanced transcription completed in {total_time:.2f} seconds")
            return enhanced_result
            
        except Exception as e:
            logger.error(f"Enhanced transcription failed: {str(e)}")
            raise
    
    async def _enhance_text(
        self,
        text: str,
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Enhance text using LLM."""
        try:
            # Determine enhancement types
            enhancement_types = []
            if options.get('grammar', True):
                enhancement_types.append(EnhancementType.GRAMMAR_CORRECTION)
            if options.get('punctuation', True):
                enhancement_types.append(EnhancementType.PUNCTUATION)
            if options.get('spelling', True):
                enhancement_types.append(EnhancementType.SPELLING)
            if options.get('diacritization', False):
                enhancement_types.append(EnhancementType.DIACRITIZATION)
            if options.get('normalization', False):
                enhancement_types.append(EnhancementType.NORMALIZATION)
            if options.get('style', False):
                enhancement_types.append(EnhancementType.STYLE_IMPROVEMENT)
            
            # Default enhancement types if none specified
            if not enhancement_types:
                enhancement_types = [
                    EnhancementType.GRAMMAR_CORRECTION,
                    EnhancementType.PUNCTUATION,
                    EnhancementType.SPELLING
                ]
            
            # Perform enhancement
            results = await self.text_enhancer.enhance_text(
                text=text,
                enhancement_types=enhancement_types,
                options=options
            )
            
            # Calculate quality metrics
            quality_metrics = self.text_enhancer.get_enhancement_quality_metrics(results)
            
            # Prepare response
            enhancement_response = {
                'original_text': text,
                'enhancements': {},
                'quality_metrics': quality_metrics,
                'best_result': None
            }
            
            # Convert results to serializable format
            best_confidence = 0
            best_text = text
            
            for enhancement_type, result in results.items():
                enhancement_response['enhancements'][enhancement_type] = {
                    'enhanced_text': result.enhanced_text,
                    'confidence_score': result.confidence_score,
                    'changes_made': result.changes_made,
                    'processing_time': result.processing_time
                }
                
                # Track best result
                if result.confidence_score > best_confidence:
                    best_confidence = result.confidence_score
                    best_text = result.enhanced_text
            
            enhancement_response['best_result'] = {
                'text': best_text,
                'confidence': best_confidence
            }
            
            return enhancement_response
            
        except Exception as e:
            logger.error(f"Text enhancement processing failed: {str(e)}")
            raise
    
    async def _analyze_text(
        self,
        text: str,
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze text using LLM."""
        try:
            # Determine analysis types
            analysis_types = []
            if options.get('sentiment', True):
                analysis_types.append(AnalysisType.SENTIMENT)
            if options.get('entities', True):
                analysis_types.append(AnalysisType.ENTITIES)
            if options.get('topics', True):
                analysis_types.append(AnalysisType.TOPICS)
            if options.get('keywords', True):
                analysis_types.append(AnalysisType.KEYWORDS)
            if options.get('complexity', True):
                analysis_types.append(AnalysisType.COMPLEXITY)
            
            # Default analysis types if none specified
            if not analysis_types:
                analysis_types = [
                    AnalysisType.SENTIMENT,
                    AnalysisType.ENTITIES,
                    AnalysisType.TOPICS,
                    AnalysisType.KEYWORDS,
                    AnalysisType.COMPLEXITY
                ]
            
            # Perform analysis
            analysis_report = await self.text_analyzer.analyze_text(
                text=text,
                analysis_types=analysis_types,
                options=options
            )
            
            # Convert to serializable format
            analysis_response = {
                'text': text,
                'language': analysis_report.language,
                'processing_time': analysis_report.processing_time,
                'metadata': analysis_report.metadata
            }
            
            # Add analysis results
            if analysis_report.sentiment:
                analysis_response['sentiment'] = {
                    'score': analysis_report.sentiment.score,
                    'label': analysis_report.sentiment.label,
                    'confidence': analysis_report.sentiment.confidence,
                    'emotions': analysis_report.sentiment.emotions
                }
            
            if analysis_report.entities:
                analysis_response['entities'] = {
                    'persons': analysis_report.entities.persons,
                    'locations': analysis_report.entities.locations,
                    'organizations': analysis_report.entities.organizations,
                    'dates': analysis_report.entities.dates,
                    'other': analysis_report.entities.other
                }
            
            if analysis_report.topics:
                analysis_response['topics'] = {
                    'main_topics': analysis_report.topics.main_topics,
                    'topic_scores': analysis_report.topics.topic_scores,
                    'categories': analysis_report.topics.categories
                }
            
            if analysis_report.complexity:
                analysis_response['complexity'] = {
                    'readability_score': analysis_report.complexity.readability_score,
                    'complexity_level': analysis_report.complexity.complexity_level,
                    'sentence_count': analysis_report.complexity.sentence_count,
                    'word_count': analysis_report.complexity.word_count,
                    'average_sentence_length': analysis_report.complexity.average_sentence_length,
                    'vocabulary_richness': analysis_report.complexity.vocabulary_richness
                }
            
            if analysis_report.keywords:
                analysis_response['keywords'] = analysis_report.keywords
            
            # Add analysis summary
            analysis_response['summary'] = self.text_analyzer.generate_analysis_summary(analysis_report)
            
            return analysis_response
            
        except Exception as e:
            logger.error(f"Text analysis processing failed: {str(e)}")
            raise
    
    def _get_best_enhanced_text(self, enhancement_result: Dict[str, Any]) -> Optional[str]:
        """Get the best enhanced text from enhancement results."""
        try:
            best_result = enhancement_result.get('best_result')
            if best_result and best_result.get('confidence', 0) > 0.5:
                return best_result.get('text')
            return None
        except Exception:
            return None
    
    async def enhance_existing_text(
        self,
        text: str,
        enhancement_options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Enhance existing text without transcription."""
        if not self.llm_enabled:
            return {
                'error': 'LLM processing is disabled',
                'original_text': text
            }
        
        try:
            return await self._enhance_text(text, enhancement_options or {})
        except Exception as e:
            logger.error(f"Text enhancement failed: {str(e)}")
            raise
    
    async def analyze_existing_text(
        self,
        text: str,
        analysis_options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Analyze existing text without transcription."""
        if not self.llm_enabled:
            return {
                'error': 'LLM processing is disabled',
                'text': text
            }
        
        try:
            return await self._analyze_text(text, analysis_options or {})
        except Exception as e:
            logger.error(f"Text analysis failed: {str(e)}")
            raise
    
    async def generate_summary(
        self,
        text: str,
        summary_type: str = 'medium',
        language: str = 'ar'
    ) -> Dict[str, Any]:
        """Generate summary of text."""
        if not self.llm_enabled:
            return {'error': 'LLM processing is disabled'}
        
        try:
            summaries = await self.llm_service.generate_summary(
                text=text,
                summary_type=summary_type,
                language=language
            )
            
            return {
                'original_text': text,
                'summaries': summaries,
                'language': language,
                'summary_type': summary_type
            }
            
        except Exception as e:
            logger.error(f"Summary generation failed: {str(e)}")
            raise
    
    async def get_llm_status(self) -> Dict[str, Any]:
        """Get LLM service status."""
        status = {
            'llm_enabled': self.llm_enabled,
            'initialized': self.initialized,
            'services': {}
        }
        
        if self.llm_enabled:
            try:
                # Get health check status
                health_status = await self.llm_service.health_check()
                logger.debug(f"LLM service health check result: {health_status}")
                
                status['services']['llm_service'] = health_status
                status['services']['text_enhancer'] = self.text_enhancer.initialized
                status['services']['text_analyzer'] = self.text_analyzer.initialized
                
                # Get available models
                available_models = await self.llm_service.get_available_models()
                logger.debug(f"Available models from LLM service: {available_models}")
                status['available_models'] = available_models
                
                status['configuration'] = {
                    'backend': self.config.LLM_BACKEND,
                    'model': self.config.LLM_MODEL,
                    'server_url': self.config.LLM_SERVER_URL,
                    'arabic_model': self.config.ARABIC_LLM_MODEL
                }
                
            except Exception as e:
                logger.error(f"Error getting LLM status: {str(e)}")
                status['error'] = str(e)
        
        return status
    
    # Maintain compatibility with original transcription service
    async def transcribe(self, *args, **kwargs) -> Dict[str, Any]:
        """Fallback to basic transcription for compatibility."""
        result = await self.transcription_service.transcribe(*args, **kwargs)
        
        # Ensure processing_info is present for compatibility with job processing system
        if 'processing_info' not in result:
            # Extract processing parameters from kwargs or use defaults
            processing_mode = kwargs.get('processing_mode', 'local')
            model_size = kwargs.get('model_size', 'medium')
            language = kwargs.get('language', 'ar')
            processing_time = result.get('processing_time', 0)
            
            result['processing_info'] = {
                'processing_mode': processing_mode,
                'model_used': 'Enhanced Transcription Service',
                'processing_time': processing_time,
                'language': language,
                'model_size': model_size,
                'engine_used': 'Enhanced Transcription Service'
            }
        
        return result
    
    def refresh_api_engine(self):
        """Refresh API engine configuration."""
        self.transcription_service.refresh_api_engine()
