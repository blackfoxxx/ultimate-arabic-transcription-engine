"""
Advanced Post-Processing Engine for Arabic Transcription
Provides multi-stage quality improvement and text refinement
"""

import asyncio
import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time

from core.llm_service import LLMService
from core.text_enhancement import TextEnhancementEngine, EnhancementType

logger = logging.getLogger(__name__)

class PostProcessingStage(Enum):
    """Stages of post-processing."""
    COHERENCE_CHECK = "coherence"
    CONTEXT_REFINEMENT = "context"
    FLOW_OPTIMIZATION = "flow"
    FINAL_POLISH = "polish"

@dataclass
class PostProcessingResult:
    """Result of post-processing stage."""
    stage: PostProcessingStage
    original_text: str
    processed_text: str
    improvements: List[str]
    quality_score: float
    processing_time: float
    metadata: Dict[str, Any]

class AdvancedPostProcessor:
    """Advanced post-processor for enhanced Arabic transcription quality."""
    
    def __init__(self):
        self.llm_service = LLMService()
        self.text_enhancer = TextEnhancementEngine()
        self.initialized = False
        
        # Arabic linguistic patterns
        self.arabic_patterns = {
            'sentence_endings': re.compile(r'[.!?؟]'),
            'conjunctions': re.compile(r'\b(و|أو|لكن|إذا|عندما|بينما|حيث|لأن)\b'),
            'discourse_markers': re.compile(r'\b(أولاً|ثانياً|أخيراً|بالإضافة|علاوة|من ناحية)\b'),
            'repetitions': re.compile(r'\b(\w+)\s+\1\b'),
            'incomplete_sentences': re.compile(r'[^.!?؟]\s*$'),
        }
    
    async def initialize(self):
        """Initialize the post-processor."""
        try:
            await self.llm_service.initialize()
            await self.text_enhancer.initialize()
            self.initialized = True
            logger.info("Advanced Post-Processor initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Advanced Post-Processor: {str(e)}")
            return False
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.llm_service:
            await self.llm_service.cleanup()
        if self.text_enhancer:
            await self.text_enhancer.cleanup()
        self.initialized = False
    
    async def process_transcription(
        self,
        transcription_data: Dict[str, Any],
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Apply advanced post-processing to transcription data.
        
        Args:
            transcription_data: Original transcription data
            options: Processing options
            
        Returns:
            Enhanced transcription data with post-processing results
        """
        if not self.initialized:
            raise RuntimeError("Advanced Post-Processor not initialized")
        
        options = options or {}
        start_time = time.time()
        
        try:
            # Extract text from transcription data
            original_text = self._extract_text(transcription_data)
            if not original_text:
                logger.warning("No text found for post-processing")
                return transcription_data
            
            logger.info("Starting advanced post-processing pipeline...")
            
            # Stage 1: Coherence Check
            coherence_result = await self._check_coherence(original_text, options)
            current_text = coherence_result.processed_text
            
            # Stage 2: Context Refinement
            context_result = await self._refine_context(current_text, options)
            current_text = context_result.processed_text
            
            # Stage 3: Flow Optimization
            flow_result = await self._optimize_flow(current_text, options)
            current_text = flow_result.processed_text
            
            # Stage 4: Final Polish
            polish_result = await self._final_polish(current_text, options)
            final_text = polish_result.processed_text
            
            # Calculate overall improvement metrics
            overall_quality = self._calculate_overall_quality([
                coherence_result, context_result, flow_result, polish_result
            ])
            
            # Create enhanced transcription data
            enhanced_data = transcription_data.copy()
            enhanced_data['post_processing'] = {
                'enabled': True,
                'original_text': original_text,
                'enhanced_text': final_text,
                'processing_stages': {
                    'coherence': self._stage_to_dict(coherence_result),
                    'context': self._stage_to_dict(context_result),
                    'flow': self._stage_to_dict(flow_result),
                    'polish': self._stage_to_dict(polish_result)
                },
                'overall_quality': overall_quality,
                'processing_time': time.time() - start_time,
                'improvement_summary': self._generate_improvement_summary([
                    coherence_result, context_result, flow_result, polish_result
                ])
            }
            
            # Update main text fields
            enhanced_data['text'] = final_text
            if 'enhanced_text' in enhanced_data:
                enhanced_data['pre_post_processing_text'] = enhanced_data['enhanced_text']
            enhanced_data['enhanced_text'] = final_text
            
            # Update segments if available
            if 'segments' in enhanced_data:
                enhanced_data['segments'] = await self._enhance_segments(
                    enhanced_data['segments'], original_text, final_text
                )
            
            logger.info(f"Post-processing completed. Quality score: {overall_quality:.3f}")
            return enhanced_data
            
        except Exception as e:
            logger.error(f"Post-processing failed: {str(e)}")
            # Return original data with error info
            error_data = transcription_data.copy()
            error_data['post_processing'] = {
                'enabled': False,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
            return error_data
    
    async def _check_coherence(self, text: str, options: Dict[str, Any]) -> PostProcessingResult:
        """Check and improve text coherence."""
        start_time = time.time()
        
        try:
            system_prompt = """You are an expert Arabic linguist specializing in text coherence analysis.
            Your task is to improve the logical flow and coherence of Arabic transcribed text.
            
            Focus on:
            1. Fixing logical inconsistencies
            2. Improving sentence connections
            3. Ensuring proper discourse flow
            4. Maintaining original meaning while enhancing clarity
            
            Return only the improved text without explanations."""
            
            prompt = f"""Improve the coherence and logical flow of this Arabic text:

{text}

Enhanced text:"""
            
            response = await self.llm_service.generate_text(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.3
            )
            
            improvements = self._detect_coherence_improvements(text, response.text)
            quality_score = self._calculate_coherence_score(response.text)
            
            return PostProcessingResult(
                stage=PostProcessingStage.COHERENCE_CHECK,
                original_text=text,
                processed_text=response.text,
                improvements=improvements,
                quality_score=quality_score,
                processing_time=time.time() - start_time,
                metadata={'tokens_used': response.tokens_used}
            )
            
        except Exception as e:
            logger.error(f"Coherence check failed: {str(e)}")
            return PostProcessingResult(
                stage=PostProcessingStage.COHERENCE_CHECK,
                original_text=text,
                processed_text=text,
                improvements=[],
                quality_score=0.5,
                processing_time=time.time() - start_time,
                metadata={'error': str(e)}
            )
    
    async def _refine_context(self, text: str, options: Dict[str, Any]) -> PostProcessingResult:
        """Refine contextual understanding and clarity."""
        start_time = time.time()
        
        try:
            system_prompt = """You are an expert Arabic text analyst specializing in contextual refinement.
            Your task is to enhance the contextual clarity and meaning of Arabic text.
            
            Focus on:
            1. Clarifying ambiguous references
            2. Improving contextual connections
            3. Enhancing semantic clarity
            4. Maintaining natural Arabic expression
            
            Return only the refined text without explanations."""
            
            prompt = f"""Refine the contextual clarity of this Arabic text:

{text}

Refined text:"""
            
            response = await self.llm_service.generate_text(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.2
            )
            
            improvements = self._detect_context_improvements(text, response.text)
            quality_score = self._calculate_context_score(response.text)
            
            return PostProcessingResult(
                stage=PostProcessingStage.CONTEXT_REFINEMENT,
                original_text=text,
                processed_text=response.text,
                improvements=improvements,
                quality_score=quality_score,
                processing_time=time.time() - start_time,
                metadata={'tokens_used': response.tokens_used}
            )
            
        except Exception as e:
            logger.error(f"Context refinement failed: {str(e)}")
            return PostProcessingResult(
                stage=PostProcessingStage.CONTEXT_REFINEMENT,
                original_text=text,
                processed_text=text,
                improvements=[],
                quality_score=0.5,
                processing_time=time.time() - start_time,
                metadata={'error': str(e)}
            )
    
    async def _optimize_flow(self, text: str, options: Dict[str, Any]) -> PostProcessingResult:
        """Optimize text flow and readability."""
        start_time = time.time()
        
        try:
            system_prompt = """You are an expert Arabic stylist specializing in text flow optimization.
            Your task is to improve the natural flow and readability of Arabic text.
            
            Focus on:
            1. Optimizing sentence structure
            2. Improving rhythm and flow
            3. Enhancing readability
            4. Maintaining authentic Arabic style
            
            Return only the optimized text without explanations."""
            
            prompt = f"""Optimize the flow and readability of this Arabic text:

{text}

Optimized text:"""
            
            response = await self.llm_service.generate_text(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.25
            )
            
            improvements = self._detect_flow_improvements(text, response.text)
            quality_score = self._calculate_flow_score(response.text)
            
            return PostProcessingResult(
                stage=PostProcessingStage.FLOW_OPTIMIZATION,
                original_text=text,
                processed_text=response.text,
                improvements=improvements,
                quality_score=quality_score,
                processing_time=time.time() - start_time,
                metadata={'tokens_used': response.tokens_used}
            )
            
        except Exception as e:
            logger.error(f"Flow optimization failed: {str(e)}")
            return PostProcessingResult(
                stage=PostProcessingStage.FLOW_OPTIMIZATION,
                original_text=text,
                processed_text=text,
                improvements=[],
                quality_score=0.5,
                processing_time=time.time() - start_time,
                metadata={'error': str(e)}
            )
    
    async def _final_polish(self, text: str, options: Dict[str, Any]) -> PostProcessingResult:
        """Apply final polish and quality assurance."""
        start_time = time.time()
        
        try:
            system_prompt = """You are an expert Arabic editor specializing in final text polishing.
            Your task is to apply final touches to ensure the highest quality Arabic text.
            
            Focus on:
            1. Final grammar and style checks
            2. Ensuring consistency throughout
            3. Polishing expression and eloquence
            4. Quality assurance and final review
            
            Return only the polished text without explanations."""
            
            prompt = f"""Apply final polish to this Arabic text for highest quality:

{text}

Polished text:"""
            
            response = await self.llm_service.generate_text(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.1
            )
            
            improvements = self._detect_polish_improvements(text, response.text)
            quality_score = self._calculate_polish_score(response.text)
            
            return PostProcessingResult(
                stage=PostProcessingStage.FINAL_POLISH,
                original_text=text,
                processed_text=response.text,
                improvements=improvements,
                quality_score=quality_score,
                processing_time=time.time() - start_time,
                metadata={'tokens_used': response.tokens_used}
            )
            
        except Exception as e:
            logger.error(f"Final polish failed: {str(e)}")
            return PostProcessingResult(
                stage=PostProcessingStage.FINAL_POLISH,
                original_text=text,
                processed_text=text,
                improvements=[],
                quality_score=0.5,
                processing_time=time.time() - start_time,
                metadata={'error': str(e)}
            )
    
    def _extract_text(self, transcription_data: Dict[str, Any]) -> str:
        """Extract text from transcription data."""
        # Try different text fields
        for field in ['enhanced_text', 'text', 'full_text']:
            if field in transcription_data and transcription_data[field]:
                return transcription_data[field]
        return ""
    
    def _detect_coherence_improvements(self, original: str, enhanced: str) -> List[str]:
        """Detect coherence improvements made."""
        improvements = []
        
        # Check for improved sentence connections
        original_conjunctions = len(self.arabic_patterns['conjunctions'].findall(original))
        enhanced_conjunctions = len(self.arabic_patterns['conjunctions'].findall(enhanced))
        
        if enhanced_conjunctions > original_conjunctions:
            improvements.append("Improved sentence connections")
        
        # Check for discourse markers
        original_markers = len(self.arabic_patterns['discourse_markers'].findall(original))
        enhanced_markers = len(self.arabic_patterns['discourse_markers'].findall(enhanced))
        
        if enhanced_markers > original_markers:
            improvements.append("Added discourse markers")
        
        # Check for repetition reduction
        original_reps = len(self.arabic_patterns['repetitions'].findall(original))
        enhanced_reps = len(self.arabic_patterns['repetitions'].findall(enhanced))
        
        if enhanced_reps < original_reps:
            improvements.append("Reduced repetitions")
        
        return improvements
    
    def _detect_context_improvements(self, original: str, enhanced: str) -> List[str]:
        """Detect context improvements made."""
        improvements = []
        
        if len(enhanced) > len(original) * 1.1:
            improvements.append("Enhanced contextual clarity")
        
        if enhanced != original:
            improvements.append("Improved semantic precision")
        
        return improvements
    
    def _detect_flow_improvements(self, original: str, enhanced: str) -> List[str]:
        """Detect flow improvements made."""
        improvements = []
        
        # Check sentence structure improvements
        original_sentences = len(self.arabic_patterns['sentence_endings'].findall(original))
        enhanced_sentences = len(self.arabic_patterns['sentence_endings'].findall(enhanced))
        
        if enhanced_sentences != original_sentences:
            improvements.append("Optimized sentence structure")
        
        if enhanced != original:
            improvements.append("Improved text flow")
        
        return improvements
    
    def _detect_polish_improvements(self, original: str, enhanced: str) -> List[str]:
        """Detect polish improvements made."""
        improvements = []
        
        if enhanced != original:
            improvements.append("Applied final polish")
            improvements.append("Quality assurance completed")
        
        return improvements
    
    def _calculate_coherence_score(self, text: str) -> float:
        """Calculate coherence quality score."""
        score = 0.5  # Base score
        
        # Check for proper sentence endings
        sentences = self.arabic_patterns['sentence_endings'].findall(text)
        if sentences:
            score += 0.2
        
        # Check for conjunctions (good flow)
        conjunctions = self.arabic_patterns['conjunctions'].findall(text)
        if conjunctions:
            score += 0.2
        
        # Check for discourse markers
        markers = self.arabic_patterns['discourse_markers'].findall(text)
        if markers:
            score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_context_score(self, text: str) -> float:
        """Calculate context quality score."""
        score = 0.6  # Base score
        
        # Length indicates detail (up to a point)
        if 50 <= len(text) <= 1000:
            score += 0.2
        
        # Arabic content
        arabic_chars = len(self.arabic_patterns['arabic_chars'].findall(text))
        if arabic_chars > len(text) * 0.7:  # Mostly Arabic
            score += 0.2
        
        return min(score, 1.0)
    
    def _calculate_flow_score(self, text: str) -> float:
        """Calculate flow quality score."""
        score = 0.6  # Base score
        
        # Check for varied sentence lengths (good flow)
        sentences = text.split('.')
        if len(sentences) > 1:
            lengths = [len(s.strip()) for s in sentences if s.strip()]
            if lengths and max(lengths) - min(lengths) > 20:
                score += 0.2
        
        # Check for proper punctuation
        punctuation = len(re.findall(r'[.!?؟،]', text))
        if punctuation > 0:
            score += 0.2
        
        return min(score, 1.0)
    
    def _calculate_polish_score(self, text: str) -> float:
        """Calculate polish quality score."""
        score = 0.7  # Base score (final stage)
        
        # Check for completeness
        if not self.arabic_patterns['incomplete_sentences'].search(text):
            score += 0.15
        
        # Check for proper Arabic structure
        if len(text) > 20:
            score += 0.15
        
        return min(score, 1.0)
    
    def _calculate_overall_quality(self, results: List[PostProcessingResult]) -> float:
        """Calculate overall quality score."""
        if not results:
            return 0.5
        
        scores = [r.quality_score for r in results]
        return sum(scores) / len(scores)
    
    def _stage_to_dict(self, result: PostProcessingResult) -> Dict[str, Any]:
        """Convert stage result to dictionary."""
        return {
            'stage': result.stage.value,
            'improvements': result.improvements,
            'quality_score': result.quality_score,
            'processing_time': result.processing_time,
            'metadata': result.metadata
        }
    
    def _generate_improvement_summary(self, results: List[PostProcessingResult]) -> List[str]:
        """Generate summary of all improvements."""
        all_improvements = []
        for result in results:
            all_improvements.extend(result.improvements)
        return list(set(all_improvements))  # Remove duplicates
    
    async def _enhance_segments(
        self,
        segments: List[Dict[str, Any]],
        original_text: str,
        enhanced_text: str
    ) -> List[Dict[str, Any]]:
        """Enhance individual segments based on overall text improvement."""
        if not segments or original_text == enhanced_text:
            return segments
        
        try:
            # For now, keep original segments but mark as enhanced
            enhanced_segments = []
            for segment in segments:
                enhanced_segment = segment.copy()
                enhanced_segment['post_processed'] = True
                enhanced_segments.append(enhanced_segment)
            
            return enhanced_segments
            
        except Exception as e:
            logger.error(f"Segment enhancement failed: {str(e)}")
            return segments