"""
Text Enhancement Engine for Arabic STT Platform
Provides advanced text processing and enhancement capabilities using LLM
"""

import asyncio
import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from core.llm_service import LLMService, LLMResponse

logger = logging.getLogger(__name__)

class EnhancementType(Enum):
    """Types of text enhancement."""
    GRAMMAR_CORRECTION = "grammar"
    DIACRITIZATION = "diacritization"
    NORMALIZATION = "normalization"
    PUNCTUATION = "punctuation"
    SPELLING = "spelling"
    STYLE_IMPROVEMENT = "style"

@dataclass
class EnhancementResult:
    """Result of text enhancement."""
    original_text: str
    enhanced_text: str
    enhancement_type: EnhancementType
    confidence_score: float
    changes_made: List[str]
    processing_time: float
    metadata: Dict[str, Any]

class TextEnhancementEngine:
    """Engine for enhancing Arabic text using LLM capabilities."""
    
    def __init__(self):
        self.llm_service = LLMService()
        self.initialized = False
        
        # Arabic text patterns
        self.arabic_patterns = {
            'diacritics': re.compile(r'[\u064B-\u065F\u0670\u0640]'),  # Arabic diacritics
            'arabic_chars': re.compile(r'[\u0600-\u06FF\u0750-\u077F]'),  # Arabic characters
            'punctuation': re.compile(r'[،؛؟!.]'),  # Arabic and common punctuation
        }
    
    async def initialize(self):
        """Initialize the text enhancement engine."""
        try:
            await self.llm_service.initialize()
            self.initialized = True
            logger.info("Text Enhancement Engine initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Text Enhancement Engine: {str(e)}")
            return False
    
    async def cleanup(self):
        """Cleanup engine resources."""
        if self.llm_service:
            await self.llm_service.cleanup()
        self.initialized = False
    
    async def enhance_text(
        self,
        text: str,
        enhancement_types: List[EnhancementType] = None,
        options: Dict[str, Any] = None
    ) -> Dict[str, EnhancementResult]:
        """
        Enhance text with specified enhancement types.
        
        Args:
            text: Text to enhance
            enhancement_types: List of enhancement types to apply
            options: Additional options for enhancement
            
        Returns:
            Dict mapping enhancement type to results
        """
        if not self.initialized:
            raise RuntimeError("Text Enhancement Engine not initialized")
        
        if enhancement_types is None:
            enhancement_types = [
                EnhancementType.GRAMMAR_CORRECTION,
                EnhancementType.PUNCTUATION,
                EnhancementType.SPELLING
            ]
        
        results = {}
        
        try:
            # Process each enhancement type
            for enhancement_type in enhancement_types:
                logger.info(f"Applying {enhancement_type.value} enhancement")
                
                result = await self._apply_enhancement(
                    text, enhancement_type, options or {}
                )
                results[enhancement_type.value] = result
                
                # Use enhanced text as input for next enhancement
                text = result.enhanced_text
            
            return results
            
        except Exception as e:
            logger.error(f"Text enhancement failed: {str(e)}")
            raise
    
    async def _apply_enhancement(
        self,
        text: str,
        enhancement_type: EnhancementType,
        options: Dict[str, Any]
    ) -> EnhancementResult:
        """Apply specific enhancement type."""
        import time
        start_time = time.time()
        
        try:
            if enhancement_type == EnhancementType.GRAMMAR_CORRECTION:
                result = await self._enhance_grammar(text, options)
            elif enhancement_type == EnhancementType.DIACRITIZATION:
                result = await self._add_diacritics(text, options)
            elif enhancement_type == EnhancementType.NORMALIZATION:
                result = await self._normalize_text(text, options)
            elif enhancement_type == EnhancementType.PUNCTUATION:
                result = await self._fix_punctuation(text, options)
            elif enhancement_type == EnhancementType.SPELLING:
                result = await self._fix_spelling(text, options)
            elif enhancement_type == EnhancementType.STYLE_IMPROVEMENT:
                result = await self._improve_style(text, options)
            else:
                raise ValueError(f"Unknown enhancement type: {enhancement_type}")
            
            processing_time = time.time() - start_time
            
            return EnhancementResult(
                original_text=text,
                enhanced_text=result['text'],
                enhancement_type=enhancement_type,
                confidence_score=result.get('confidence', 0.8),
                changes_made=result.get('changes', []),
                processing_time=processing_time,
                metadata=result.get('metadata', {})
            )
            
        except Exception as e:
            logger.error(f"Enhancement {enhancement_type.value} failed: {str(e)}")
            raise
    
    async def _enhance_grammar(self, text: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance grammar and sentence structure."""
        try:
            response = await self.llm_service.enhance_arabic_text(
                text=text,
                enhancement_type="grammar"
            )
            
            changes = self._detect_changes(text, response.text)
            
            return {
                'text': response.text,
                'confidence': 0.85,
                'changes': changes,
                'metadata': {
                    'tokens_used': response.tokens_used,
                    'model': response.model,
                    'original_length': len(text),
                    'enhanced_length': len(response.text)
                }
            }
        except Exception as e:
            logger.error(f"Grammar enhancement failed: {str(e)}")
            return {'text': text, 'confidence': 0.0, 'changes': [], 'metadata': {}}
    
    async def _add_diacritics(self, text: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Add Arabic diacritical marks."""
        try:
            response = await self.llm_service.enhance_arabic_text(
                text=text,
                enhancement_type="diacritization"
            )
            
            # Check if diacritics were added
            diacritics_added = len(self.arabic_patterns['diacritics'].findall(response.text)) > \
                              len(self.arabic_patterns['diacritics'].findall(text))
            
            confidence = 0.9 if diacritics_added else 0.5
            
            return {
                'text': response.text,
                'confidence': confidence,
                'changes': ['Added diacritical marks'] if diacritics_added else [],
                'metadata': {
                    'tokens_used': response.tokens_used,
                    'model': response.model,
                    'diacritics_added': diacritics_added
                }
            }
        except Exception as e:
            logger.error(f"Diacritization failed: {str(e)}")
            return {'text': text, 'confidence': 0.0, 'changes': [], 'metadata': {}}
    
    async def _normalize_text(self, text: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize dialectal Arabic to Modern Standard Arabic."""
        try:
            response = await self.llm_service.enhance_arabic_text(
                text=text,
                enhancement_type="normalization"
            )
            
            changes = self._detect_changes(text, response.text)
            
            return {
                'text': response.text,
                'confidence': 0.8,
                'changes': changes,
                'metadata': {
                    'tokens_used': response.tokens_used,
                    'model': response.model,
                    'normalization_applied': len(changes) > 0
                }
            }
        except Exception as e:
            logger.error(f"Text normalization failed: {str(e)}")
            return {'text': text, 'confidence': 0.0, 'changes': [], 'metadata': {}}
    
    async def _fix_punctuation(self, text: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Fix and improve punctuation."""
        try:
            # Create punctuation enhancement prompt
            system_prompt = """You are an expert in Arabic punctuation and formatting. 
            Fix and improve the punctuation in the given text. Add missing punctuation marks, 
            fix incorrect usage, and ensure proper spacing around punctuation marks."""
            
            prompt = f"Fix the punctuation in this Arabic text:\n\n{text}"
            
            response = await self.llm_service.generate_text(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.2
            )
            
            changes = self._detect_punctuation_changes(text, response.text)
            
            return {
                'text': response.text,
                'confidence': 0.9,
                'changes': changes,
                'metadata': {
                    'tokens_used': response.tokens_used,
                    'model': response.model,
                    'punctuation_fixed': len(changes) > 0
                }
            }
        except Exception as e:
            logger.error(f"Punctuation fixing failed: {str(e)}")
            return {'text': text, 'confidence': 0.0, 'changes': [], 'metadata': {}}
    
    async def _fix_spelling(self, text: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Fix spelling errors."""
        try:
            system_prompt = """You are an expert in Arabic spelling and orthography. 
            Correct any spelling errors in the given text while preserving the original meaning 
            and style. Focus on common spelling mistakes and typos."""
            
            prompt = f"Correct spelling errors in this Arabic text:\n\n{text}"
            
            response = await self.llm_service.generate_text(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.1
            )
            
            changes = self._detect_changes(text, response.text)
            
            return {
                'text': response.text,
                'confidence': 0.85,
                'changes': changes,
                'metadata': {
                    'tokens_used': response.tokens_used,
                    'model': response.model,
                    'spelling_corrections': len(changes)
                }
            }
        except Exception as e:
            logger.error(f"Spelling correction failed: {str(e)}")
            return {'text': text, 'confidence': 0.0, 'changes': [], 'metadata': {}}
    
    async def _improve_style(self, text: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Improve text style and readability."""
        try:
            system_prompt = """You are an expert Arabic language editor. 
            Improve the style, flow, and readability of the given text while preserving 
            its original meaning. Make it more eloquent and professional."""
            
            prompt = f"Improve the style and readability of this Arabic text:\n\n{text}"
            
            response = await self.llm_service.generate_text(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.4
            )
            
            changes = self._detect_changes(text, response.text)
            
            return {
                'text': response.text,
                'confidence': 0.8,
                'changes': changes,
                'metadata': {
                    'tokens_used': response.tokens_used,
                    'model': response.model,
                    'style_improvements': len(changes)
                }
            }
        except Exception as e:
            logger.error(f"Style improvement failed: {str(e)}")
            return {'text': text, 'confidence': 0.0, 'changes': [], 'metadata': {}}
    
    def _detect_changes(self, original: str, enhanced: str) -> List[str]:
        """Detect changes between original and enhanced text."""
        changes = []
        
        # Basic change detection
        if len(enhanced) != len(original):
            if len(enhanced) > len(original):
                changes.append("Text expanded with additional content")
            else:
                changes.append("Text condensed or simplified")
        
        # Word-level changes
        original_words = original.split()
        enhanced_words = enhanced.split()
        
        if len(enhanced_words) != len(original_words):
            changes.append(f"Word count changed from {len(original_words)} to {len(enhanced_words)}")
        
        # Check for specific improvements
        original_sentences = original.count('.') + original.count('؟') + original.count('!')
        enhanced_sentences = enhanced.count('.') + enhanced.count('؟') + enhanced.count('!')
        
        if enhanced_sentences > original_sentences:
            changes.append("Improved sentence structure")
        
        return changes
    
    def _detect_punctuation_changes(self, original: str, enhanced: str) -> List[str]:
        """Detect punctuation-specific changes."""
        changes = []
        
        # Count punctuation marks
        original_punct = len(self.arabic_patterns['punctuation'].findall(original))
        enhanced_punct = len(self.arabic_patterns['punctuation'].findall(enhanced))
        
        if enhanced_punct > original_punct:
            changes.append(f"Added {enhanced_punct - original_punct} punctuation marks")
        elif enhanced_punct < original_punct:
            changes.append(f"Removed {original_punct - enhanced_punct} punctuation marks")
        
        # Check for specific improvements
        if '،' in enhanced and '،' not in original:
            changes.append("Added Arabic commas")
        if '؟' in enhanced and '؟' not in original:
            changes.append("Added Arabic question marks")
        
        return changes
    
    async def batch_enhance(
        self,
        texts: List[str],
        enhancement_types: List[EnhancementType] = None,
        options: Dict[str, Any] = None
    ) -> List[Dict[str, EnhancementResult]]:
        """Enhance multiple texts in batch."""
        try:
            results = []
            
            for i, text in enumerate(texts):
                logger.info(f"Processing text {i+1}/{len(texts)}")
                result = await self.enhance_text(text, enhancement_types, options)
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Batch enhancement failed: {str(e)}")
            raise
    
    def get_enhancement_quality_metrics(
        self,
        results: Dict[str, EnhancementResult]
    ) -> Dict[str, Any]:
        """Calculate quality metrics for enhancement results."""
        metrics = {
            'total_enhancements': len(results),
            'average_confidence': 0.0,
            'total_processing_time': 0.0,
            'changes_summary': {},
            'enhancement_breakdown': {}
        }
        
        if not results:
            return metrics
        
        total_confidence = 0
        total_changes = 0
        
        for enhancement_type, result in results.items():
            metrics['enhancement_breakdown'][enhancement_type] = {
                'confidence': result.confidence_score,
                'changes_count': len(result.changes_made),
                'processing_time': result.processing_time
            }
            
            total_confidence += result.confidence_score
            total_changes += len(result.changes_made)
            metrics['total_processing_time'] += result.processing_time
        
        metrics['average_confidence'] = total_confidence / len(results)
        metrics['total_changes'] = total_changes
        
        return metrics
