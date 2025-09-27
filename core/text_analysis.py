"""
Text Analysis Engine for Arabic STT Platform
Provides comprehensive text analysis capabilities using LLM
"""

import asyncio
import logging
import json
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import time

from core.llm_service import LLMService

logger = logging.getLogger(__name__)

class AnalysisType(Enum):
    """Types of text analysis."""
    SENTIMENT = "sentiment"
    EMOTION = "emotion"
    TOPICS = "topics"
    ENTITIES = "entities"
    KEYWORDS = "keywords"
    LANGUAGE_DETECTION = "language"
    COMPLEXITY = "complexity"
    READABILITY = "readability"

@dataclass
class SentimentResult:
    """Sentiment analysis result."""
    score: float  # -1 to 1
    label: str    # positive, negative, neutral
    confidence: float
    emotions: List[str]

@dataclass
class EntityResult:
    """Named entity recognition result."""
    persons: List[str]
    locations: List[str]
    organizations: List[str]
    dates: List[str]
    other: List[str]

@dataclass
class TopicResult:
    """Topic modeling result."""
    main_topics: List[str]
    topic_scores: Dict[str, float]
    categories: List[str]

@dataclass
class ComplexityResult:
    """Text complexity analysis result."""
    readability_score: float
    complexity_level: str
    sentence_count: int
    word_count: int
    average_sentence_length: float
    vocabulary_richness: float

@dataclass
class AnalysisReport:
    """Complete text analysis report."""
    text: str
    sentiment: Optional[SentimentResult]
    entities: Optional[EntityResult]
    topics: Optional[TopicResult]
    complexity: Optional[ComplexityResult]
    keywords: List[str]
    language: str
    processing_time: float
    metadata: Dict[str, Any]

class TextAnalysisEngine:
    """Engine for comprehensive Arabic text analysis using LLM."""
    
    def __init__(self):
        self.llm_service = LLMService()
        self.initialized = False
        
        # Arabic language patterns
        self.arabic_patterns = {
            'sentences': re.compile(r'[.!?؟]+'),
            'words': re.compile(r'\b[\u0600-\u06FF]+\b'),
            'punctuation': re.compile(r'[،؛؟!.]+'),
        }
        
        # Complexity level thresholds
        self.complexity_thresholds = {
            'simple': (0, 3),
            'moderate': (3, 6),
            'complex': (6, 10),
            'very_complex': (10, float('inf'))
        }
    
    async def initialize(self):
        """Initialize the text analysis engine."""
        try:
            await self.llm_service.initialize()
            self.initialized = True
            logger.info("Text Analysis Engine initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Text Analysis Engine: {str(e)}")
            return False
    
    async def cleanup(self):
        """Cleanup engine resources."""
        if self.llm_service:
            await self.llm_service.cleanup()
        self.initialized = False
    
    async def analyze_text(
        self,
        text: str,
        analysis_types: List[AnalysisType] = None,
        options: Dict[str, Any] = None
    ) -> AnalysisReport:
        """
        Perform comprehensive text analysis.
        
        Args:
            text: Text to analyze
            analysis_types: List of analysis types to perform
            options: Additional analysis options
            
        Returns:
            Complete analysis report
        """
        if not self.initialized:
            raise RuntimeError("Text Analysis Engine not initialized")
        
        if analysis_types is None:
            analysis_types = [
                AnalysisType.SENTIMENT,
                AnalysisType.ENTITIES,
                AnalysisType.TOPICS,
                AnalysisType.KEYWORDS,
                AnalysisType.COMPLEXITY
            ]
        
        start_time = time.time()
        
        try:
            # Initialize results
            sentiment = None
            entities = None
            topics = None
            complexity = None
            keywords = []
            language = "ar"  # Default to Arabic
            
            # Perform requested analyses
            for analysis_type in analysis_types:
                logger.info(f"Performing {analysis_type.value} analysis")
                
                if analysis_type == AnalysisType.SENTIMENT:
                    sentiment = await self._analyze_sentiment(text, options or {})
                elif analysis_type == AnalysisType.ENTITIES:
                    entities = await self._extract_entities(text, options or {})
                elif analysis_type == AnalysisType.TOPICS:
                    topics = await self._analyze_topics(text, options or {})
                elif analysis_type == AnalysisType.KEYWORDS:
                    keywords = await self._extract_keywords(text, options or {})
                elif analysis_type == AnalysisType.COMPLEXITY:
                    complexity = await self._analyze_complexity(text, options or {})
                elif analysis_type == AnalysisType.LANGUAGE_DETECTION:
                    language = await self._detect_language(text, options or {})
            
            processing_time = time.time() - start_time
            
            return AnalysisReport(
                text=text,
                sentiment=sentiment,
                entities=entities,
                topics=topics,
                complexity=complexity,
                keywords=keywords,
                language=language,
                processing_time=processing_time,
                metadata={
                    'analysis_types': [t.value for t in analysis_types],
                    'text_length': len(text),
                    'word_count': len(text.split()),
                    'character_count': len(text)
                }
            )
            
        except Exception as e:
            logger.error(f"Text analysis failed: {str(e)}")
            raise
    
    async def _analyze_sentiment(self, text: str, options: Dict[str, Any]) -> SentimentResult:
        """Analyze sentiment and emotions."""
        try:
            sentiment_data = await self.llm_service.analyze_sentiment(text)
            
            return SentimentResult(
                score=sentiment_data.get('score', 0.0),
                label=sentiment_data.get('label', 'neutral'),
                confidence=sentiment_data.get('confidence', 0.5),
                emotions=sentiment_data.get('emotions', [])
            )
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {str(e)}")
            return SentimentResult(
                score=0.0,
                label='neutral',
                confidence=0.0,
                emotions=[]
            )
    
    async def _extract_entities(self, text: str, options: Dict[str, Any]) -> EntityResult:
        """Extract named entities."""
        try:
            entities_data = await self.llm_service.extract_entities(text)
            
            return EntityResult(
                persons=entities_data.get('persons', []),
                locations=entities_data.get('locations', []),
                organizations=entities_data.get('organizations', []),
                dates=entities_data.get('dates', []),
                other=entities_data.get('other', [])
            )
            
        except Exception as e:
            logger.error(f"Entity extraction failed: {str(e)}")
            return EntityResult(
                persons=[],
                locations=[],
                organizations=[],
                dates=[],
                other=[]
            )
    
    async def _analyze_topics(self, text: str, options: Dict[str, Any]) -> TopicResult:
        """Analyze topics and categories."""
        try:
            system_prompt = """You are an expert in topic modeling and text categorization.
            Analyze the given text and respond with a JSON object containing:
            - main_topics: list of 3-5 main topics
            - topic_scores: dictionary with topic names as keys and relevance scores (0-1) as values
            - categories: list of general categories this text belongs to
            
            Respond only with valid JSON, no additional text."""
            
            prompt = f"Analyze the topics in this Arabic text:\n\n{text}"
            
            response = await self.llm_service.generate_text(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.3
            )
            
            try:
                topics_data = json.loads(response.text)
                return TopicResult(
                    main_topics=topics_data.get('main_topics', []),
                    topic_scores=topics_data.get('topic_scores', {}),
                    categories=topics_data.get('categories', [])
                )
            except json.JSONDecodeError:
                return TopicResult(
                    main_topics=[],
                    topic_scores={},
                    categories=[]
                )
                
        except Exception as e:
            logger.error(f"Topic analysis failed: {str(e)}")
            return TopicResult(
                main_topics=[],
                topic_scores={},
                categories=[]
            )
    
    async def _extract_keywords(self, text: str, options: Dict[str, Any]) -> List[str]:
        """Extract important keywords and phrases."""
        try:
            system_prompt = """You are an expert in keyword extraction.
            Extract the most important keywords and key phrases from the given text.
            Return a JSON array of strings containing 10-15 most relevant keywords.
            
            Respond only with a valid JSON array, no additional text."""
            
            prompt = f"Extract keywords from this Arabic text:\n\n{text}"
            
            response = await self.llm_service.generate_text(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.2
            )
            
            try:
                keywords = json.loads(response.text)
                if isinstance(keywords, list):
                    return keywords[:15]  # Limit to 15 keywords
                return []
            except json.JSONDecodeError:
                return []
                
        except Exception as e:
            logger.error(f"Keyword extraction failed: {str(e)}")
            return []
    
    async def _analyze_complexity(self, text: str, options: Dict[str, Any]) -> ComplexityResult:
        """Analyze text complexity and readability."""
        try:
            # Basic metrics calculation
            sentences = self.arabic_patterns['sentences'].split(text)
            sentences = [s.strip() for s in sentences if s.strip()]
            sentence_count = len(sentences)
            
            words = self.arabic_patterns['words'].findall(text)
            word_count = len(words)
            
            # Calculate basic metrics
            avg_sentence_length = word_count / max(sentence_count, 1)
            
            # Vocabulary richness (unique words / total words)
            unique_words = len(set(words))
            vocab_richness = unique_words / max(word_count, 1)
            
            # Use LLM for advanced complexity analysis
            system_prompt = """You are an expert in Arabic text complexity analysis.
            Analyze the complexity and readability of the given text and respond with a JSON object:
            - readability_score: float (0-10, where 0 is very easy and 10 is very difficult)
            - complexity_factors: list of factors that make the text complex or simple
            
            Respond only with valid JSON, no additional text."""
            
            prompt = f"Analyze the complexity of this Arabic text:\n\n{text}"
            
            response = await self.llm_service.generate_text(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.1
            )
            
            llm_analysis = {}
            try:
                llm_analysis = json.loads(response.text)
            except json.JSONDecodeError:
                pass
            
            readability_score = llm_analysis.get('readability_score', avg_sentence_length / 2)
            
            # Determine complexity level
            complexity_level = 'simple'
            for level, (min_val, max_val) in self.complexity_thresholds.items():
                if min_val <= readability_score < max_val:
                    complexity_level = level
                    break
            
            return ComplexityResult(
                readability_score=readability_score,
                complexity_level=complexity_level,
                sentence_count=sentence_count,
                word_count=word_count,
                average_sentence_length=avg_sentence_length,
                vocabulary_richness=vocab_richness
            )
            
        except Exception as e:
            logger.error(f"Complexity analysis failed: {str(e)}")
            # Return basic metrics on failure
            return ComplexityResult(
                readability_score=5.0,
                complexity_level='moderate',
                sentence_count=len(text.split('.')),
                word_count=len(text.split()),
                average_sentence_length=len(text.split()) / max(len(text.split('.')), 1),
                vocabulary_richness=0.5
            )
    
    async def _detect_language(self, text: str, options: Dict[str, Any]) -> str:
        """Detect the language of the text."""
        try:
            # Check for Arabic characters
            arabic_chars = len(self.arabic_patterns['words'].findall(text))
            total_chars = len(text.replace(' ', ''))
            
            if total_chars > 0:
                arabic_ratio = arabic_chars / total_chars
                if arabic_ratio > 0.7:
                    return 'ar'
                elif arabic_ratio > 0.3:
                    return 'ar-mixed'
                else:
                    return 'other'
            
            return 'unknown'
            
        except Exception as e:
            logger.error(f"Language detection failed: {str(e)}")
            return 'unknown'
    
    async def batch_analyze(
        self,
        texts: List[str],
        analysis_types: List[AnalysisType] = None,
        options: Dict[str, Any] = None
    ) -> List[AnalysisReport]:
        """Analyze multiple texts in batch."""
        try:
            results = []
            
            for i, text in enumerate(texts):
                logger.info(f"Analyzing text {i+1}/{len(texts)}")
                result = await self.analyze_text(text, analysis_types, options)
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Batch analysis failed: {str(e)}")
            raise
    
    def generate_analysis_summary(self, report: AnalysisReport) -> Dict[str, Any]:
        """Generate a summary of analysis results."""
        summary = {
            'text_stats': {
                'word_count': len(report.text.split()),
                'character_count': len(report.text),
                'language': report.language,
                'processing_time': report.processing_time
            }
        }
        
        if report.sentiment:
            summary['sentiment'] = {
                'label': report.sentiment.label,
                'score': report.sentiment.score,
                'confidence': report.sentiment.confidence,
                'has_emotions': len(report.sentiment.emotions) > 0
            }
        
        if report.entities:
            entity_counts = {
                'persons': len(report.entities.persons),
                'locations': len(report.entities.locations),
                'organizations': len(report.entities.organizations),
                'dates': len(report.entities.dates),
                'other': len(report.entities.other)
            }
            summary['entities'] = {
                'total_entities': sum(entity_counts.values()),
                'breakdown': entity_counts
            }
        
        if report.topics:
            summary['topics'] = {
                'topic_count': len(report.topics.main_topics),
                'has_categories': len(report.topics.categories) > 0,
                'main_topics': report.topics.main_topics[:3]  # Top 3 topics
            }
        
        if report.complexity:
            summary['complexity'] = {
                'level': report.complexity.complexity_level,
                'readability_score': report.complexity.readability_score,
                'avg_sentence_length': report.complexity.average_sentence_length
            }
        
        summary['keywords_count'] = len(report.keywords)
        
        return summary
    
    def export_report(self, report: AnalysisReport, format: str = 'json') -> str:
        """Export analysis report in specified format."""
        if format.lower() == 'json':
            # Convert dataclasses to dict for JSON serialization
            report_dict = {
                'text': report.text,
                'sentiment': asdict(report.sentiment) if report.sentiment else None,
                'entities': asdict(report.entities) if report.entities else None,
                'topics': asdict(report.topics) if report.topics else None,
                'complexity': asdict(report.complexity) if report.complexity else None,
                'keywords': report.keywords,
                'language': report.language,
                'processing_time': report.processing_time,
                'metadata': report.metadata
            }
            return json.dumps(report_dict, ensure_ascii=False, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    async def compare_texts(
        self,
        text1: str,
        text2: str,
        comparison_aspects: List[str] = None
    ) -> Dict[str, Any]:
        """Compare two texts across various dimensions."""
        if comparison_aspects is None:
            comparison_aspects = ['sentiment', 'topics', 'complexity', 'style']
        
        try:
            # Analyze both texts
            report1 = await self.analyze_text(text1)
            report2 = await self.analyze_text(text2)
            
            comparison = {
                'text1_summary': self.generate_analysis_summary(report1),
                'text2_summary': self.generate_analysis_summary(report2),
                'differences': {},
                'similarities': []
            }
            
            # Compare sentiment
            if 'sentiment' in comparison_aspects and report1.sentiment and report2.sentiment:
                sentiment_diff = abs(report1.sentiment.score - report2.sentiment.score)
                comparison['differences']['sentiment'] = {
                    'text1_sentiment': report1.sentiment.label,
                    'text2_sentiment': report2.sentiment.label,
                    'score_difference': sentiment_diff,
                    'similar': sentiment_diff < 0.3
                }
            
            # Compare complexity
            if 'complexity' in comparison_aspects and report1.complexity and report2.complexity:
                complexity_diff = abs(report1.complexity.readability_score - report2.complexity.readability_score)
                comparison['differences']['complexity'] = {
                    'text1_level': report1.complexity.complexity_level,
                    'text2_level': report2.complexity.complexity_level,
                    'score_difference': complexity_diff,
                    'similar': complexity_diff < 1.0
                }
            
            # Find common topics
            if 'topics' in comparison_aspects and report1.topics and report2.topics:
                common_topics = set(report1.topics.main_topics) & set(report2.topics.main_topics)
                comparison['similarities'].extend([f"Common topic: {topic}" for topic in common_topics])
            
            # Find common entities
            if report1.entities and report2.entities:
                common_persons = set(report1.entities.persons) & set(report2.entities.persons)
                common_locations = set(report1.entities.locations) & set(report2.entities.locations)
                comparison['similarities'].extend([f"Common person: {person}" for person in common_persons])
                comparison['similarities'].extend([f"Common location: {location}" for location in common_locations])
            
            return comparison
            
        except Exception as e:
            logger.error(f"Text comparison failed: {str(e)}")
            raise
