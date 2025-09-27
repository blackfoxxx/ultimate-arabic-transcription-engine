"""
Enhanced Timestamp Processor for Arabic STT Platform
Provides detailed timestamp support with sentence and word-level precision
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import timedelta
import statistics

logger = logging.getLogger(__name__)

class EnhancedTimestampProcessor:
    """
    Advanced timestamp processing for enhanced transcript output.
    Provides sentence-level, paragraph-level, and word-level timestamp alignment.
    """
    
    def __init__(self):
        self.sentence_boundary_patterns = [
            r'[.!?]+\s+',  # Standard sentence endings
            r'[.!?]+$',    # Sentence ending at end of text
            r'[،؛]\s+',    # Arabic comma and semicolon
            r'[.!?؟]+\s+'  # Arabic question mark
        ]
        
        # Arabic-specific patterns
        self.arabic_sentence_markers = [
            '؟',  # Arabic question mark
            '!',  # Exclamation
            '.',  # Period
            '؛',  # Arabic semicolon
        ]
    
    def enhance_timestamps(self, transcript_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance transcript data with detailed timestamp information.
        
        Args:
            transcript_data: Original transcript data with segments
            
        Returns:
            Enhanced transcript data with sentence and paragraph timestamps
        """
        try:
            logger.info("Enhancing timestamps for transcript")
            
            # Create enhanced copy of transcript data
            enhanced_data = transcript_data.copy()
            
            # Process segments for enhanced timestamps
            segments = transcript_data.get('segments', [])
            if not segments:
                logger.warning("No segments found in transcript data")
                return enhanced_data
            
            # Generate sentence-level timestamps
            sentence_timestamps = self._generate_sentence_timestamps(segments)
            enhanced_data['sentence_timestamps'] = sentence_timestamps
            
            # Generate paragraph-level timestamps
            paragraph_timestamps = self._generate_paragraph_timestamps(sentence_timestamps)
            enhanced_data['paragraph_timestamps'] = paragraph_timestamps
            
            # Generate enhanced segments with word alignment
            enhanced_segments = self._enhance_segment_word_alignment(segments)
            enhanced_data['enhanced_segments'] = enhanced_segments
            
            # Generate timestamp statistics
            timestamp_stats = self._calculate_timestamp_statistics(segments, sentence_timestamps)
            enhanced_data['timestamp_statistics'] = timestamp_stats
            
            logger.info(f"Generated {len(sentence_timestamps)} sentence timestamps")
            logger.info(f"Generated {len(paragraph_timestamps)} paragraph timestamps")
            
            return enhanced_data
            
        except Exception as e:
            logger.error(f"Timestamp enhancement failed: {str(e)}")
            # Return original data if enhancement fails
            return transcript_data
    
    def _generate_sentence_timestamps(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate sentence-level timestamps from segments."""
        sentence_timestamps = []
        sentence_id = 1
        
        for segment in segments:
            segment_text = segment.get('text', '').strip()
            segment_start = segment.get('start', 0)
            segment_end = segment.get('end', 0)
            segment_words = segment.get('words', [])
            
            if not segment_text:
                continue
            
            # Split segment into sentences
            sentences = self._split_into_sentences(segment_text)
            
            if len(sentences) <= 1:
                # Single sentence in segment
                sentence_timestamps.append({
                    'sentence_id': sentence_id,
                    'text': segment_text,
                    'start': segment_start,
                    'end': segment_end,
                    'duration': segment_end - segment_start,
                    'word_count': len(segment_text.split()),
                    'confidence': segment.get('avg_logprob', 0),
                    'words': segment_words
                })
                sentence_id += 1
            else:
                # Multiple sentences in segment - need to distribute timestamps
                sentence_positions = self._calculate_sentence_positions(
                    segment_text, sentences, segment_words
                )
                
                for i, sentence in enumerate(sentences):
                    if sentence.strip():
                        # Calculate sentence timing based on word positions or text length
                        sentence_timing = self._estimate_sentence_timing(
                            sentence, segment_start, segment_end, 
                            sentence_positions[i], len(sentences), i
                        )
                        
                        sentence_timestamps.append({
                            'sentence_id': sentence_id,
                            'text': sentence.strip(),
                            'start': sentence_timing['start'],
                            'end': sentence_timing['end'],
                            'duration': sentence_timing['end'] - sentence_timing['start'],
                            'word_count': len(sentence.strip().split()),
                            'confidence': segment.get('avg_logprob', 0),
                            'words': sentence_timing.get('words', []),
                            'estimated': True  # Mark as estimated timing
                        })
                        sentence_id += 1
        
        return sentence_timestamps
    
    def _generate_paragraph_timestamps(self, sentence_timestamps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate paragraph-level timestamps from sentences."""
        if not sentence_timestamps:
            return []
        
        paragraphs = []
        current_paragraph = {
            'paragraph_id': 1,
            'sentences': [],
            'text': '',
            'start': None,
            'end': None,
            'word_count': 0,
            'sentence_count': 0
        }
        
        sentences_per_paragraph = 3  # Default sentences per paragraph
        
        for sentence in sentence_timestamps:
            # Add sentence to current paragraph
            current_paragraph['sentences'].append(sentence)
            current_paragraph['text'] += sentence['text'] + ' '
            current_paragraph['word_count'] += sentence['word_count']
            current_paragraph['sentence_count'] += 1
            
            # Set paragraph start/end times
            if current_paragraph['start'] is None:
                current_paragraph['start'] = sentence['start']
            current_paragraph['end'] = sentence['end']
            
            # Check if paragraph should end
            should_end_paragraph = (
                current_paragraph['sentence_count'] >= sentences_per_paragraph or
                current_paragraph['word_count'] > 100 or  # Long paragraph
                sentence['text'].strip().endswith(('؟', '!', '.'))  # Strong sentence ending
            )
            
            if should_end_paragraph:
                # Finalize current paragraph
                current_paragraph['text'] = current_paragraph['text'].strip()
                current_paragraph['duration'] = current_paragraph['end'] - current_paragraph['start']
                paragraphs.append(current_paragraph.copy())
                
                # Start new paragraph
                current_paragraph = {
                    'paragraph_id': len(paragraphs) + 1,
                    'sentences': [],
                    'text': '',
                    'start': None,
                    'end': None,
                    'word_count': 0,
                    'sentence_count': 0
                }
        
        # Add final paragraph if it has content
        if current_paragraph['sentences']:
            current_paragraph['text'] = current_paragraph['text'].strip()
            current_paragraph['duration'] = current_paragraph['end'] - current_paragraph['start']
            paragraphs.append(current_paragraph)
        
        return paragraphs
    
    def _enhance_segment_word_alignment(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enhance segments with better word alignment."""
        enhanced_segments = []
        
        for segment in segments:
            enhanced_segment = segment.copy()
            words = segment.get('words', [])
            
            if words:
                # Calculate word timing statistics
                word_durations = []
                for word in words:
                    duration = word.get('end', 0) - word.get('start', 0)
                    if duration > 0:
                        word_durations.append(duration)
                
                if word_durations:
                    enhanced_segment['word_timing_stats'] = {
                        'average_word_duration': statistics.mean(word_durations),
                        'median_word_duration': statistics.median(word_durations),
                        'total_speech_time': sum(word_durations),
                        'words_per_second': len(words) / (segment['end'] - segment['start'])
                    }
                
                # Identify long pauses between words
                pauses = []
                for i in range(len(words) - 1):
                    current_end = words[i].get('end', 0)
                    next_start = words[i + 1].get('start', 0)
                    pause_duration = next_start - current_end
                    
                    if pause_duration > 0.5:  # Pause longer than 500ms
                        pauses.append({
                            'position': i,
                            'duration': pause_duration,
                            'before_word': words[i].get('word', ''),
                            'after_word': words[i + 1].get('word', '')
                        })
                
                enhanced_segment['pauses'] = pauses
            
            enhanced_segments.append(enhanced_segment)
        
        return enhanced_segments
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using Arabic-aware patterns."""
        # Combine all sentence boundary patterns
        pattern = '|'.join(self.sentence_boundary_patterns)
        
        # Split but keep delimiters
        parts = re.split(f'({pattern})', text)
        
        sentences = []
        current_sentence = ""
        
        for part in parts:
            current_sentence += part
            
            # Check if this part ends a sentence
            if re.match(pattern, part):
                sentences.append(current_sentence.strip())
                current_sentence = ""
        
        # Add remaining text as final sentence
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
        
        return [s for s in sentences if s.strip()]
    
    def _calculate_sentence_positions(
        self, 
        segment_text: str, 
        sentences: List[str], 
        segment_words: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Calculate relative positions of sentences within segment."""
        positions = []
        text_position = 0
        
        for sentence in sentences:
            sentence_start_pos = segment_text.find(sentence.strip(), text_position)
            sentence_end_pos = sentence_start_pos + len(sentence.strip())
            
            # Find corresponding words for this sentence
            sentence_words = []
            if segment_words:
                sentence_words = self._find_words_for_text_range(
                    segment_words, sentence.strip()
                )
            
            positions.append({
                'start_char': sentence_start_pos,
                'end_char': sentence_end_pos,
                'relative_start': sentence_start_pos / len(segment_text),
                'relative_end': sentence_end_pos / len(segment_text),
                'words': sentence_words
            })
            
            text_position = sentence_end_pos
        
        return positions
    
    def _estimate_sentence_timing(
        self,
        sentence: str,
        segment_start: float,
        segment_end: float,
        position_info: Dict[str, Any],
        total_sentences: int,
        sentence_index: int
    ) -> Dict[str, Any]:
        """Estimate timing for a sentence within a segment."""
        segment_duration = segment_end - segment_start
        
        # Use word-level timing if available
        words = position_info.get('words', [])
        if words and len(words) > 0:
            # Use actual word timestamps
            word_starts = [w.get('start', 0) for w in words if 'start' in w]
            word_ends = [w.get('end', 0) for w in words if 'end' in w]
            
            if word_starts and word_ends:
                return {
                    'start': min(word_starts),
                    'end': max(word_ends),
                    'words': words
                }
        
        # Fallback to proportional timing
        relative_start = position_info.get('relative_start', sentence_index / total_sentences)
        relative_end = position_info.get('relative_end', (sentence_index + 1) / total_sentences)
        
        estimated_start = segment_start + (segment_duration * relative_start)
        estimated_end = segment_start + (segment_duration * relative_end)
        
        return {
            'start': estimated_start,
            'end': estimated_end,
            'words': []
        }
    
    def _find_words_for_text_range(self, segment_words: List[Dict[str, Any]], sentence_text: str) -> List[Dict[str, Any]]:
        """Find words that correspond to a sentence within a segment."""
        sentence_words = sentence_text.strip().split()
        matching_words = []
        
        # Simple word matching approach
        word_index = 0
        for segment_word in segment_words:
            word_text = segment_word.get('word', '').strip()
            
            if word_index < len(sentence_words):
                sentence_word = sentence_words[word_index].strip()
                
                # Check if words match (allowing for some variation)
                if (word_text.lower() == sentence_word.lower() or 
                    word_text.lower() in sentence_word.lower() or
                    sentence_word.lower() in word_text.lower()):
                    matching_words.append(segment_word)
                    word_index += 1
        
        return matching_words
    
    def _calculate_timestamp_statistics(
        self, 
        segments: List[Dict[str, Any]], 
        sentence_timestamps: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate statistics about timestamp distribution."""
        try:
            segment_durations = [s['end'] - s['start'] for s in segments if 'start' in s and 'end' in s]
            sentence_durations = [s['duration'] for s in sentence_timestamps if 'duration' in s]
            
            stats = {
                'total_segments': len(segments),
                'total_sentences': len(sentence_timestamps),
                'average_segment_duration': statistics.mean(segment_durations) if segment_durations else 0,
                'average_sentence_duration': statistics.mean(sentence_durations) if sentence_durations else 0,
                'sentences_per_segment': len(sentence_timestamps) / len(segments) if segments else 0,
                'estimated_sentences': sum(1 for s in sentence_timestamps if s.get('estimated', False))
            }
            
            if segment_durations:
                stats.update({
                    'min_segment_duration': min(segment_durations),
                    'max_segment_duration': max(segment_durations),
                    'median_segment_duration': statistics.median(segment_durations)
                })
            
            if sentence_durations:
                stats.update({
                    'min_sentence_duration': min(sentence_durations),
                    'max_sentence_duration': max(sentence_durations),
                    'median_sentence_duration': statistics.median(sentence_durations)
                })
            
            return stats
            
        except Exception as e:
            logger.error(f"Statistics calculation failed: {str(e)}")
            return {'error': str(e)}
    
    def format_timestamp_for_display(self, seconds: float, format_type: str = 'standard') -> str:
        """Format timestamp for different display purposes."""
        if format_type == 'srt':
            # SRT format: HH:MM:SS,mmm
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            milliseconds = int((seconds % 1) * 1000)
            return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"
        
        elif format_type == 'vtt':
            # WebVTT format: MM:SS.mmm
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes:02d}:{secs:06.3f}"
        
        elif format_type == 'human':
            # Human-readable format
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            
            if hours > 0:
                return f"{hours}h {minutes}m {secs}s"
            elif minutes > 0:
                return f"{minutes}m {secs}s"
            else:
                return f"{secs}s"
        
        else:  # standard
            # Standard format: MM:SS
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes:02d}:{secs:02d}"
