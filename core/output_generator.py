"""
Output generation module for Arabic STT Platform
Generates multiple output formats from transcription data with enhanced timestamps
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List
from pathlib import Path
import re

from config import Config
from core.enhanced_timestamp_processor import EnhancedTimestampProcessor

logger = logging.getLogger(__name__)

class OutputGenerator:
    """Generates various output formats from transcription data with enhanced timestamp support."""
    
    def __init__(self):
        self.config = Config()
        self.timestamp_processor = EnhancedTimestampProcessor()
    
    async def generate_output(
        self,
        transcript_data: Dict[str, Any],
        job_id: str,
        output_format: str,
        enhance_timestamps: bool = True,
        generate_separate_analysis: bool = True
    ) -> str:
        """
        Generate output file in specified format with enhanced timestamps.
        
        Args:
            transcript_data: Transcription result data
            job_id: Unique job identifier
            output_format: Output format (txt, srt, vtt, json)
            enhance_timestamps: Whether to use enhanced timestamp processing
            generate_separate_analysis: Whether to generate separate analysis files
            
        Returns:
            Path to generated output file
        """
        try:
            # Enhance timestamps if requested and segments are available
            if enhance_timestamps and transcript_data.get('segments'):
                logger.info("Enhancing timestamps for output generation")
                transcript_data = self.timestamp_processor.enhance_timestamps(transcript_data)
            
            output_path = self.config.RESULTS_FOLDER / f"{job_id}.{output_format}"
            
            if output_format == 'txt':
                await self._generate_txt(transcript_data, output_path)
            elif output_format == 'srt':
                await self._generate_srt(transcript_data, output_path)
            elif output_format == 'vtt':
                await self._generate_vtt(transcript_data, output_path)
            elif output_format == 'json':
                await self._generate_json(transcript_data, output_path)
            else:
                raise ValueError(f"Unsupported output format: {output_format}")
            
            # Generate separate analysis files if requested
            if generate_separate_analysis:
                await self._generate_separate_analysis_files(transcript_data, job_id)
            
            logger.info(f"Generated {output_format} output with enhanced timestamps: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Output generation failed for {output_format}: {str(e)}")
            raise
    
    async def _generate_txt(self, transcript_data: Dict[str, Any], output_path: Path) -> None:
        """Generate plain text transcript."""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                # Write header
                f.write("Arabic Speech-to-Text Transcript\n")
                f.write("=" * 40 + "\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Duration: {transcript_data.get('duration', 0):.2f} seconds\n")
                f.write(f"Language: {transcript_data.get('language', 'Unknown')}\n")
                f.write(f"Model: {transcript_data.get('model_size', 'Unknown')}\n")
                
                # Check if multi-speaker
                if transcript_data.get('multi_speaker', False):
                    f.write(f"Speakers: {transcript_data.get('total_speakers', 1)}\n")
                    f.write(f"Processing: Multi-speaker with diarization\n")
                else:
                    f.write(f"Processing: Single speaker\n")
                
                f.write("\n")
                
                # Write main transcript
                if transcript_data.get('multi_speaker', False):
                    # Multi-speaker transcript
                    combined_transcript = transcript_data.get('combined_transcript', '')
                    f.write("COMBINED TRANSCRIPT\n")
                    f.write("-" * 20 + "\n")
                    f.write(combined_transcript)
                    
                    # Individual speaker transcripts
                    speakers = transcript_data.get('speakers', [])
                    if speakers:
                        f.write("\n\n" + "=" * 40 + "\n")
                        f.write("INDIVIDUAL SPEAKER TRANSCRIPTS\n")
                        f.write("=" * 40 + "\n")
                        
                        for speaker in speakers:
                            speaker_label = speaker['speaker_label']
                            speaker_transcript = speaker['transcript']
                            duration = speaker.get('duration', 0)
                            
                            f.write(f"\n{speaker_label}\n")
                            f.write("-" * len(speaker_label) + "\n")
                            f.write(f"Duration: {duration:.1f}s\n")
                            
                            # Speaker characteristics
                            characteristics = speaker.get('characteristics', {})
                            if characteristics:
                                avg_pitch = characteristics.get('avg_pitch_hz', 0)
                                speaking_rate = characteristics.get('speaking_rate_opm', 0)
                                if avg_pitch > 0:
                                    f.write(f"Average Pitch: {avg_pitch:.0f} Hz\n")
                                if speaking_rate > 0:
                                    f.write(f"Speaking Rate: {speaking_rate:.0f} onsets/min\n")
                            
                            f.write("\nTranscript:\n")
                            cleaned_text = self._clean_text(speaker_transcript)
                            formatted_text = self._format_paragraphs(cleaned_text)
                            f.write(formatted_text)
                            f.write("\n")
                else:
                    # Single speaker transcript
                    text = transcript_data.get('text', '')
                    cleaned_text = self._clean_text(text)
                    formatted_text = self._format_paragraphs(cleaned_text)
                    f.write(formatted_text)
                
                # Write detailed segment breakdown if available
                segments = transcript_data.get('segments', [])
                sentence_timestamps = transcript_data.get('sentence_timestamps', [])
                paragraph_timestamps = transcript_data.get('paragraph_timestamps', [])
                
                # If we have enhanced timestamps, use them for detailed breakdown
                if sentence_timestamps:
                    f.write("\n\n" + "=" * 40 + "\n")
                    f.write("SENTENCE-BY-SENTENCE BREAKDOWN\n")
                    f.write("=" * 40 + "\n")
                    
                    for sentence in sentence_timestamps:
                        start_time = self._format_timestamp(sentence['start'])
                        end_time = self._format_timestamp(sentence['end'])
                        duration = sentence.get('duration', 0)
                        confidence = sentence.get('confidence', 0)
                        estimated = sentence.get('estimated', False)
                        
                        f.write(f"[{start_time} - {end_time}] ({duration:.1f}s)")
                        if estimated:
                            f.write(" *estimated")
                        f.write(f" [conf: {confidence:.2f}]")
                        f.write(f"\n{sentence['text'].strip()}\n\n")
                
                elif segments and len(segments) > 1 and not transcript_data.get('multi_speaker', False):
                    f.write("\n\n" + "=" * 40 + "\n")
                    f.write("DETAILED BREAKDOWN\n")
                    f.write("=" * 40 + "\n")
                    
                    for segment in segments:
                        start_time = self._format_timestamp(segment['start'])
                        end_time = self._format_timestamp(segment['end'])
                        f.write(f"[{start_time} - {end_time}]\n")
                        f.write(f"{segment['text'].strip()}\n\n")
                
                # Add paragraph-level timestamps if available
                if paragraph_timestamps:
                    f.write("\n" + "=" * 40 + "\n")
                    f.write("PARAGRAPH BREAKDOWN\n")
                    f.write("=" * 40 + "\n")
                    
                    for paragraph in paragraph_timestamps:
                        start_time = self._format_timestamp(paragraph['start'])
                        end_time = self._format_timestamp(paragraph['end'])
                        duration = paragraph.get('duration', 0)
                        sentence_count = paragraph.get('sentence_count', 0)
                        word_count = paragraph.get('word_count', 0)
                        
                        f.write(f"Paragraph {paragraph['paragraph_id']}: [{start_time} - {end_time}] ")
                        f.write(f"({duration:.1f}s, {sentence_count} sentences, {word_count} words)\n")
                        f.write(f"{paragraph['text']}\n\n")
                
                # Add timing statistics if available
                timestamp_stats = transcript_data.get('timestamp_statistics', {})
                if timestamp_stats:
                    f.write("\n" + "=" * 40 + "\n")
                    f.write("TIMING STATISTICS\n")
                    f.write("=" * 40 + "\n")
                    
                    f.write(f"Total Segments: {timestamp_stats.get('total_segments', 0)}\n")
                    f.write(f"Total Sentences: {timestamp_stats.get('total_sentences', 0)}\n")
                    f.write(f"Average Segment Duration: {timestamp_stats.get('average_segment_duration', 0):.2f}s\n")
                    f.write(f"Average Sentence Duration: {timestamp_stats.get('average_sentence_duration', 0):.2f}s\n")
                    f.write(f"Sentences per Segment: {timestamp_stats.get('sentences_per_segment', 0):.1f}\n")
                    
                    if timestamp_stats.get('estimated_sentences', 0) > 0:
                        f.write(f"Estimated Sentence Timestamps: {timestamp_stats['estimated_sentences']}\n")
                        f.write("(*estimated timestamps are calculated based on text position)\n")
                        
        except Exception as e:
            logger.error(f"TXT generation failed: {str(e)}")
            raise
    
    async def _generate_srt(self, transcript_data: Dict[str, Any], output_path: Path) -> None:
        """Generate SubRip (SRT) subtitle format with enhanced timestamps."""
        try:
            # Use sentence timestamps if available, otherwise fall back to segments
            sentence_timestamps = transcript_data.get('sentence_timestamps', [])
            segments = transcript_data.get('segments', [])
            
            # Prefer sentence-level timestamps for better subtitle granularity
            subtitle_items = sentence_timestamps if sentence_timestamps else segments
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for i, item in enumerate(subtitle_items, 1):
                    # SRT format:
                    # 1
                    # 00:00:01,000 --> 00:00:04,000
                    # Text content
                    
                    start_time = self.timestamp_processor.format_timestamp_for_display(
                        item['start'], 'srt'
                    )
                    end_time = self.timestamp_processor.format_timestamp_for_display(
                        item['end'], 'srt'
                    )
                    text = self._clean_text(item['text'])
                    
                    # Split long lines for better subtitle display
                    text = self._wrap_subtitle_text(text)
                    
                    # Add confidence indicator for estimated timestamps
                    if item.get('estimated', False):
                        text = f"{text} *"
                    
                    f.write(f"{i}\n")
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{text}\n\n")
                    
        except Exception as e:
            logger.error(f"SRT generation failed: {str(e)}")
            raise
    
    async def _generate_vtt(self, transcript_data: Dict[str, Any], output_path: Path) -> None:
        """Generate WebVTT subtitle format with enhanced timestamps."""
        try:
            # Use sentence timestamps if available, otherwise fall back to segments
            sentence_timestamps = transcript_data.get('sentence_timestamps', [])
            segments = transcript_data.get('segments', [])
            
            # Prefer sentence-level timestamps for better subtitle granularity
            subtitle_items = sentence_timestamps if sentence_timestamps else segments
            
            with open(output_path, 'w', encoding='utf-8') as f:
                # WebVTT header
                f.write("WEBVTT\n")
                f.write("Kind: captions\n")
                f.write("Language: ar\n")
                
                # Add metadata about enhanced timestamps
                if sentence_timestamps:
                    f.write("NOTE: Enhanced with sentence-level timestamps\n")
                    timestamp_stats = transcript_data.get('timestamp_statistics', {})
                    if timestamp_stats:
                        total_sentences = timestamp_stats.get('total_sentences', 0)
                        estimated_sentences = timestamp_stats.get('estimated_sentences', 0)
                        f.write(f"NOTE: {total_sentences} sentences, {estimated_sentences} estimated\n")
                
                f.write("\n")
                
                for item in subtitle_items:
                    start_time = self.timestamp_processor.format_timestamp_for_display(
                        item['start'], 'vtt'
                    )
                    end_time = self.timestamp_processor.format_timestamp_for_display(
                        item['end'], 'vtt'
                    )
                    text = self._clean_text(item['text'])
                    
                    # Split long lines for better subtitle display
                    text = self._wrap_subtitle_text(text)
                    
                    # Add confidence indicator for estimated timestamps
                    if item.get('estimated', False):
                        text = f"{text} *"
                    
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{text}\n\n")
                    
        except Exception as e:
            logger.error(f"VTT generation failed: {str(e)}")
            raise
    
    async def _generate_json(self, transcript_data: Dict[str, Any], output_path: Path) -> None:
        """Generate structured JSON output with full metadata and enhanced timestamps."""
        try:
            # Create comprehensive JSON structure
            output_data = {
                'metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'platform': 'Arabic STT Platform',
                    'version': '1.0.0',
                    'language': transcript_data.get('language'),
                    'language_probability': transcript_data.get('language_probability'),
                    'duration': transcript_data.get('duration'),
                    'processing_time': transcript_data.get('processing_time'),
                    'model_size': transcript_data.get('model_size'),
                    'device': transcript_data.get('device'),
                    'multi_speaker': transcript_data.get('multi_speaker', False),
                    'total_speakers': transcript_data.get('total_speakers', 1),
                    'enhanced_timestamps': bool(transcript_data.get('sentence_timestamps'))
                },
                'quality_metrics': transcript_data.get('transcript_metadata', {}),
                'statistics': self._calculate_statistics(transcript_data)
            }
            
            # Add enhanced timestamp data if available
            if transcript_data.get('sentence_timestamps'):
                output_data['sentence_timestamps'] = transcript_data['sentence_timestamps']
                
            if transcript_data.get('paragraph_timestamps'):
                output_data['paragraph_timestamps'] = transcript_data['paragraph_timestamps']
                
            if transcript_data.get('enhanced_segments'):
                output_data['enhanced_segments'] = transcript_data['enhanced_segments']
                
            if transcript_data.get('timestamp_statistics'):
                output_data['timestamp_statistics'] = transcript_data['timestamp_statistics']
            
            # Handle multi-speaker vs single-speaker data differently
            if transcript_data.get('multi_speaker', False):
                # Multi-speaker format
                output_data['transcript'] = {
                    'combined_text': transcript_data.get('combined_transcript', ''),
                    'type': 'multi_speaker'
                }
                
                # Individual speaker data
                output_data['speakers'] = []
                for speaker in transcript_data.get('speakers', []):
                    speaker_data = {
                        'speaker_id': speaker['speaker_id'],
                        'speaker_label': speaker['speaker_label'],
                        'transcript': speaker['transcript'],
                        'word_count': len(speaker['transcript'].split()),
                        'duration': speaker.get('duration', 0),
                        'confidence': speaker.get('confidence', 0),
                        'characteristics': speaker.get('characteristics', {}),
                        'segments': speaker.get('segments', [])
                    }
                    output_data['speakers'].append(speaker_data)
                
                # Processing metadata
                if 'processing_metadata' in transcript_data:
                    output_data['processing_metadata'] = transcript_data['processing_metadata']
                
            else:
                # Single speaker format
                output_data['transcript'] = {
                    'full_text': transcript_data.get('text', ''),
                    'word_count': len(transcript_data.get('text', '').split()),
                    'segment_count': len(transcript_data.get('segments', [])),
                    'type': 'single_speaker'
                }
                
                # Process segments with enhanced data
                output_data['segments'] = []
                for segment in transcript_data.get('segments', []):
                    segment_data = {
                        'id': segment['id'],
                        'start': segment['start'],
                        'end': segment['end'],
                        'duration': segment['end'] - segment['start'],
                        'text': segment['text'].strip(),
                        'confidence_scores': {
                            'avg_logprob': segment['avg_logprob'],
                            'compression_ratio': segment['compression_ratio'],
                            'no_speech_prob': segment['no_speech_prob']
                        },
                        'words': segment.get('words', [])
                    }
                    output_data['segments'].append(segment_data)
            
            # Write JSON with proper formatting
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"JSON generation failed: {str(e)}")
            raise
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format timestamp as MM:SS for display."""
        return self.timestamp_processor.format_timestamp_for_display(seconds, 'standard')
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Fix common transcription artifacts
        text = re.sub(r'\b(uh|um|er|ah)\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\.{2,}', '...', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _format_paragraphs(self, text: str) -> str:
        """Format text into readable paragraphs."""
        if not text:
            return ""
        
        # Split on sentence endings followed by capital letters
        sentences = re.split(r'([.!?]+\s+)', text)
        
        # Group sentences into paragraphs (every 3-4 sentences)
        paragraphs = []
        current_paragraph = ""
        sentence_count = 0
        
        for i in range(0, len(sentences), 2):
            if i + 1 < len(sentences):
                sentence = sentences[i] + sentences[i + 1]
                current_paragraph += sentence
                sentence_count += 1
                
                if sentence_count >= 3 or len(current_paragraph) > 200:
                    paragraphs.append(current_paragraph.strip())
                    current_paragraph = ""
                    sentence_count = 0
        
        if current_paragraph.strip():
            paragraphs.append(current_paragraph.strip())
        
        return '\n\n'.join(paragraphs)
    
    def _wrap_subtitle_text(self, text: str, max_length: int = 50) -> str:
        """Wrap text for subtitle display."""
        if len(text) <= max_length:
            return text
        
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            if len(current_line + " " + word) <= max_length:
                current_line += (" " + word) if current_line else word
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
        
        return '\n'.join(lines)
    
    def _calculate_statistics(self, transcript_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate transcript statistics."""
        try:
            segments = transcript_data.get('segments', [])
            text = transcript_data.get('text', '')
            
            if not segments:
                return {}
            
            # Basic stats
            total_duration = transcript_data.get('duration', 0)
            word_count = len(text.split())
            
            # Speech rate (words per minute)
            speech_rate = (word_count / total_duration * 60) if total_duration > 0 else 0
            
            # Confidence scores
            avg_confidence = sum(s['avg_logprob'] for s in segments) / len(segments)
            min_confidence = min(s['avg_logprob'] for s in segments)
            max_confidence = max(s['avg_logprob'] for s in segments)
            
            # Silence analysis
            total_speech_time = sum(s['end'] - s['start'] for s in segments)
            silence_ratio = (total_duration - total_speech_time) / total_duration if total_duration > 0 else 0
            
            # Segment analysis
            segment_lengths = [s['end'] - s['start'] for s in segments]
            avg_segment_length = sum(segment_lengths) / len(segment_lengths) if segment_lengths else 0
            
            return {
                'word_count': word_count,
                'speech_rate_wpm': round(speech_rate, 1),
                'total_duration': round(total_duration, 2),
                'speech_duration': round(total_speech_time, 2),
                'silence_ratio': round(silence_ratio, 3),
                'average_confidence': round(avg_confidence, 3),
                'confidence_range': {
                    'min': round(min_confidence, 3),
                    'max': round(max_confidence, 3)
                },
                'segments': {
                    'total': len(segments),
                    'average_length': round(avg_segment_length, 2)
                }
            }
            
        except Exception as e:
            logger.error(f"Statistics calculation failed: {str(e)}")
            return {'error': str(e)}
    
    async def generate_summary_report(self, transcript_data: Dict[str, Any], job_id: str) -> str:
        """Generate a comprehensive summary report."""
        try:
            output_path = self.config.RESULTS_FOLDER / f"{job_id}_summary.txt"
            stats = self._calculate_statistics(transcript_data)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("ARABIC STT TRANSCRIPTION SUMMARY REPORT\n")
                f.write("=" * 50 + "\n\n")
                
                # Basic Information
                f.write("BASIC INFORMATION:\n")
                f.write(f"• Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"• Language: {transcript_data.get('language', 'Unknown')}\n")
                f.write(f"• Model Used: {transcript_data.get('model_size', 'Unknown')}\n")
                f.write(f"• Processing Device: {transcript_data.get('device', 'Unknown')}\n")
                f.write(f"• Processing Time: {transcript_data.get('processing_time', 0):.2f} seconds\n\n")
                
                # Content Statistics
                f.write("CONTENT STATISTICS:\n")
                f.write(f"• Total Duration: {stats.get('total_duration', 0):.2f} seconds\n")
                f.write(f"• Speech Duration: {stats.get('speech_duration', 0):.2f} seconds\n")
                f.write(f"• Word Count: {stats.get('word_count', 0)}\n")
                f.write(f"• Speech Rate: {stats.get('speech_rate_wpm', 0):.1f} words/minute\n")
                f.write(f"• Total Segments: {stats.get('segments', {}).get('total', 0)}\n")
                f.write(f"• Silence Ratio: {stats.get('silence_ratio', 0):.1%}\n\n")
                
                # Quality Metrics
                f.write("QUALITY METRICS:\n")
                lang_prob = transcript_data.get('language_probability', 0)
                f.write(f"• Language Confidence: {lang_prob:.1%}\n")
                f.write(f"• Average Confidence: {stats.get('average_confidence', 0):.3f}\n")
                
                confidence_range = stats.get('confidence_range', {})
                f.write(f"• Confidence Range: {confidence_range.get('min', 0):.3f} to {confidence_range.get('max', 0):.3f}\n\n")
                
                # Preview
                preview_text = transcript_data.get('text', '')[:200]
                if len(transcript_data.get('text', '')) > 200:
                    preview_text += "..."
                
                f.write("TRANSCRIPT PREVIEW:\n")
                f.write(f'"{preview_text}"\n\n')
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Summary report generation failed: {str(e)}")
            raise

    async def _generate_separate_analysis_files(self, transcript_data: Dict[str, Any], job_id: str) -> None:
        """Generate separate files for speaker analysis and sentiment analysis."""
        try:
            # Generate speaker analysis file if speaker data exists
            if transcript_data.get('multi_speaker', False) or transcript_data.get('speakers'):
                await self._generate_speaker_analysis_file(transcript_data, job_id)
            
            # Generate sentiment analysis file if analysis data exists
            if transcript_data.get('analysis_results') or transcript_data.get('sentiment'):
                await self._generate_sentiment_analysis_file(transcript_data, job_id)
                
        except Exception as e:
            logger.error(f"Separate analysis file generation failed: {str(e)}")
            raise

    async def _generate_speaker_analysis_file(self, transcript_data: Dict[str, Any], job_id: str) -> None:
        """Generate detailed speaker analysis file."""
        try:
            output_path = self.config.RESULTS_FOLDER / f"{job_id}_speaker_analysis.txt"
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("SPEAKER DIARIZATION & ANALYSIS REPORT\n")
                f.write("=" * 50 + "\n\n")
                
                # Basic speaker information
                f.write("SPEAKER OVERVIEW:\n")
                f.write(f"• Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"• Total Speakers: {transcript_data.get('total_speakers', 1)}\n")
                f.write(f"• Multi-speaker Processing: {transcript_data.get('multi_speaker', False)}\n")
                f.write(f"• Total Duration: {transcript_data.get('duration', 0):.2f} seconds\n\n")
                
                # Individual speaker analysis
                speakers = transcript_data.get('speakers', [])
                if speakers:
                    f.write("INDIVIDUAL SPEAKER ANALYSIS:\n")
                    f.write("=" * 30 + "\n\n")
                    
                    for i, speaker in enumerate(speakers, 1):
                        speaker_id = speaker.get('speaker_id', f'Speaker_{i}')
                        speaker_label = speaker.get('speaker_label', f'Speaker {i}')
                        
                        f.write(f"{speaker_label} ({speaker_id})\n")
                        f.write("-" * len(f"{speaker_label} ({speaker_id})") + "\n")
                        
                        # Basic stats
                        duration = speaker.get('duration', 0)
                        word_count = len(speaker.get('transcript', '').split())
                        confidence = speaker.get('confidence', 0)
                        
                        f.write(f"Duration: {duration:.2f} seconds\n")
                        f.write(f"Word Count: {word_count}\n")
                        f.write(f"Confidence: {confidence:.3f}\n")
                        
                        # Speaking rate
                        if duration > 0:
                            speaking_rate = (word_count / duration) * 60
                            f.write(f"Speaking Rate: {speaking_rate:.1f} words/minute\n")
                        
                        # Voice characteristics
                        characteristics = speaker.get('characteristics', {})
                        if characteristics:
                            f.write("\nVoice Characteristics:\n")
                            
                            avg_pitch = characteristics.get('avg_pitch_hz', 0)
                            if avg_pitch > 0:
                                f.write(f"• Average Pitch: {avg_pitch:.0f} Hz\n")
                            
                            pitch_range = characteristics.get('pitch_range_hz', 0)
                            if pitch_range > 0:
                                f.write(f"• Pitch Range: {pitch_range:.0f} Hz\n")
                            
                            energy = characteristics.get('avg_energy', 0)
                            if energy > 0:
                                f.write(f"• Average Energy: {energy:.3f}\n")
                            
                            speaking_rate_opm = characteristics.get('speaking_rate_opm', 0)
                            if speaking_rate_opm > 0:
                                f.write(f"• Speaking Rate: {speaking_rate_opm:.0f} onsets/min\n")
                        
                        # Segments breakdown
                        segments = speaker.get('segments', [])
                        if segments:
                            f.write(f"\nSpeaking Segments ({len(segments)} total):\n")
                            for j, segment in enumerate(segments[:10], 1):  # Show first 10 segments
                                start_time = self._format_timestamp(segment.get('start', 0))
                                end_time = self._format_timestamp(segment.get('end', 0))
                                segment_text = segment.get('text', '')[:100]
                                if len(segment.get('text', '')) > 100:
                                    segment_text += "..."
                                f.write(f"  {j}. [{start_time} - {end_time}] {segment_text}\n")
                            
                            if len(segments) > 10:
                                f.write(f"  ... and {len(segments) - 10} more segments\n")
                        
                        f.write("\n" + "=" * 30 + "\n\n")
                
                # Processing metadata
                processing_metadata = transcript_data.get('processing_metadata', {})
                if processing_metadata:
                    f.write("PROCESSING DETAILS:\n")
                    f.write("-" * 20 + "\n")
                    
                    diarization_method = processing_metadata.get('diarization_method', 'Unknown')
                    f.write(f"Diarization Method: {diarization_method}\n")
                    
                    processing_time = processing_metadata.get('diarization_time', 0)
                    if processing_time > 0:
                        f.write(f"Diarization Processing Time: {processing_time:.2f} seconds\n")
                    
                    quality_score = processing_metadata.get('diarization_quality', 0)
                    if quality_score > 0:
                        f.write(f"Diarization Quality Score: {quality_score:.3f}\n")
            
            logger.info(f"Generated speaker analysis file: {output_path}")
            
        except Exception as e:
            logger.error(f"Speaker analysis file generation failed: {str(e)}")
            raise

    async def _generate_sentiment_analysis_file(self, transcript_data: Dict[str, Any], job_id: str) -> None:
        """Generate detailed sentiment and text analysis file."""
        try:
            output_path = self.config.RESULTS_FOLDER / f"{job_id}_sentiment_analysis.txt"
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("SENTIMENT & TEXT ANALYSIS REPORT\n")
                f.write("=" * 50 + "\n\n")
                
                # Basic information
                f.write("ANALYSIS OVERVIEW:\n")
                f.write(f"• Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"• Text Length: {len(transcript_data.get('text', ''))} characters\n")
                f.write(f"• Word Count: {len(transcript_data.get('text', '').split())}\n")
                
                # Get analysis results
                analysis_results = transcript_data.get('analysis_results', {})
                
                # Sentiment Analysis
                if 'sentiment' in analysis_results:
                    sentiment = analysis_results['sentiment']
                    f.write("\n" + "=" * 30 + "\n")
                    f.write("SENTIMENT ANALYSIS:\n")
                    f.write("=" * 30 + "\n")
                    
                    f.write(f"Overall Sentiment: {sentiment.get('label', 'Unknown').title()}\n")
                    f.write(f"Sentiment Score: {sentiment.get('score', 0):.3f} (range: -1 to +1)\n")
                    f.write(f"Confidence: {sentiment.get('confidence', 0):.3f}\n")
                    
                    emotions = sentiment.get('emotions', [])
                    if emotions:
                        f.write(f"Detected Emotions: {', '.join(emotions)}\n")
                
                # Entity Recognition
                if 'entities' in analysis_results:
                    entities = analysis_results['entities']
                    f.write("\n" + "=" * 30 + "\n")
                    f.write("NAMED ENTITY RECOGNITION:\n")
                    f.write("=" * 30 + "\n")
                    
                    entity_types = ['persons', 'locations', 'organizations', 'dates', 'other']
                    for entity_type in entity_types:
                        entity_list = entities.get(entity_type, [])
                        if entity_list:
                            f.write(f"{entity_type.title()}: {', '.join(entity_list)}\n")
                
                # Topic Analysis
                if 'topics' in analysis_results:
                    topics = analysis_results['topics']
                    f.write("\n" + "=" * 30 + "\n")
                    f.write("TOPIC ANALYSIS:\n")
                    f.write("=" * 30 + "\n")
                    
                    main_topics = topics.get('main_topics', [])
                    if main_topics:
                        f.write(f"Main Topics: {', '.join(main_topics)}\n")
                    
                    categories = topics.get('categories', [])
                    if categories:
                        f.write(f"Categories: {', '.join(categories)}\n")
                    
                    topic_scores = topics.get('topic_scores', {})
                    if topic_scores:
                        f.write("\nTopic Scores:\n")
                        for topic, score in topic_scores.items():
                            f.write(f"• {topic}: {score:.3f}\n")
                
                # Keywords
                if 'keywords' in analysis_results:
                    keywords = analysis_results['keywords']
                    if keywords:
                        f.write("\n" + "=" * 30 + "\n")
                        f.write("KEYWORD EXTRACTION:\n")
                        f.write("=" * 30 + "\n")
                        f.write(f"Key Terms: {', '.join(keywords)}\n")
                
                # Complexity Analysis
                if 'complexity' in analysis_results:
                    complexity = analysis_results['complexity']
                    f.write("\n" + "=" * 30 + "\n")
                    f.write("TEXT COMPLEXITY ANALYSIS:\n")
                    f.write("=" * 30 + "\n")
                    
                    f.write(f"Complexity Level: {complexity.get('complexity_level', 'Unknown').title()}\n")
                    f.write(f"Readability Score: {complexity.get('readability_score', 0):.2f}\n")
                    f.write(f"Sentence Count: {complexity.get('sentence_count', 0)}\n")
                    f.write(f"Average Sentence Length: {complexity.get('average_sentence_length', 0):.1f} words\n")
                    f.write(f"Vocabulary Richness: {complexity.get('vocabulary_richness', 0):.3f}\n")
                
                # Analysis Summary
                if 'summary' in analysis_results:
                    summary = analysis_results['summary']
                    f.write("\n" + "=" * 30 + "\n")
                    f.write("ANALYSIS SUMMARY:\n")
                    f.write("=" * 30 + "\n")
                    
                    text_stats = summary.get('text_stats', {})
                    f.write(f"Processing Time: {text_stats.get('processing_time', 0):.2f} seconds\n")
                    f.write(f"Language: {text_stats.get('language', 'Unknown')}\n")
                    
                    if 'sentiment' in summary:
                        sent_summary = summary['sentiment']
                        f.write(f"Sentiment Summary: {sent_summary.get('label', 'Unknown')} ")
                        f.write(f"(confidence: {sent_summary.get('confidence', 0):.2f})\n")
                    
                    if 'entities' in summary:
                        ent_summary = summary['entities']
                        f.write(f"Total Entities Found: {ent_summary.get('total_entities', 0)}\n")
                    
                    if 'topics' in summary:
                        topic_summary = summary['topics']
                        f.write(f"Topics Identified: {topic_summary.get('topic_count', 0)}\n")
                
                # Text preview for context
                f.write("\n" + "=" * 30 + "\n")
                f.write("TEXT SAMPLE:\n")
                f.write("=" * 30 + "\n")
                
                text_sample = transcript_data.get('text', '')[:500]
                if len(transcript_data.get('text', '')) > 500:
                    text_sample += "..."
                f.write(f'"{text_sample}"\n')
            
            logger.info(f"Generated sentiment analysis file: {output_path}")
            
        except Exception as e:
            logger.error(f"Sentiment analysis file generation failed: {str(e)}")
            raise

    async def generate_timestamp_report(self, transcript_data: Dict[str, Any], job_id: str) -> str:
        """Generate a detailed timestamp analysis report."""
        try:
            output_path = self.config.RESULTS_FOLDER / f"{job_id}_timestamps.txt"
            
            # Ensure timestamps are enhanced
            if not transcript_data.get('sentence_timestamps'):
                transcript_data = self.timestamp_processor.enhance_timestamps(transcript_data)
            
            sentence_timestamps = transcript_data.get('sentence_timestamps', [])
            paragraph_timestamps = transcript_data.get('paragraph_timestamps', [])
            enhanced_segments = transcript_data.get('enhanced_segments', [])
            timestamp_stats = transcript_data.get('timestamp_statistics', {})
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("ENHANCED TIMESTAMP ANALYSIS REPORT\n")
                f.write("=" * 50 + "\n\n")
                
                # Basic timing information
                f.write("TIMING OVERVIEW:\n")
                f.write(f"• Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"• Total Duration: {transcript_data.get('duration', 0):.2f} seconds\n")
                f.write(f"• Total Segments: {timestamp_stats.get('total_segments', 0)}\n")
                f.write(f"• Total Sentences: {timestamp_stats.get('total_sentences', 0)}\n")
                f.write(f"• Total Paragraphs: {len(paragraph_timestamps)}\n\n")
                
                # Timing statistics
                if timestamp_stats:
                    f.write("TIMING STATISTICS:\n")
                    f.write(f"• Average Segment Duration: {timestamp_stats.get('average_segment_duration', 0):.2f}s\n")
                    f.write(f"• Average Sentence Duration: {timestamp_stats.get('average_sentence_duration', 0):.2f}s\n")
                    f.write(f"• Sentences per Segment: {timestamp_stats.get('sentences_per_segment', 0):.1f}\n")
                    f.write(f"• Estimated Timestamps: {timestamp_stats.get('estimated_sentences', 0)}\n\n")
                
                # Sentence-level timing details
                if sentence_timestamps:
                    f.write("SENTENCE TIMING DETAILS:\n")
                    f.write("-" * 30 + "\n")
                    
                    for i, sentence in enumerate(sentence_timestamps[:10], 1):  # Show first 10
                        start_time = self.timestamp_processor.format_timestamp_for_display(
                            sentence['start'], 'human'
                        )
                        end_time = self.timestamp_processor.format_timestamp_for_display(
                            sentence['end'], 'human'
                        )
                        duration = sentence.get('duration', 0)
                        word_count = sentence.get('word_count', 0)
                        estimated = " (estimated)" if sentence.get('estimated') else ""
                        
                        f.write(f"#{i}: [{start_time} - {end_time}] {duration:.1f}s, {word_count} words{estimated}\n")
                        f.write(f"     \"{sentence['text'][:100]}{'...' if len(sentence['text']) > 100 else ''}\"\n\n")
                    
                    if len(sentence_timestamps) > 10:
                        f.write(f"... and {len(sentence_timestamps) - 10} more sentences\n\n")
                
                # Paragraph timing details
                if paragraph_timestamps:
                    f.write("PARAGRAPH TIMING DETAILS:\n")
                    f.write("-" * 30 + "\n")
                    
                    for paragraph in paragraph_timestamps:
                        start_time = self.timestamp_processor.format_timestamp_for_display(
                            paragraph['start'], 'human'
                        )
                        end_time = self.timestamp_processor.format_timestamp_for_display(
                            paragraph['end'], 'human'
                        )
                        duration = paragraph.get('duration', 0)
                        sentence_count = paragraph.get('sentence_count', 0)
                        word_count = paragraph.get('word_count', 0)
                        
                        f.write(f"Paragraph {paragraph['paragraph_id']}: [{start_time} - {end_time}]\n")
                        f.write(f"Duration: {duration:.1f}s, Sentences: {sentence_count}, Words: {word_count}\n")
                        f.write(f"Text: \"{paragraph['text'][:150]}{'...' if len(paragraph['text']) > 150 else ''}\"\n\n")
                
                # Word-level timing analysis (if available)
                if enhanced_segments:
                    f.write("WORD-LEVEL TIMING ANALYSIS:\n")
                    f.write("-" * 30 + "\n")
                    
                    for i, segment in enumerate(enhanced_segments[:3], 1):  # Show first 3 segments
                        word_stats = segment.get('word_timing_stats', {})
                        if word_stats:
                            f.write(f"Segment {i}:\n")
                            f.write(f"• Average word duration: {word_stats.get('average_word_duration', 0):.3f}s\n")
                            f.write(f"• Words per second: {word_stats.get('words_per_second', 0):.1f}\n")
                            f.write(f"• Total speech time: {word_stats.get('total_speech_time', 0):.2f}s\n")
                            
                            # Show pauses
                            pauses = segment.get('pauses', [])
                            if pauses:
                                f.write(f"• Notable pauses: {len(pauses)}\n")
                                for pause in pauses[:3]:  # Show first 3 pauses
                                    f.write(f"  - {pause['duration']:.2f}s pause between '{pause['before_word']}' and '{pause['after_word']}'\n")
                            f.write("\n")
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Timestamp report generation failed: {str(e)}")
            raise
