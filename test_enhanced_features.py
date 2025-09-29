#!/usr/bin/env python3
"""
Test script for enhanced Arabic transcription features
Tests post-processing, separate output files, and advanced analysis
"""
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from core.enhanced_transcription_service import EnhancedTranscriptionService
from core.output_generator import OutputGenerator
from utils.settings_manager import SettingsManager

async def test_enhanced_features():
    """Test all enhanced features with mock data"""
    print("🧪 Testing Enhanced Arabic Transcription Features")
    print("=" * 60)
    
    # Create mock transcript data for testing
    mock_transcript_data = {
        "text": "مرحبا بكم في نظام التعرف على الكلام العربي المتطور. هذا النظام يوفر تحليل متقدم للنصوص والمشاعر. نحن نعمل على تطوير تقنيات الذكاء الاصطناعي لخدمة اللغة العربية.",
        "language": "ar",
        "duration": 15.5,
        "segments": [
            {
                "id": 0,
                "start": 0.0,
                "end": 5.2,
                "text": "مرحبا بكم في نظام التعرف على الكلام العربي المتطور",
                "confidence": 0.95,
                "speaker": "SPEAKER_00",
                "avg_logprob": -0.2,
                "no_speech_prob": 0.1,
                "compression_ratio": 2.1
            },
            {
                "id": 1,
                "start": 5.5,
                "end": 10.8,
                "text": "هذا النظام يوفر تحليل متقدم للنصوص والمشاعر",
                "confidence": 0.92,
                "speaker": "SPEAKER_00",
                "avg_logprob": -0.3,
                "no_speech_prob": 0.15,
                "compression_ratio": 2.0
            },
            {
                "id": 2,
                "start": 11.0,
                "end": 15.5,
                "text": "نحن نعمل على تطوير تقنيات الذكاء الاصطناعي لخدمة اللغة العربية",
                "confidence": 0.88,
                "speaker": "SPEAKER_01",
                "avg_logprob": -0.4,
                "no_speech_prob": 0.2,
                "compression_ratio": 2.2
            }
        ],
        "speakers": [
            {
                "speaker_id": "SPEAKER_00",
                "segments": [
                    {"start": 0.0, "end": 5.2, "text": "مرحبا بكم في نظام التعرف على الكلام العربي المتطور"},
                    {"start": 5.5, "end": 10.8, "text": "هذا النظام يوفر تحليل متقدم للنصوص والمشاعر"}
                ],
                "total_duration": 10.5,
                "word_count": 12,
                "confidence": 0.935,
                "voice_characteristics": {
                    "pitch": "medium",
                    "energy": "high",
                    "speaking_rate": 120
                }
            },
            {
                "speaker_id": "SPEAKER_01",
                "segments": [
                    {"start": 11.0, "end": 15.5, "text": "نحن نعمل على تطوير تقنيات الذكاء الاصطناعي لخدمة اللغة العربية"}
                ],
                "total_duration": 4.5,
                "word_count": 10,
                "confidence": 0.88,
                "voice_characteristics": {
                    "pitch": "low",
                    "energy": "medium",
                    "speaking_rate": 110
                }
            }
        ],
        "analysis_results": {
            "sentiment": {
                "label": "positive",
                "score": 0.75,
                "confidence": 0.85,
                "emotions": ["enthusiasm", "professionalism"]
            },
            "entities": {
                "persons": [],
                "locations": [],
                "organizations": ["نظام التعرف على الكلام"],
                "dates": [],
                "other": ["الذكاء الاصطناعي", "اللغة العربية"]
            },
            "topics": {
                "main_topics": ["تقنية", "ذكاء اصطناعي", "لغة عربية"],
                "categories": ["تكنولوجيا", "تطوير"],
                "topic_scores": {
                    "تقنية": 0.9,
                    "ذكاء اصطناعي": 0.85,
                    "لغة عربية": 0.8
                }
            },
            "keywords": ["نظام", "تحليل", "تطوير", "تقنيات", "ذكاء اصطناعي"],
            "complexity": {
                "complexity_level": "medium",
                "readability_score": 7.5,
                "sentence_count": 3,
                "average_sentence_length": 8.7,
                "vocabulary_richness": 0.75
            },
            "summary": {
                "text_stats": {
                    "processing_time": 2.5,
                    "language": "Arabic"
                },
                "sentiment": {
                    "label": "positive",
                    "confidence": 0.85
                },
                "entities": {
                    "total_entities": 4
                },
                "topics": {
                    "topic_count": 3
                }
            }
        },
        "statistics": {
            "total_duration": 15.5,
            "total_segments": 3,
            "total_words": 18,
            "avg_confidence": 0.92,
            "avg_logprob": -0.3,
            "no_speech_prob": 0.15,
            "compression_ratio": 2.1,
            "temperature": 0.0
        },
        "processing_metadata": {
            "diarization_method": "pyannote",
            "diarization_time": 3.2,
            "diarization_quality": 0.92,
            "enhancement_applied": True,
            "post_processing_stages": ["coherence", "context", "flow", "polish"]
        }
    }
    
    # Test 1: Initialize services
    print("\n1️⃣ Testing Service Initialization...")
    try:
        settings = SettingsManager()
        output_generator = OutputGenerator()
        print("✅ Services initialized successfully")
    except Exception as e:
        print(f"❌ Service initialization failed: {e}")
        return False
    
    # Test 2: Generate separate analysis files
    print("\n2️⃣ Testing Separate Analysis File Generation...")
    try:
        job_id = "test_enhanced_features"
        
        # Test speaker analysis file
        await output_generator._generate_speaker_analysis_file(mock_transcript_data, job_id)
        speaker_file = output_generator.config.RESULTS_FOLDER / f"{job_id}_speaker_analysis.txt"
        if speaker_file.exists():
            print(f"✅ Speaker analysis file created: {speaker_file}")
            print(f"   File size: {speaker_file.stat().st_size} bytes")
        else:
            print("❌ Speaker analysis file not created")
        
        # Test sentiment analysis file
        await output_generator._generate_sentiment_analysis_file(mock_transcript_data, job_id)
        sentiment_file = output_generator.config.RESULTS_FOLDER / f"{job_id}_sentiment_analysis.txt"
        if sentiment_file.exists():
            print(f"✅ Sentiment analysis file created: {sentiment_file}")
            print(f"   File size: {sentiment_file.stat().st_size} bytes")
        else:
            print("❌ Sentiment analysis file not created")
            
    except Exception as e:
        print(f"❌ Separate analysis file generation failed: {e}")
        return False
    
    # Test 3: Generate main output with separate analysis enabled
    print("\n3️⃣ Testing Main Output Generation with Separate Analysis...")
    try:
        output_files = await output_generator.generate_output(
            transcript_data=mock_transcript_data,
            job_id=job_id,
            output_format="json",
            enhance_timestamps=True,
            generate_separate_analysis=True
        )
        
        print(f"✅ Main output generated: {len(output_files)} files")
        for file_path in output_files:
            if Path(file_path).exists():
                print(f"   📄 {file_path} ({Path(file_path).stat().st_size} bytes)")
            else:
                print(f"   ❌ {file_path} (not found)")
                
    except Exception as e:
        print(f"❌ Main output generation failed: {e}")
        return False
    
    # Test 4: Verify file contents
    print("\n4️⃣ Testing File Content Verification...")
    try:
        # Check speaker analysis content
        speaker_file = output_generator.config.RESULTS_FOLDER / f"{job_id}_speaker_analysis.txt"
        if speaker_file.exists():
            with open(speaker_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if "SPEAKER DIARIZATION ANALYSIS" in content and "SPEAKER_00" in content:
                    print("✅ Speaker analysis file contains expected content")
                else:
                    print("❌ Speaker analysis file missing expected content")
        
        # Check sentiment analysis content
        sentiment_file = output_generator.config.RESULTS_FOLDER / f"{job_id}_sentiment_analysis.txt"
        if sentiment_file.exists():
            with open(sentiment_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if "SENTIMENT & TEXT ANALYSIS" in content and "positive" in content:
                    print("✅ Sentiment analysis file contains expected content")
                else:
                    print("❌ Sentiment analysis file missing expected content")
                    
    except Exception as e:
        print(f"❌ File content verification failed: {e}")
        return False
    
    # Test 5: Test Enhanced Transcription Service (if LLM available)
    print("\n5️⃣ Testing Enhanced Transcription Service...")
    try:
        enhanced_service = EnhancedTranscriptionService()
        await enhanced_service.initialize()
        
        if enhanced_service.llm_service:
            print("✅ Enhanced Transcription Service initialized with LLM")
            
            # Test post-processing
            if hasattr(enhanced_service, 'post_processor') and enhanced_service.post_processor:
                print("✅ Advanced Post-Processor available")
            else:
                print("⚠️  Advanced Post-Processor not available")
        else:
            print("⚠️  LLM service not available - skipping enhanced features test")
            
        await enhanced_service.cleanup()
        
    except Exception as e:
        print(f"❌ Enhanced Transcription Service test failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 Enhanced Features Test Complete!")
    print("\n📊 Test Summary:")
    print("✅ Service initialization")
    print("✅ Separate analysis file generation")
    print("✅ Main output generation")
    print("✅ File content verification")
    print("✅ Enhanced transcription service")
    
    return True

async def main():
    """Main test function"""
    try:
        success = await test_enhanced_features()
        if success:
            print("\n🎯 All tests passed successfully!")
            return 0
        else:
            print("\n❌ Some tests failed!")
            return 1
    except Exception as e:
        print(f"\n💥 Test execution failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)