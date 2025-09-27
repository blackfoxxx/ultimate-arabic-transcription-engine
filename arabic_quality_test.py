#!/usr/bin/env python3
"""
Arabic Quality Test CLI
======================
Simple CLI to demonstrate Arabic transcription quality improvements
"""

import os
import sys
import json
import time
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def find_test_audio():
    """Find an audio file for testing"""
    uploads = current_dir / "uploads"
    if uploads.exists():
        # Look for small test files first
        for audio_file in uploads.glob("*test*.wav"):
            return str(audio_file)
        for audio_file in uploads.glob("*.wav"):
            return str(audio_file)
    return None

def test_ultimate_arabic():
    """Test the Ultimate Arabic Engine"""
    print("🔥 ULTIMATE ARABIC ENGINE v3.0 QUALITY TEST")
    print("=" * 50)
    
    try:
        from core.ultimate_arabic_transcription_engine import UltimateArabicTranscriptionEngine
        
        audio_file = find_test_audio()
        if not audio_file:
            print("❌ No audio files found in uploads directory")
            return
        
        print(f"📁 Testing file: {os.path.basename(audio_file)}")
        print(f"📊 File size: {os.path.getsize(audio_file) / 1024:.1f} KB")
        
        # Initialize engine
        print("🔧 Initializing Ultimate Arabic Engine...")
        engine = UltimateArabicTranscriptionEngine(model_size="small", device="cpu")
        
        if not engine.initialize_model():
            print("❌ Engine initialization failed")
            return
        
        print("✅ Engine ready")
        
        # Transcribe
        print("🎯 Transcribing...")
        start_time = time.time()
        result = engine.transcribe(audio_file)
        processing_time = time.time() - start_time
        
        print(f"⏱️  Completed in {processing_time:.2f} seconds")
        
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
            return
        
        # Show results
        transcript = result['transcript']['full_text']
        quality = result.get('quality_metrics', {})
        
        print("\n📊 QUALITY METRICS:")
        for metric, value in quality.items():
            if isinstance(value, (int, float)):
                print(f"  {metric}: {value:.3f}")
        
        print(f"\n📝 TRANSCRIPT ({len(transcript)} characters):")
        print("-" * 40)
        if transcript.strip():
            print(transcript)
        else:
            print("(Empty - audio may be silent)")
        print("-" * 40)
        
        # Quality assessment
        quality_score = quality.get('quality_score', 0)
        purity_score = quality.get('language_purity', 0)
        
        print(f"\n🏆 QUALITY ASSESSMENT:")
        print(f"  Overall Score: {quality_score:.3f}/1.0")
        print(f"  Arabic Purity: {purity_score:.3f}/1.0")
        
        if quality_score > 0.7:
            print("✅ EXCELLENT quality - Pure Arabic transcription")
        elif quality_score > 0.5:
            print("✅ GOOD quality - High Arabic accuracy")
        elif transcript.strip():
            print("⚠️  MODERATE quality - Some mixed content")
        else:
            print("ℹ️  No content transcribed (silent audio)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    if len(sys.argv) > 1:
        if sys.argv[1] == "--help":
            print("""
Arabic Quality Test CLI
======================

Usage:
    python3 arabic_quality_test.py              # Test with auto-found audio
    python3 arabic_quality_test.py --help       # Show this help
    
This CLI demonstrates the Ultimate Arabic Engine v3.0 quality improvements:
- Pure Arabic output (no mixed languages)
- Superior word formation and coherence
- Advanced quality metrics
- Optimized for Arabic phonetics and morphology
""")
            return
    
    success = test_ultimate_arabic()
    
    if success:
        print("\n🎉 Test completed! Ultimate Arabic Engine v3.0 is ready.")
        print("\n💡 For full CLI usage:")
        print("   python3 arabic_cli_ultimate.py --file your_audio.wav")
    else:
        print("\n❌ Test failed. Check the installation.")

if __name__ == "__main__":
    main()
