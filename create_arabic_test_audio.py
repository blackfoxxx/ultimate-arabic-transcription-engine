#!/usr/bin/env python3
"""
Create a test audio file with Arabic speech for testing the enhanced transcription system
"""
import os
import sys
from pathlib import Path

def create_arabic_test_audio():
    """Create a test audio file with Arabic speech using TTS"""
    try:
        # Try to use pyttsx3 for text-to-speech
        import pyttsx3
        
        # Arabic text for testing
        arabic_text = "مرحبا بكم في نظام التعرف على الكلام العربي. هذا اختبار لجودة النسخ والتحليل المتقدم."
        
        # Initialize TTS engine
        engine = pyttsx3.init()
        
        # Set properties
        engine.setProperty('rate', 150)  # Speed of speech
        engine.setProperty('volume', 0.9)  # Volume level
        
        # Try to set Arabic voice if available
        voices = engine.getProperty('voices')
        for voice in voices:
            if 'arabic' in voice.name.lower() or 'ar' in voice.id.lower():
                engine.setProperty('voice', voice.id)
                break
        
        # Create output directory
        output_dir = Path("uploads")
        output_dir.mkdir(exist_ok=True)
        
        # Save to file
        output_file = output_dir / "arabic_test_audio.wav"
        engine.save_to_file(arabic_text, str(output_file))
        engine.runAndWait()
        
        print(f"✅ Created Arabic test audio file: {output_file}")
        print(f"   Text: {arabic_text}")
        return str(output_file)
        
    except ImportError:
        print("❌ pyttsx3 not available. Creating a simple tone instead...")
        return create_simple_tone()
    except Exception as e:
        print(f"❌ Error creating Arabic audio: {e}")
        return create_simple_tone()

def create_simple_tone():
    """Fallback: Create a simple tone for testing"""
    import numpy as np
    import wave
    
    filename = "uploads/simple_test_audio.wav"
    duration = 3
    sample_rate = 16000
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Generate multiple tones to simulate speech patterns
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Create a more complex waveform that might trigger transcription
    audio_data = (
        0.3 * np.sin(2 * np.pi * 200 * t) +  # Low frequency
        0.3 * np.sin(2 * np.pi * 400 * t) +  # Mid frequency
        0.2 * np.sin(2 * np.pi * 800 * t) +  # High frequency
        0.1 * np.random.normal(0, 0.1, len(t))  # Some noise
    )
    
    # Add amplitude modulation to simulate speech
    modulation = 0.5 + 0.5 * np.sin(2 * np.pi * 5 * t)
    audio_data = audio_data * modulation
    
    # Scale to 16-bit range
    audio_data = (audio_data * 16000).astype(np.int16)
    
    # Write to WAV file
    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())
    
    print(f"✅ Created simple test audio file: {filename}")
    print(f"   Duration: {duration}s, Sample rate: {sample_rate}Hz")
    return filename

if __name__ == "__main__":
    create_arabic_test_audio()