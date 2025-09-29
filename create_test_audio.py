#!/usr/bin/env python3
"""
Create a test audio file for CLI testing
"""
import numpy as np
import wave
import os

def create_test_audio(filename="uploads/test_audio.wav", duration=5, sample_rate=16000):
    """Create a simple test audio file with a sine wave"""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Generate a sine wave
    frequency = 440  # A4 note
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_data = np.sin(2 * np.pi * frequency * t)
    
    # Scale to 16-bit range
    audio_data = (audio_data * 32767).astype(np.int16)
    
    # Write to WAV file
    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())
    
    print(f"✅ Created test audio file: {filename}")
    print(f"   Duration: {duration}s, Sample rate: {sample_rate}Hz")

if __name__ == "__main__":
    create_test_audio()