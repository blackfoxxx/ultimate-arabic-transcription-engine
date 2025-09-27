"""
Example usage scripts for Arabic STT Platform API
"""

import requests
import time
import json
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:5000"
API_KEY = "arabic-stt-api-key"  # Set this if API key is required

def upload_and_process_file(file_path, model_size="medium", language="ar", output_formats=None):
    """
    Upload and process an audio/video file.
    
    Args:
        file_path: Path to the audio/video file
        model_size: Whisper model size (tiny, base, small, medium, large)
        language: Language code (ar, en, auto)
        output_formats: List of output formats (txt, srt, vtt, json)
    
    Returns:
        Dictionary with job results
    """
    if output_formats is None:
        output_formats = ["txt", "srt"]
    
    # Prepare the request
    files = {'file': open(file_path, 'rb')}
    data = {
        'model_size': model_size,
        'language': language,
        'output_formats': output_formats,
        'noise_reduction': 'auto'
    }
    
    headers = {}
    if API_KEY:
        headers['Authorization'] = f'Bearer {API_KEY}'
    
    try:
        print(f"🚀 Uploading {file_path}...")
        
        # Upload file
        response = requests.post(
            f"{BASE_URL}/api/transcribe",
            files=files,
            data=data,
            headers=headers
        )
        
        if response.status_code != 202:
            print(f"❌ Upload failed: {response.json()}")
            return None
        
        job_data = response.json()
        job_id = job_data['job_id']
        print(f"✅ Upload successful. Job ID: {job_id}")
        
        # Poll for completion
        print("⏳ Processing...")
        while True:
            status_response = requests.get(f"{BASE_URL}/status/{job_id}")
            
            if status_response.status_code != 200:
                print(f"❌ Status check failed: {status_response.json()}")
                return None
            
            status_data = status_response.json()
            print(f"   Status: {status_data['status']} - {status_data.get('message', '')}")
            
            if status_data['status'] == 'completed':
                print("🎉 Processing completed!")
                return download_results(job_id, output_formats)
            elif status_data['status'] == 'failed':
                print(f"❌ Processing failed: {status_data.get('message', '')}")
                return None
            
            time.sleep(2)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None
    finally:
        files['file'].close()

def download_results(job_id, formats):
    """Download results for a completed job."""
    results = {}
    
    for fmt in formats:
        try:
            print(f"📥 Downloading {fmt.upper()} format...")
            response = requests.get(f"{BASE_URL}/download/{job_id}/{fmt}")
            
            if response.status_code == 200:
                # Save file
                filename = f"transcript_{job_id}.{fmt}"
                with open(filename, 'wb') as f:
                    f.write(response.content)
                results[fmt] = filename
                print(f"   ✅ Saved as {filename}")
            else:
                print(f"   ❌ Failed to download {fmt}: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Download error for {fmt}: {e}")
    
    return results

def batch_process_directory(directory_path, **kwargs):
    """
    Process all audio/video files in a directory.
    
    Args:
        directory_path: Path to directory containing audio/video files
        **kwargs: Arguments to pass to upload_and_process_file
    """
    directory = Path(directory_path)
    
    if not directory.exists():
        print(f"❌ Directory not found: {directory_path}")
        return
    
    # Supported file extensions
    audio_extensions = {'.mp3', '.wav', '.flac', '.m4a', '.ogg', '.aac'}
    video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.webm'}
    supported_extensions = audio_extensions | video_extensions
    
    # Find all supported files
    files = [f for f in directory.rglob('*') 
             if f.suffix.lower() in supported_extensions]
    
    if not files:
        print(f"❌ No supported audio/video files found in {directory_path}")
        return
    
    print(f"📁 Found {len(files)} files to process")
    
    results = {}
    for i, file_path in enumerate(files, 1):
        print(f"\n--- Processing file {i}/{len(files)}: {file_path.name} ---")
        
        result = upload_and_process_file(str(file_path), **kwargs)
        results[str(file_path)] = result
    
    # Summary
    print(f"\n{'='*60}")
    print("BATCH PROCESSING SUMMARY")
    print(f"{'='*60}")
    
    successful = sum(1 for r in results.values() if r is not None)
    print(f"✅ Successful: {successful}/{len(files)}")
    print(f"❌ Failed: {len(files) - successful}/{len(files)}")
    
    return results

def get_processing_history():
    """Get and display processing history."""
    try:
        response = requests.get(f"{BASE_URL}/history")
        
        if response.status_code != 200:
            print(f"❌ Failed to get history: {response.json()}")
            return
        
        history = response.json()
        
        if not history:
            print("📋 No processing history found")
            return
        
        print(f"\n📋 Processing History ({len(history)} jobs)")
        print("-" * 80)
        
        for job in history:
            status_icon = {
                'completed': '✅',
                'processing': '⏳',
                'failed': '❌',
                'uploaded': '📤'
            }.get(job['status'], '❓')
            
            print(f"{status_icon} {job['original_filename']}")
            print(f"   Job ID: {job['job_id']}")
            print(f"   Status: {job['status']}")
            print(f"   Created: {job['created_at']}")
            if job['completed_at']:
                print(f"   Completed: {job['completed_at']}")
            print()
    
    except Exception as e:
        print(f"❌ Error getting history: {e}")

def main():
    """Main example function."""
    print("🎙️ Arabic STT Platform - API Examples")
    print("=" * 50)
    
    # Example 1: Single file processing
    print("\n1️⃣ Single File Processing Example")
    print("-" * 30)
    
    # You would replace this with an actual audio file path
    example_file = "example_audio.mp3"
    
    if Path(example_file).exists():
        results = upload_and_process_file(
            example_file,
            model_size="medium",
            language="ar",
            output_formats=["txt", "srt", "json"]
        )
        
        if results:
            print(f"📄 Results saved: {list(results.values())}")
    else:
        print(f"⚠️ Example file not found: {example_file}")
        print("   Place an audio file in the current directory to test")
    
    # Example 2: Get processing history
    print("\n2️⃣ Processing History")
    print("-" * 30)
    get_processing_history()
    
    # Example 3: Batch processing info
    print("\n3️⃣ Batch Processing")
    print("-" * 30)
    print("To process all files in a directory:")
    print("""
    from examples.api_usage import batch_process_directory
    
    results = batch_process_directory(
        "/path/to/audio/files",
        model_size="medium",
        language="ar",
        output_formats=["txt", "srt"]
    )
    """)

if __name__ == '__main__':
    main()
