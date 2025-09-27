#!/usr/bin/env python3
"""
Arabic STT CLI - Ultimate Quality Version
========================================

Command-line interface for high-quality Arabic speech-to-text transcription
using the Ultimate Arabic Engine v3.0 with progressive quality optimization.

Usage Examples:
    python arabic_cli_ultimate.py --file audio.wav --output transcript.txt
    python arabic_cli_ultimate.py --file audio.mp3 --model large-v2 --engine ultimate
    python arabic_cli_ultimate.py --file audio.wav --compare-engines --output-dir results/
    python arabic_cli_ultimate.py --batch-process *.wav --engine ultimate --format json
"""

import argparse
import os
import sys
import time
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass

# Set up Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import our engines
try:
    from core.ultimate_arabic_transcription_engine import UltimateArabicTranscriptionEngine
    ULTIMATE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Ultimate Arabic Engine not available: {e}")
    ULTIMATE_AVAILABLE = False

try:
    from core.advanced_arabic_transcription_engine import AdvancedArabicTranscriptionEngine
    ADVANCED_AVAILABLE = True
except ImportError:
    ADVANCED_AVAILABLE = False

try:
    from core.enhanced_arabic_transcription_engine import EnhancedArabicTranscriptionEngine
    ENHANCED_AVAILABLE = True
except ImportError:
    ENHANCED_AVAILABLE = False

try:
    from faster_whisper import WhisperModel
    STANDARD_AVAILABLE = True
except ImportError:
    STANDARD_AVAILABLE = False

@dataclass
class ProcessingResult:
    """Results from Arabic transcription processing"""
    engine: str
    model_size: str
    processing_time: float
    text: str
    quality_metrics: Dict[str, Any]
    segments: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class ArabicSTTCLI:
    """Command-line interface for Arabic STT with multiple engines"""
    
    def __init__(self):
        self.setup_logging()
        self.engines = {}
        self.results = []
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def initialize_engines(self, model_size: str = "large-v2"):
        """Initialize available transcription engines"""
        print(f"🔧 Initializing Arabic transcription engines (model: {model_size})...")
        
        # Ultimate Arabic Engine v3.0 (highest priority)
        if ULTIMATE_AVAILABLE:
            try:
                print("📥 Loading Ultimate Arabic Engine v3.0...")
                engine = UltimateArabicTranscriptionEngine(model_size=model_size, device="cpu")
                if engine.initialize_model():
                    self.engines['ultimate'] = engine
                    print("✅ Ultimate Arabic Engine v3.0 ready - Maximum Quality Mode")
                else:
                    print("❌ Ultimate Arabic Engine v3.0 model initialization failed")
            except Exception as e:
                print(f"❌ Ultimate Arabic Engine v3.0 failed: {e}")
        
        # Advanced Arabic Engine v2.0
        if ADVANCED_AVAILABLE:
            try:
                print("📥 Loading Advanced Arabic Engine v2.0...")
                engine = AdvancedArabicTranscriptionEngine(model_size=model_size, device="cpu")
                self.engines['advanced'] = engine
                print("✅ Advanced Arabic Engine v2.0 ready")
            except Exception as e:
                print(f"❌ Advanced Arabic Engine v2.0 failed: {e}")
        
        # Enhanced Arabic Engine v1.0
        if ENHANCED_AVAILABLE:
            try:
                print("📥 Loading Enhanced Arabic Engine v1.0...")
                engine = EnhancedArabicTranscriptionEngine()
                self.engines['enhanced'] = engine
                print("✅ Enhanced Arabic Engine v1.0 ready")
            except Exception as e:
                print(f"❌ Enhanced Arabic Engine v1.0 failed: {e}")
        
        # Standard Whisper
        if STANDARD_AVAILABLE:
            try:
                print(f"📥 Loading Standard Whisper ({model_size})...")
                engine = WhisperModel(
                    model_size, 
                    device="cpu",
                    compute_type="int8"
                )
                self.engines['standard'] = engine
                print("✅ Standard Whisper ready")
            except Exception as e:
                print(f"❌ Standard Whisper failed: {e}")
        
        if not self.engines:
            print("❌ No transcription engines available!")
            sys.exit(1)
        
        print(f"🚀 Initialized {len(self.engines)} engines: {list(self.engines.keys())}")
    
    def validate_audio_file(self, file_path: str) -> bool:
        """Validate audio file exists and is supported format"""
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return False
        
        supported_formats = ['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac', '.aiff']
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext not in supported_formats:
            print(f"❌ Unsupported format: {file_ext}")
            print(f"✅ Supported formats: {', '.join(supported_formats)}")
            return False
        
        # Check file size
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        print(f"📁 File: {os.path.basename(file_path)} ({file_size:.1f} MB)")
        
        # Estimate duration if possible
        try:
            import librosa
            duration = librosa.get_duration(path=file_path)
            print(f"⏱️  Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        except:
            pass
        
        return True
    
    async def transcribe_with_ultimate(self, file_path: str) -> ProcessingResult:
        """Transcribe using Ultimate Arabic Engine v3.0"""
        engine = self.engines['ultimate']
        start_time = time.time()
        
        print("🔥 Processing with Ultimate Arabic Engine v3.0...")
        result = engine.transcribe(file_path)
        
        processing_time = time.time() - start_time
        
        if 'error' in result:
            raise Exception(f"Ultimate Arabic Engine error: {result['error']}")
        
        return ProcessingResult(
            engine="ultimate_v3",
            model_size=engine.model_size,
            processing_time=processing_time,
            text=result['transcript']['full_text'],
            quality_metrics=result.get('quality_metrics', {}),
            segments=result['transcript'].get('segments', []),
            metadata=result.get('metadata', {})
        )
    
    async def transcribe_with_advanced(self, file_path: str) -> ProcessingResult:
        """Transcribe using Advanced Arabic Engine v2.0"""
        engine = self.engines['advanced']
        start_time = time.time()
        
        print("🚀 Processing with Advanced Arabic Engine v2.0...")
        result = await engine.transcribe_arabic_advanced(
            audio_path=file_path,
            model_size=engine.model_size,
            enable_preprocessing=True
        )
        
        processing_time = time.time() - start_time
        
        return ProcessingResult(
            engine="advanced_v2",
            model_size=engine.model_size,
            processing_time=processing_time,
            text=result['transcript']['full_text'],
            quality_metrics=result.get('quality_metrics', {}),
            segments=result['transcript'].get('segments', []),
            metadata=result.get('metadata', {})
        )
    
    async def transcribe_with_enhanced(self, file_path: str) -> ProcessingResult:
        """Transcribe using Enhanced Arabic Engine v1.0"""
        engine = self.engines['enhanced']
        start_time = time.time()
        
        print("🎯 Processing with Enhanced Arabic Engine v1.0...")
        result = await engine.transcribe_with_enhanced_arabic(
            audio_path=file_path,
            enable_preprocessing=True
        )
        
        processing_time = time.time() - start_time
        
        return ProcessingResult(
            engine="enhanced_v1",
            model_size="medium",
            processing_time=processing_time,
            text=result['transcript']['full_text'],
            quality_metrics=result.get('quality_metrics', {}),
            segments=result['transcript'].get('segments', []),
            metadata=result.get('metadata', {})
        )
    
    async def transcribe_with_standard(self, file_path: str, model_size: str) -> ProcessingResult:
        """Transcribe using Standard Whisper"""
        engine = self.engines['standard']
        start_time = time.time()
        
        print("📝 Processing with Standard Whisper...")
        
        # Standard Whisper transcription
        segments, info = engine.transcribe(
            file_path,
            language="ar",
            beam_size=5,
            best_of=5,
            temperature=0.0
        )
        
        processing_time = time.time() - start_time
        
        # Process segments
        transcript_segments = []
        full_text = ""
        
        for segment in segments:
            transcript_segments.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip(),
                "confidence": getattr(segment, 'avg_logprob', 0.0)
            })
            full_text += segment.text.strip() + " "
        
        full_text = full_text.strip()
        
        # Calculate basic quality metrics
        import re
        arabic_chars = len(re.findall(r'[\u0600-\u06FF]', full_text))
        total_chars = len(re.sub(r'\s', '', full_text))
        arabic_ratio = arabic_chars / max(total_chars, 1)
        
        quality_metrics = {
            "arabic_char_ratio": arabic_ratio,
            "quality_score": arabic_ratio * 0.5,  # Basic estimate
            "confidence_avg": info.language_probability,
            "language_purity": arabic_ratio,
            "engine": "standard_whisper"
        }
        
        return ProcessingResult(
            engine="standard",
            model_size=model_size,
            processing_time=processing_time,
            text=full_text,
            quality_metrics=quality_metrics,
            segments=transcript_segments,
            metadata={
                "language": info.language,
                "language_probability": info.language_probability,
                "duration": info.duration
            }
        )
    
    async def process_single_file(self, file_path: str, engine_name: str = "ultimate") -> ProcessingResult:
        """Process single audio file with specified engine"""
        if not self.validate_audio_file(file_path):
            raise ValueError(f"Invalid audio file: {file_path}")
        
        print(f"\n🎵 Processing: {os.path.basename(file_path)}")
        print("=" * 60)
        
        # Select and run engine
        if engine_name == "ultimate" and "ultimate" in self.engines:
            result = await self.transcribe_with_ultimate(file_path)
        elif engine_name == "advanced" and "advanced" in self.engines:
            result = await self.transcribe_with_advanced(file_path)
        elif engine_name == "enhanced" and "enhanced" in self.engines:
            result = await self.transcribe_with_enhanced(file_path)
        elif engine_name == "standard" and "standard" in self.engines:
            result = await self.transcribe_with_standard(file_path, "large-v2")
        else:
            # Default to best available
            if "ultimate" in self.engines:
                result = await self.transcribe_with_ultimate(file_path)
            elif "advanced" in self.engines:
                result = await self.transcribe_with_advanced(file_path)
            elif "enhanced" in self.engines:
                result = await self.transcribe_with_enhanced(file_path)
            elif "standard" in self.engines:
                result = await self.transcribe_with_standard(file_path, "large-v2")
            else:
                raise Exception("No engines available")
        
        return result
    
    async def compare_engines(self, file_path: str) -> List[ProcessingResult]:
        """Compare transcription quality across all available engines"""
        if not self.validate_audio_file(file_path):
            raise ValueError(f"Invalid audio file: {file_path}")
        
        print(f"\n🔄 Comparing engines for: {os.path.basename(file_path)}")
        print("=" * 60)
        
        results = []
        
        # Test each available engine
        for engine_name in self.engines.keys():
            try:
                print(f"\n--- Testing {engine_name.upper()} engine ---")
                
                if engine_name == "ultimate":
                    result = await self.transcribe_with_ultimate(file_path)
                elif engine_name == "advanced":
                    result = await self.transcribe_with_advanced(file_path)
                elif engine_name == "enhanced":
                    result = await self.transcribe_with_enhanced(file_path)
                elif engine_name == "standard":
                    result = await self.transcribe_with_standard(file_path, "large-v2")
                else:
                    continue
                
                results.append(result)
                
                # Show quick summary
                quality = result.quality_metrics.get('quality_score', 0)
                purity = result.quality_metrics.get('language_purity', 0)
                print(f"✅ {engine_name.upper()}: Quality {quality:.3f} | Purity {purity:.3f} | Time {result.processing_time:.1f}s")
                
            except Exception as e:
                print(f"❌ {engine_name.upper()} failed: {e}")
                continue
        
        return results
    
    def save_results(self, results: List[ProcessingResult], output_path: str, format_type: str = "txt"):
        """Save transcription results to file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format_type == "json":
            # Save as JSON with full details
            json_data = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_engines": len(results),
                "results": []
            }
            
            for result in results:
                json_data["results"].append({
                    "engine": result.engine,
                    "model_size": result.model_size,
                    "processing_time": result.processing_time,
                    "text": result.text,
                    "quality_metrics": result.quality_metrics,
                    "segments_count": len(result.segments),
                    "metadata": result.metadata
                })
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
            
        else:
            # Save as text with comparison
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"Arabic STT CLI Results\n")
                f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Engines: {len(results)}\n")
                f.write("=" * 80 + "\n\n")
                
                # Sort by quality score
                sorted_results = sorted(results, key=lambda x: x.quality_metrics.get('quality_score', 0), reverse=True)
                
                for i, result in enumerate(sorted_results, 1):
                    f.write(f"RESULT #{i}: {result.engine.upper()}\n")
                    f.write(f"Model: {result.model_size}\n")
                    f.write(f"Processing Time: {result.processing_time:.2f} seconds\n")
                    
                    # Quality metrics
                    quality = result.quality_metrics
                    f.write(f"Quality Score: {quality.get('quality_score', 0):.3f}\n")
                    f.write(f"Arabic Purity: {quality.get('language_purity', 0):.3f}\n")
                    f.write(f"Confidence: {quality.get('confidence_avg', 0):.3f}\n")
                    
                    f.write(f"\nTranscript:\n")
                    f.write(f"{result.text}\n")
                    f.write("\n" + "=" * 80 + "\n\n")
        
        print(f"📄 Results saved to: {output_path}")
    
    def print_quality_comparison(self, results: List[ProcessingResult]):
        """Print detailed quality comparison"""
        if not results:
            print("No results to compare")
            return
        
        print(f"\n📊 QUALITY COMPARISON ({len(results)} engines)")
        print("=" * 80)
        
        # Sort by quality score
        sorted_results = sorted(results, key=lambda x: x.quality_metrics.get('quality_score', 0), reverse=True)
        
        # Header
        print(f"{'Engine':<15} {'Quality':<8} {'Purity':<8} {'Confidence':<10} {'Time':<6} {'Words':<6}")
        print("-" * 80)
        
        # Results
        for result in sorted_results:
            quality = result.quality_metrics
            word_count = len(result.text.split())
            
            print(f"{result.engine:<15} "
                  f"{quality.get('quality_score', 0):.3f}    "
                  f"{quality.get('language_purity', 0):.3f}    "
                  f"{quality.get('confidence_avg', 0):.3f}      "
                  f"{result.processing_time:.1f}s   "
                  f"{word_count}")
        
        print("-" * 80)
        
        # Best result
        best = sorted_results[0]
        print(f"\n🏆 BEST ENGINE: {best.engine.upper()}")
        print(f"Quality Score: {best.quality_metrics.get('quality_score', 0):.3f}")
        print(f"Processing Time: {best.processing_time:.2f} seconds")
        print(f"\nBest Transcript Preview:")
        preview = best.text[:200] + "..." if len(best.text) > 200 else best.text
        print(f"📝 {preview}")
    
    def print_help(self):
        """Print usage help"""
        print("""
🎤 Arabic STT CLI - Ultimate Quality Version
==========================================

BASIC USAGE:
    python arabic_cli_ultimate.py --file audio.wav
    python arabic_cli_ultimate.py --file audio.mp3 --output transcript.txt

ENGINE SELECTION:
    --engine ultimate    # Ultimate Arabic v3.0 (highest quality)
    --engine advanced    # Advanced Arabic v2.0
    --engine enhanced    # Enhanced Arabic v1.0
    --engine standard    # Standard Whisper

COMPARISON MODE:
    --compare-engines    # Test all available engines

OUTPUT OPTIONS:
    --output file.txt    # Save transcript as text
    --output file.json   # Save detailed results as JSON
    --format txt|json    # Output format (default: txt)

MODEL SIZE:
    --model small|medium|large-v2    # Whisper model size

BATCH PROCESSING:
    --batch-dir /path/to/audio/files/
    --batch-pattern "*.wav"

EXAMPLES:
    # Single file with Ultimate engine
    python arabic_cli_ultimate.py --file sample.wav --engine ultimate
    
    # Compare all engines
    python arabic_cli_ultimate.py --file sample.wav --compare-engines
    
    # Batch processing
    python arabic_cli_ultimate.py --batch-dir ./audio/ --engine ultimate
    
    # Save detailed JSON results
    python arabic_cli_ultimate.py --file sample.wav --compare-engines --output results.json --format json
""")

async def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="Arabic STT CLI - Ultimate Quality Version",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Input options
    parser.add_argument('--file', '-f', type=str, help='Audio file to transcribe')
    parser.add_argument('--batch-dir', type=str, help='Directory for batch processing')
    parser.add_argument('--batch-pattern', type=str, default='*.wav', help='File pattern for batch processing')
    
    # Engine options
    parser.add_argument('--engine', '-e', type=str, 
                       choices=['ultimate', 'advanced', 'enhanced', 'standard', 'auto'],
                       default='ultimate', help='Transcription engine to use')
    parser.add_argument('--compare-engines', '-c', action='store_true',
                       help='Compare all available engines')
    parser.add_argument('--model', '-m', type=str, default='large-v2',
                       choices=['tiny', 'base', 'small', 'medium', 'large', 'large-v2'],
                       help='Whisper model size')
    
    # Output options
    parser.add_argument('--output', '-o', type=str, help='Output file path')
    parser.add_argument('--output-dir', type=str, help='Output directory for batch processing')
    parser.add_argument('--format', type=str, choices=['txt', 'json'], default='txt',
                       help='Output format')
    
    # Utility options
    parser.add_argument('--help-examples', action='store_true', help='Show usage examples')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Create CLI instance
    cli = ArabicSTTCLI()
    
    if args.help_examples:
        cli.print_help()
        return
    
    if not args.file and not args.batch_dir:
        print("❌ Error: Must specify --file or --batch-dir")
        print("Use --help-examples for usage information")
        return
    
    try:
        # Initialize engines
        cli.initialize_engines(model_size=args.model)
        
        results = []
        
        if args.file:
            # Single file processing
            if args.compare_engines:
                print(f"🔄 Comparing all engines for: {args.file}")
                results = await cli.compare_engines(args.file)
            else:
                print(f"🎵 Processing with {args.engine} engine: {args.file}")
                result = await cli.process_single_file(args.file, args.engine)
                results = [result]
            
            # Show quality comparison
            if len(results) > 1:
                cli.print_quality_comparison(results)
            else:
                result = results[0]
                print(f"\n✅ PROCESSING COMPLETE")
                print(f"Engine: {result.engine}")
                print(f"Quality Score: {result.quality_metrics.get('quality_score', 0):.3f}")
                print(f"Processing Time: {result.processing_time:.2f} seconds")
                print(f"Word Count: {len(result.text.split())}")
                
                # Show transcript preview
                preview = result.text[:300] + "..." if len(result.text) > 300 else result.text
                print(f"\n📝 Transcript Preview:\n{preview}")
        
        elif args.batch_dir:
            # Batch processing
            import glob
            
            pattern = os.path.join(args.batch_dir, args.batch_pattern)
            audio_files = glob.glob(pattern)
            
            if not audio_files:
                print(f"❌ No files found matching: {pattern}")
                return
            
            print(f"📁 Batch processing {len(audio_files)} files...")
            
            for i, file_path in enumerate(audio_files, 1):
                print(f"\n[{i}/{len(audio_files)}] Processing: {os.path.basename(file_path)}")
                try:
                    result = await cli.process_single_file(file_path, args.engine)
                    results.append(result)
                    
                    # Auto-save individual results
                    if args.output_dir:
                        output_name = f"{Path(file_path).stem}_transcript.{args.format}"
                        output_path = os.path.join(args.output_dir, output_name)
                        cli.save_results([result], output_path, args.format)
                
                except Exception as e:
                    print(f"❌ Failed to process {file_path}: {e}")
                    continue
            
            print(f"\n🎉 Batch processing complete! Processed {len(results)}/{len(audio_files)} files")
        
        # Save results if output specified
        if args.output and results:
            cli.save_results(results, args.output, args.format)
        
        print(f"\n🚀 Arabic STT CLI processing complete!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
