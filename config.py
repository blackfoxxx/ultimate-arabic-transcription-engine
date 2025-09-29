"""
Configuration settings for the Arabic STT Platform
"""

import os
from pathlib import Path

class Config:
    """Application configuration."""
     # Large file processing optimizations
    LARGE_FILE_OPTIMIZATIONS = {
        'chunk_processing_threshold_mb': 50,  # Files >50MB use chunked processing
        'memory_cleanup_threshold_mb': 100,   # Force cleanup for files >100MB
        'timeout_multiplier': 3,              # Dynamic timeout = estimate * 3
        'large_file_timeout_bonus': 2,        # +2 seconds per MB for >100MB files
        'reduced_beam_size_threshold_mb': 100, # Reduce beam size for very large files
        'vad_silence_threshold_large_ms': 1000 # Longer silence detection for large files
    }
    
    # Aya Arabic Enhancement settings
    AYA_ENHANCEMENT_ENABLED = os.environ.get('AYA_ENHANCEMENT_ENABLED', 'True').lower() == 'true'
    AYA_MODEL_NAME = os.environ.get('AYA_MODEL_NAME', 'aya:8b')
    AYA_OLLAMA_HOST = os.environ.get('AYA_OLLAMA_HOST', 'http://localhost:11434')
    AYA_REPLACE_ORIGINAL = os.environ.get('AYA_REPLACE_ORIGINAL', 'True').lower() == 'true'
    AYA_CHUNK_SIZE = int(os.environ.get('AYA_CHUNK_SIZE', '5'))  # Segments per chunk
    AYA_TIMEOUT_SECONDS = int(os.environ.get('AYA_TIMEOUT_SECONDS', '600'))  # 10 minutes
    
    # Enhanced Arabic Transcription settings
    ENHANCED_ARABIC_ENABLED = os.environ.get('ENHANCED_ARABIC_ENABLED', 'True').lower() == 'true'
    ENHANCED_ARABIC_PREPROCESSING = os.environ.get('ENHANCED_ARABIC_PREPROCESSING', 'True').lower() == 'true'
    ENHANCED_ARABIC_MODEL_SIZE = os.environ.get('ENHANCED_ARABIC_MODEL_SIZE', 'medium')  # Recommended: medium or large-v2
    
    # Advanced Arabic Transcription settings v2.0
    ADVANCED_ARABIC_ENABLED = os.environ.get('ADVANCED_ARABIC_ENABLED', 'True').lower() == 'true'
    ADVANCED_ARABIC_PREPROCESSING = os.environ.get('ADVANCED_ARABIC_PREPROCESSING', 'True').lower() == 'true'
    ADVANCED_ARABIC_MODEL_SIZE = os.environ.get('ADVANCED_ARABIC_MODEL_SIZE', 'large-v2')  # Use larger model for better quality
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'arabic-stt-platform-secret-key-change-in-production'
    DEBUG = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    # Server settings
    HOST = os.environ.get('HOST', '0.0.0.0')
    PORT = int(os.environ.get('PORT', 5002))
    
    # File storage
    BASE_DIR = Path(__file__).parent
    UPLOAD_FOLDER = BASE_DIR / 'uploads'
    RESULTS_FOLDER = BASE_DIR / 'results' 
    TEMP_FOLDER = BASE_DIR / 'temp'
    MODELS_FOLDER = BASE_DIR / 'models'
    
    # File size limits (in bytes)
    MAX_CONTENT_LENGTH = 5 * 1024 * 1024 * 1024  # 5GB max file size (increased for large audio files)
    
    # Audio processing settings
    WHISPER_MODEL_SIZE = os.environ.get('WHISPER_MODEL_SIZE', 'medium')
    WHISPER_DEVICE = os.environ.get('WHISPER_DEVICE', 'auto')  # auto, cpu, cuda
    ENABLE_RNNOISE = os.environ.get('ENABLE_RNNOISE', 'True').lower() == 'true'
    
    # API settings
    API_KEY_REQUIRED = os.environ.get('API_KEY_REQUIRED', 'False').lower() == 'true'
    API_KEY = os.environ.get('API_KEY', 'arabic-stt-api-key')
    RATE_LIMIT = os.environ.get('RATE_LIMIT', '100/hour')
    
    # Logging
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FILE = BASE_DIR / 'logs' / 'arabic_stt.log'
    
    # Redis (for task queuing)
    REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
    
    # Supported languages
    SUPPORTED_LANGUAGES = {
        'ar': 'Arabic',
        'en': 'English',
        'auto': 'Auto-detect'
    }
    
    # Processing mode options
    PROCESSING_MODE = os.environ.get('PROCESSING_MODE', 'local')  # 'local' or 'api'
    
    # OpenAI API settings
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')
    OPENAI_MODEL = os.environ.get('OPENAI_MODEL', 'whisper-1')
    OPENAI_BASE_URL = os.environ.get('OPENAI_BASE_URL', 'https://api.openai.com/v1')
    
    # Whisper model sizes
    WHISPER_MODELS = {
        'tiny': 'Fastest, lowest accuracy',
        'base': 'Fast, good for real-time',
        'small': 'Balanced speed/accuracy', 
        'medium': 'Good accuracy (recommended)',
        'large': 'Best accuracy, slower'
    }
    
    # Processing modes
    PROCESSING_MODES = {
        'local': 'Local Whisper Models',
        'api': 'OpenAI API (Cloud)'
    }
    
    # Output formats
    OUTPUT_FORMATS = {
        'txt': 'Plain text transcript',
        'srt': 'SubRip subtitles',
        'vtt': 'WebVTT subtitles',
        'json': 'Structured JSON with timestamps'
    }
    
    # Audio processing options
    NOISE_REDUCTION_OPTIONS = {
        'none': 'No noise reduction',
        'auto': 'Automatic detection',
        'rnnoise': 'RNNoise AI model',
        'traditional': 'Traditional filters'
    }
    
    # LLM Integration settings
    LLM_ENABLED = os.environ.get('LLM_ENABLED', 'True').lower() == 'true'
    LLM_BACKEND = os.environ.get('LLM_BACKEND', 'ollama')  # 'ollama', 'transformers', 'openai_compatible'
    LLM_MODEL = os.environ.get('LLM_MODEL', 'llama3.2:3b')
    LLM_SERVER_URL = os.environ.get('LLM_SERVER_URL', 'http://localhost:11434')
    ARABIC_LLM_MODEL = os.environ.get('ARABIC_LLM_MODEL', 'aya:35b')
    LLM_MAX_TOKENS = int(os.environ.get('LLM_MAX_TOKENS', '4096'))
    LLM_TEMPERATURE = float(os.environ.get('LLM_TEMPERATURE', '0.7'))
    
    # LLM Features
    LLM_FEATURES = {
        'text_enhancement': 'Text correction and improvement',
        'sentiment_analysis': 'Emotion and sentiment detection',
        'summarization': 'Content summarization',
        'entity_extraction': 'Named entity recognition',
        'topic_analysis': 'Topic modeling and categorization',
        'keyword_extraction': 'Important terms identification'
    }
    
    # Text Enhancement Options
    ENHANCEMENT_OPTIONS = {
        'grammar': 'Grammar correction',
        'diacritization': 'Add Arabic diacritics',
        'normalization': 'Dialect to MSA conversion',
        'punctuation': 'Punctuation improvement',
        'spelling': 'Spelling correction',
        'style': 'Style enhancement'
    }
    
    # Analysis Options
    ANALYSIS_OPTIONS = {
        'sentiment': 'Sentiment analysis',
        'emotion': 'Emotion detection',
        'topics': 'Topic identification',
        'entities': 'Named entity recognition',
        'keywords': 'Keyword extraction',
        'complexity': 'Text complexity analysis'
    }
    
    # Arabic-Specific Transcription Optimizations
    ARABIC_OPTIMIZATIONS = {
        'default_model': 'medium',  # Minimum recommended for Arabic
        'language': 'ar',
        'initial_prompt': 'هذا تسجيل صوتي باللغة العربية. الرجاء كتابة النص بدقة مع علامات الترقيم المناسبة.',
        'beam_size': 5,
        'best_of': 5,
        'temperature': [0.0, 0.2, 0.4],  # Conservative for better accuracy
        'condition_on_previous_text': True,
        'vad_filter': True,
        'vad_parameters': {
            'min_silence_duration_ms': 500,
            'speech_pad_ms': 400
        },
        'enable_noise_reduction': True,
        'word_timestamps': True
    }
    
    # Large File Processing Optimizations
    LARGE_FILE_OPTIMIZATIONS = {
        'chunk_processing_threshold_mb': 50,  # Files >50MB use chunked processing
        'memory_cleanup_threshold_mb': 100,   # Force cleanup for files >100MB
        'timeout_multiplier': 3,              # Dynamic timeout = estimate * 3
        'large_file_timeout_bonus': 2,        # +2 seconds per MB for >100MB files
        'reduced_beam_size_threshold_mb': 100, # Reduce beam size for very large files
        'vad_silence_threshold_large_ms': 1000 # Longer silence detection for large files
    }

class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'

class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    API_KEY_REQUIRED = True
    
class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True
    DEBUG = True
