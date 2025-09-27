"""
Test suite for Arabic STT Platform
"""

import pytest
import tempfile
import os
from pathlib import Path
import json

# Test the configuration
def test_config_loading():
    """Test configuration loading."""
    from config import Config
    config = Config()
    
    assert config.WHISPER_MODEL_SIZE == 'medium'
    assert config.HOST == '0.0.0.0'
    assert config.PORT == 5000

# Test file manager
def test_file_manager():
    """Test file manager functionality."""
    from utils.file_manager import FileManager
    
    # Create temporary database for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Mock config for testing
        class TestConfig:
            BASE_DIR = Path(temp_dir)
            UPLOAD_FOLDER = Path(temp_dir) / 'uploads'
            RESULTS_FOLDER = Path(temp_dir) / 'results'
            TEMP_FOLDER = Path(temp_dir) / 'temp'
            MODELS_FOLDER = Path(temp_dir) / 'models'
            LOG_FILE = Path(temp_dir) / 'logs' / 'test.log'
        
        # Test initialization
        file_manager = FileManager()
        file_manager.config = TestConfig()
        file_manager.db_path = TestConfig.BASE_DIR / 'test.db'
        file_manager._init_database()
        
        # Test job ID validation
        assert not file_manager.validate_job_id('invalid-id')
        
        # Test directory creation
        file_manager.ensure_directories()
        assert TestConfig.UPLOAD_FOLDER.exists()
        assert TestConfig.RESULTS_FOLDER.exists()

# Test audio processor
def test_audio_processor():
    """Test audio processor initialization."""
    from core.audio_processor import AudioProcessor
    
    processor = AudioProcessor()
    assert processor is not None
    
    # Test audio validation methods exist
    assert hasattr(processor, 'validate_audio_file')
    assert hasattr(processor, 'get_audio_info')

# Test transcription engine
def test_transcription_engine():
    """Test transcription engine initialization."""
    from core.transcription_engine import TranscriptionEngine
    
    engine = TranscriptionEngine()
    assert engine is not None
    assert engine.device in ['cpu', 'cuda']

# Test output generator
def test_output_generator():
    """Test output generator."""
    from core.output_generator import OutputGenerator
    
    generator = OutputGenerator()
    assert generator is not None
    
    # Test text cleaning
    test_text = "  This is    a test   with   extra   spaces  "
    cleaned = generator._clean_text(test_text)
    assert cleaned == "This is a test with extra spaces"

# Test logger setup
def test_logger_setup():
    """Test logger configuration."""
    from utils.logger import setup_logger
    
    logger = setup_logger('test')
    assert logger is not None
    assert logger.name == 'test'

if __name__ == '__main__':
    pytest.main([__file__])
