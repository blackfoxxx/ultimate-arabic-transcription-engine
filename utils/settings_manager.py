"""
Settings Manager for persistent configuration storage
"""

import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from config import Config

logger = logging.getLogger(__name__)

class SettingsManager:
    """Manages persistent settings storage using FileManager."""
    
    def __init__(self, file_manager=None):
        """Initialize with optional file_manager dependency injection."""
        self.config = Config()
        self._file_manager = file_manager
        self._cache = {}
        self._load_default_settings()
    
    @property
    def file_manager(self):
        """Lazy load file_manager to avoid circular imports."""
        if self._file_manager is None:
            from utils.file_manager import FileManager
            self._file_manager = FileManager()
        return self._file_manager
    
    def _load_default_settings(self):
        """Load default settings from environment and config."""
        # Default settings with descriptions
        self.defaults = {
            'processing_mode': {
                'value': self.config.PROCESSING_MODE,
                'description': 'Default processing mode (local or api)',
                'encrypted': False
            },
            'openai_api_key': {
                'value': self.config.OPENAI_API_KEY,
                'description': 'OpenAI API key for cloud processing',
                'encrypted': True
            },
            'whisper_model_size': {
                'value': getattr(self.config, 'WHISPER_MODEL_SIZE', 'medium'),
                'description': 'Default Whisper model size for local processing',
                'encrypted': False
            },
            'openai_model': {
                'value': self.config.OPENAI_MODEL,
                'description': 'OpenAI model to use for API processing',
                'encrypted': False
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a setting value with caching."""
        # Check cache first
        if key in self._cache:
            return self._cache[key]
        
        # Get from persistent storage
        stored_value = self.file_manager.get_setting(key)
        
        if stored_value is not None:
            self._cache[key] = stored_value
            return stored_value
        
        # Check defaults
        if key in self.defaults:
            default_value = self.defaults[key]['value']
            if default_value:
                return default_value
        
        return default
    
    def set(self, key: str, value: str, description: Optional[str] = None) -> bool:
        """Set a setting value with automatic encryption for sensitive data."""
        # Determine if value should be encrypted
        encrypted = self._should_encrypt(key)
        
        # Use default description if not provided
        if description is None and key in self.defaults:
            description = self.defaults[key]['description']
        
        # Save to persistent storage
        success = self.file_manager.save_setting(key, value, encrypted=encrypted, description=description)
        
        if success:
            # Update cache
            self._cache[key] = value
            logger.info(f"Setting '{key}' updated successfully")
        
        return success
    
    def _should_encrypt(self, key: str) -> bool:
        """Determine if a setting should be encrypted."""
        sensitive_keys = ['api_key', 'secret', 'token', 'password', 'credential']
        return any(sensitive in key.lower() for sensitive in sensitive_keys)
    
    def get_all(self) -> Dict[str, Any]:
        """Get all settings with metadata."""
        return self.file_manager.get_all_settings()
    
    def delete(self, key: str) -> bool:
        """Delete a setting."""
        success = self.file_manager.delete_setting(key)
        if success and key in self._cache:
            del self._cache[key]
        return success
    
    def clear_cache(self):
        """Clear the settings cache."""
        self._cache.clear()
    
    def initialize_defaults(self) -> bool:
        """Initialize default settings if they don't exist."""
        try:
            for key, config in self.defaults.items():
                existing_value = self.file_manager.get_setting(key)
                if existing_value is None and config['value']:
                    self.set(key, config['value'], config['description'])
                    logger.info(f"Initialized default setting: {key}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize default settings: {str(e)}")
            return False
    
    def get_openai_api_key(self) -> Optional[str]:
        """Get the OpenAI API key, checking both persistent storage and environment."""
        # Check persistent storage first
        api_key = self.get('openai_api_key')
        if api_key:
            return api_key
        
        # Fallback to environment variable
        return os.environ.get('OPENAI_API_KEY')
    
    def set_openai_api_key(self, api_key: str) -> bool:
        """Set the OpenAI API key with validation."""
        if not api_key or not api_key.startswith('sk-'):
            logger.warning("Invalid OpenAI API key format")
            return False
        
        return self.set('openai_api_key', api_key, 'OpenAI API key for cloud processing')
    
    def get_processing_mode(self) -> str:
        """Get the current processing mode."""
        return self.get('processing_mode', 'local')
    
    def set_processing_mode(self, mode: str) -> bool:
        """Set the processing mode with validation."""
        if mode not in ['local', 'api']:
            logger.warning(f"Invalid processing mode: {mode}")
            return False
        
        return self.set('processing_mode', mode, 'Current processing mode')
    
    def is_api_configured(self) -> bool:
        """Check if OpenAI API is properly configured."""
        api_key = self.get_openai_api_key()
        return bool(api_key and api_key.startswith('sk-'))
    
    def validate_api_mode_switch(self) -> bool:
        """Validate if switching to API mode is possible."""
        return self.is_api_configured()
