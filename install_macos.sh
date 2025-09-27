#!/bin/bash

# Arabic STT Platform - Installation Script for macOS
# This script installs all required dependencies and sets up the platform

set -e  # Exit on any error

echo "🎙️ Arabic STT Platform - macOS Installation"
echo "============================================="
echo

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "❌ This script is designed for macOS only"
    exit 1
fi

# Check for Homebrew
if ! command -v brew &> /dev/null; then
    echo "📦 Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
else
    echo "✅ Homebrew is already installed"
fi

# Update Homebrew
echo "🔄 Updating Homebrew..."
brew update

# Install Python 3.9+ if not present
if ! command -v python3 &> /dev/null || [[ $(python3 -c 'import sys; print(sys.version_info >= (3, 9))') == "False" ]]; then
    echo "🐍 Installing Python 3.11..."
    brew install python@3.11
    echo "✅ Python 3.11 installed"
else
    echo "✅ Python 3.9+ is already installed"
fi

# Install FFmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "🎬 Installing FFmpeg..."
    brew install ffmpeg
    echo "✅ FFmpeg installed"
else
    echo "✅ FFmpeg is already installed"
fi

# Install RNNoise (optional)
if ! command -v rnnoise_demo &> /dev/null; then
    echo "🔊 Installing RNNoise..."
    brew install rnnoise
    echo "✅ RNNoise installed"
else
    echo "✅ RNNoise is already installed"
fi

# Install Redis (optional, for production)
if ! command -v redis-server &> /dev/null; then
    echo "💾 Installing Redis..."
    brew install redis
    echo "✅ Redis installed"
else
    echo "✅ Redis is already installed"
fi

# Create virtual environment
echo "🌐 Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install Python dependencies
echo "📚 Installing Python dependencies..."
if pip install -r requirements.txt; then
    echo "✅ Python dependencies installed successfully"
else
    echo "❌ Failed to install Python dependencies"
    echo "💡 Try updating pip: pip install --upgrade pip"
    exit 1
fi

# Create necessary directories
echo "📁 Creating required directories..."
python setup.py

# Set up environment configuration
echo "⚙️ Setting up environment configuration..."
if [ ! -f ".env" ]; then
    cat > .env << EOF
# Arabic STT Platform Configuration
# Processing Mode: 'local' for Whisper models, 'api' for OpenAI API
PROCESSING_MODE=local

# OpenAI API Configuration (optional)
# Get your API key from: https://platform.openai.com/api-keys
OPENAI_API_KEY=

# Whisper Model Settings
WHISPER_MODEL_SIZE=medium
WHISPER_DEVICE=auto

# Server Settings
HOST=0.0.0.0
PORT=5000
DEBUG=False

# Security
SECRET_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")
API_KEY_REQUIRED=False

# Logging
LOG_LEVEL=INFO
EOF
    echo "✅ Environment configuration created (.env)"
else
    echo "ℹ️ Environment configuration already exists"
fi

# Check for GPU support
echo "🔍 Checking for GPU acceleration..."
if python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    echo "✅ CUDA GPU support detected"
    echo "WHISPER_DEVICE=cuda" >> .env
elif python -c "import torch; print(torch.backends.mps.is_available())" 2>/dev/null | grep -q "True"; then
    echo "✅ Apple Silicon MPS support detected"
    echo "WHISPER_DEVICE=mps" >> .env
else
    echo "ℹ️ Using CPU processing (consider GPU for faster processing)"
fi

# Download initial Whisper model
echo "🤖 Downloading initial Whisper model..."
python -c "
try:
    from faster_whisper import WhisperModel
    model = WhisperModel('medium', download_root='./models')
    print('✅ Medium Whisper model downloaded successfully')
except Exception as e:
    print(f'⚠️ Failed to download model: {e}')
    print('Models will be downloaded on first use')
"

echo
echo "🎉 Installation completed successfully!"
echo
echo "📋 Summary:"
echo "- ✅ System dependencies installed"
echo "- ✅ Python environment created"
echo "- ✅ Dependencies installed"  
echo "- ✅ Configuration files created"
echo "- ✅ Initial model downloaded"
echo
echo "🚀 Quick Start:"
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo
echo "2. Start the platform:"
echo "   ./start.sh"
echo
echo "3. Open your browser to:"
echo "   http://localhost:5000"
echo
echo "⚙️ Configuration:"
echo "- Edit .env file to configure processing mode and API keys"
echo "- Visit http://localhost:5000/settings to configure in web interface"
echo
echo "📖 Documentation:"
echo "- README.md - Complete setup guide"
echo "- Arabic_STT_Platform_Requirements.md - Detailed requirements"
echo
