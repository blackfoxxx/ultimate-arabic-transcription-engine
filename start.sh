#!/bin/bash

# Start script for Arabic STT Platform
# This script initializes the application with proper environment setup

echo "Starting Arabic STT Platform..."

# Create necessary directories if they don't exist
mkdir -p uploads results temp models logs

# Set proper permissions
chmod 755 uploads results temp models logs

# Export environment variables
export FLASK_APP=app.py
export PYTHONPATH=/app

# Start the Flask application
echo "Launching Flask application on port 5000..."
python app.py