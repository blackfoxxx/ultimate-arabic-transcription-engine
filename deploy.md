# Arabic STT Platform - Deployment Guide

## Prerequisites Check ✅

- ✅ **Dependencies**: All required packages are listed in `requirements.txt`
- ✅ **Configuration**: `config.py` is properly configured with environment variables
- ✅ **Docker Files**: `Dockerfile`, `docker-compose.yml`, `nginx.conf`, and `start.sh` are ready
- ✅ **Application**: Flask app starts successfully (tested locally)

## Deployment Options

### Option 1: Docker Deployment (Recommended)

#### Prerequisites
- Docker Desktop must be running
- At least 8GB RAM available
- 10GB+ free disk space (for models)

#### Steps
1. **Start Docker Desktop** (currently not running)
2. **Build the image**:
   ```bash
   docker build -t arabic-stt-platform .
   ```

3. **Deploy with docker-compose**:
   ```bash
   docker-compose up -d
   ```

4. **Access the application**:
   - Main app: http://localhost:5000
   - With nginx proxy: http://localhost:80

#### Services Included
- **arabic-stt**: Main Flask application
- **redis**: For caching and session management
- **nginx**: Reverse proxy for production

### Option 2: Local Python Deployment

#### Prerequisites
- Python 3.11+
- FFmpeg installed
- 8GB+ RAM
- CUDA-compatible GPU (optional, for better performance)

#### Steps
1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set environment variables** (optional):
   ```bash
   set WHISPER_DEVICE=cuda  # For GPU acceleration
   set DEBUG=False          # For production
   set HOST=0.0.0.0        # To accept external connections
   set PORT=5000           # Default port
   ```

3. **Run the application**:
   ```bash
   python app.py
   ```

4. **Access**: http://localhost:5002 (default port from config)

### Option 3: Production Deployment

#### For VPS/Cloud Deployment
1. **Upload project files** to your server
2. **Install Docker** on the server
3. **Configure environment variables**:
   ```bash
   export WHISPER_DEVICE=cpu  # Unless GPU available
   export DEBUG=False
   export API_KEY_REQUIRED=True
   export API_KEY=your-secure-api-key
   ```

4. **Deploy with docker-compose**:
   ```bash
   docker-compose up -d
   ```

5. **Configure SSL** (recommended):
   - Uncomment HTTPS section in `nginx.conf`
   - Add SSL certificates to `./ssl/` directory
   - Update docker-compose volumes for SSL

## Configuration Notes

### Environment Variables
- `WHISPER_MODEL_SIZE`: tiny, base, small, medium, large (default: medium)
- `WHISPER_DEVICE`: auto, cpu, cuda (default: auto)
- `PROCESSING_MODE`: local, api (default: local)
- `OPENAI_API_KEY`: Required if using API mode
- `DEBUG`: True/False (default: False in production)

### Hardware Requirements
- **Minimum**: 4GB RAM, 2 CPU cores
- **Recommended**: 8GB+ RAM, 4+ CPU cores
- **Optimal**: 16GB+ RAM, GPU with 8GB+ VRAM

### Model Downloads
- Models are downloaded automatically on first use
- Large models (3GB+) require stable internet
- Models are cached in `~/.cache/whisper/`

## Troubleshooting

### Common Issues
1. **Docker not running**: Start Docker Desktop
2. **Port conflicts**: Change ports in `docker-compose.yml`
3. **Memory issues**: Use smaller Whisper models
4. **GPU not detected**: Install CUDA drivers

### Performance Optimization
- Use GPU acceleration when available
- Choose appropriate Whisper model size
- Enable Redis for caching
- Use nginx for load balancing

## Security Considerations
- Set strong API keys in production
- Use HTTPS in production
- Limit file upload sizes
- Monitor resource usage
- Regular security updates

## Next Steps
1. Start Docker Desktop
2. Run: `docker-compose up -d`
3. Access: http://localhost:5000
4. Upload audio files for transcription

The application is ready for deployment! 🚀