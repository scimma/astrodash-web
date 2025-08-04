# AstroDash Production Backend

A production-grade FastAPI backend for astronomical spectrum classification with support for DASH (CNN) and Transformer models.

## 🚀 Features

- **Spectrum Classification**: Support for DASH and Transformer models
- **File Processing**: Handle .dat, .txt, .lnw, .fits files
- **OSC Integration**: Fetch spectra from Open Supernova Catalog
- **User Model Upload**: Upload and use custom ML models
- **Batch Processing**: Process multiple files simultaneously
- **Line List Analysis**: Element/ion line list functionality
- **Redshift Estimation**: DASH-only redshift estimation
- **Template Analysis**: Spectral template comparison
- **Production Ready**: Security, monitoring, logging, and Docker support

## 🏗️ Architecture

The backend follows a layered architecture:

```
app/
├── api/v1/           # API endpoints
├── config/           # Configuration management
├── core/             # Core utilities (security, monitoring, etc.)
├── domain/           # Business logic and models
│   ├── models/       # Domain models
│   ├── repositories/ # Repository interfaces
│   └── services/     # Business logic services
└── infrastructure/   # External integrations
    ├── database/     # Database models and repositories
    ├── ml/           # ML models and processors
    └── storage/      # File storage and OSC integration
```

## 🛠️ Installation

### Prerequisites

- Python 3.11+
- micromamba (recommended) or conda
- Git

### Setup

1. **Clone and navigate to the backend:**
   ```bash
   cd prod_backend
   ```

2. **Activate the environment:**
   ```bash
   micromamba activate astroweb
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements/prod.txt
   ```

4. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Run database migrations:**
   ```bash
   alembic upgrade head
   ```

6. **Start the server:**
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

## 🐳 Docker Deployment

### Build and run with Docker:

```bash
# Build the image
docker build -t astrodash-backend .

# Run the container
docker run -p 8000:8000 astrodash-backend
```

### Using Docker Compose:

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f backend
```

## 📊 API Endpoints

### Core Endpoints
- `GET /` - API information
- `GET /health` - Health check with metrics
- `GET /docs` - Interactive API documentation

### Spectrum Processing
- `POST /api/v1/process` - Process spectrum for classification
- `POST /api/v1/estimate-redshift` - Estimate redshift (DASH only)

### Analysis Options
- `GET /api/v1/analysis-options` - Get available analysis options
- `GET /api/v1/template-statistics` - Get template statistics

### Line List
- `GET /api/v1/line-list` - Get line list data
- `GET /api/v1/line-list/elements` - Get available elements
- `GET /api/v1/line-list/element/{element}` - Get element wavelengths
- `GET /api/v1/line-list/filter` - Filter by wavelength range

### Batch Processing
- `POST /api/v1/batch-process` - Process multiple files

### Model Management
- `POST /api/v1/upload-model` - Upload custom model
- `GET /api/v1/models` - List uploaded models
- `GET /api/v1/models/{model_id}` - Get model details

## 🔧 Configuration

### Environment Variables

Key configuration options in `.env`:

```env
# General
ENVIRONMENT=production
DEBUG=false
API_PREFIX=/api/v1

# Security
SECRET_KEY=your-secret-key
ALLOWED_HOSTS=localhost,127.0.0.1
CORS_ORIGINS=https://your-frontend.com

# Database
DATABASE_URL=sqlite:///./astrodash.db

# Storage
STORAGE_DIR=storage
USER_MODEL_DIR=app/astrodash_models/user_uploaded

# ML Models
DASH_MODEL_PATH=app/astrodash_models/zeroZ/pytorch_model.pth
TRANSFORMER_MODEL_PATH=app/astrodash_models/yuqing_models/TF_wiserep_v6.pt
TEMPLATE_PATH=app/astrodash_models/sn_and_host_templates.npz
LINE_LIST_PATH=app/astrodash_models/sneLineList.txt

# External APIs
OSC_API_URL=https://api.sne.space
```

## 🧪 Testing

### Run tests:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/unit/ -v
python -m pytest tests/integration/ -v
python -m pytest tests/test_api.py -v

# Run with coverage
python -m pytest tests/ --cov=app --cov-report=term-missing
```

## 📈 Monitoring

### Health Check

The `/health` endpoint provides comprehensive health information:

```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T00:00:00",
  "version": "1.0.0",
  "metrics": {
    "uptime_seconds": 3600,
    "total_requests": 1000,
    "total_errors": 5,
    "system": {
      "cpu_percent": 25.5,
      "memory_percent": 60.2,
      "disk_percent": 45.0
    }
  }
}
```

### Logging

Structured logging is configured with different levels:
- `INFO` - General application logs
- `WARNING` - Non-critical issues
- `ERROR` - Errors that need attention
- `DEBUG` - Detailed debugging information

## 🔒 Security

### Security Features

- **CORS Protection**: Configurable CORS origins
- **Trusted Hosts**: Validate incoming host headers
- **Security Headers**: XSS protection, content type options
- **File Upload Validation**: File type and size restrictions
- **Rate Limiting**: Configurable request rate limits
- **Input Sanitization**: Sanitize filenames and user inputs

### Production Security Checklist

- [ ] Change default `SECRET_KEY`
- [ ] Configure proper `ALLOWED_HOSTS`
- [ ] Set up HTTPS/TLS
- [ ] Configure firewall rules
- [ ] Set up proper file permissions
- [ ] Enable rate limiting
- [ ] Set up monitoring and alerting

## 🚀 Deployment

### Quick Deployment

Use the deployment script:

```bash
# Make executable
chmod +x deploy.sh

# Run deployment
./deploy.sh

# With Docker
./deploy.sh --docker
```

### Production Deployment

1. **Environment Setup:**
   ```bash
   cp production.env .env
   # Edit .env with production values
   ```

2. **Database Setup:**
   ```bash
   alembic upgrade head
   ```

3. **Start with Gunicorn:**
   ```bash
   gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
   ```

4. **Behind Reverse Proxy:**
   - Configure Nginx/Apache
   - Set up SSL certificates
   - Configure load balancing

## 📝 Development

### Code Structure

- **API Layer**: FastAPI routers in `app/api/v1/`
- **Service Layer**: Business logic in `app/domain/services/`
- **Repository Layer**: Data access in `app/infrastructure/`
- **Domain Models**: Core entities in `app/domain/models/`

### Adding New Features

1. **Domain Model**: Define in `app/domain/models/`
2. **Repository**: Implement in `app/infrastructure/`
3. **Service**: Add business logic in `app/domain/services/`
4. **API Endpoint**: Create in `app/api/v1/`
5. **Tests**: Add unit and integration tests

## 🤝 Contributing

1. Follow the layered architecture
2. Add tests for new features
3. Update documentation
4. Follow PEP 8 style guidelines
5. Use type hints throughout

## 📄 License

This project is part of the AstroDash application suite.

## 🆘 Support

For issues and questions:
- Check the API documentation at `/docs`
- Review the health endpoint at `/health`
- Check application logs
- Run the test suite to verify functionality
