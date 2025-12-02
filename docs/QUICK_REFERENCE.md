# RAG Cache Developer Quick Reference

A concise reference guide for developers working with RAG Cache.

## Quick Start

```bash
# Clone and setup
git clone <repo>
cd "Rag cache"
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env and add OPENAI_API_KEY

# Start services
docker-compose up -d redis qdrant

# Run application
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Verify
curl http://localhost:8000/health
```

---

## API Cheat Sheet

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Basic health check |
| `GET` | `/ready` | Detailed readiness check |
| `POST` | `/api/v1/query` | Process query with caching |
| `GET` | `/api/v1/metrics` | Get metrics (JSON) |
| `GET` | `/api/v1/metrics/prometheus` | Get metrics (Prometheus) |

### Query Request

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is machine learning?",
    "use_cache": true,
    "provider": "openai",
    "model": "gpt-3.5-turbo",
    "max_tokens": 500,
    "temperature": 0.7
  }'
```

### Response Format

```json
{
  "response": "Machine learning is...",
  "provider": "openai",
  "model": "gpt-3.5-turbo",
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 150,
    "total_tokens": 160
  },
  "cache_info": {
    "cache_hit": true,
    "cache_type": "semantic",
    "similarity_score": 0.92
  },
  "latency_ms": 45.2
}
```

---

## Environment Variables

### Required

```bash
OPENAI_API_KEY=sk-...           # OpenAI API key
```

### Optional

```bash
# Application
APP_ENV=development             # development|staging|production
LOG_LEVEL=INFO                  # DEBUG|INFO|WARNING|ERROR
DEBUG=false                     # Enable debug mode

# API
API_HOST=0.0.0.0
API_PORT=8000

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=                 # Leave empty for no auth

# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333

# LLM
DEFAULT_MODEL=gpt-3.5-turbo
DEFAULT_MAX_TOKENS=1000
DEFAULT_TEMPERATURE=0.7

# Cache
CACHE_TTL_SECONDS=3600
SEMANTIC_SIMILARITY_THRESHOLD=0.85
ENABLE_SEMANTIC_CACHE=true
ENABLE_EXACT_CACHE=true

# Embedding
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

---

## Project Structure

```
app/
├── main.py              # FastAPI app entry point
├── config.py            # Configuration management
├── exceptions.py        # Custom exceptions
│
├── api/                 # API layer
│   ├── routes/          # Endpoint definitions
│   │   ├── health.py    # Health checks
│   │   ├── query.py     # Query endpoint
│   │   └── metrics.py   # Metrics endpoint
│   ├── middleware/      # Request middleware
│   └── deps.py          # Dependency injection
│
├── services/            # Business logic
│   └── query_service.py # Query orchestration
│
├── cache/               # Cache layer
│   ├── redis_cache.py   # Redis operations
│   └── qdrant_*.py      # Qdrant operations
│
├── llm/                 # LLM providers
│   ├── provider.py      # Base provider
│   ├── openai_provider.py
│   ├── anthropic_provider.py
│   ├── factory.py       # Provider factory
│   ├── rate_limiter.py  # Rate limiting
│   └── retry.py         # Retry logic
│
├── embeddings/          # Embedding generation
│   └── embedding_generator.py
│
├── pipeline/            # Processing pipeline
│   ├── semantic_matcher.py
│   ├── query_normalizer.py
│   └── performance_monitor.py
│
├── models/              # Pydantic models
│   ├── query.py         # Request models
│   ├── response.py      # Response models
│   └── cache_entry.py   # Cache models
│
└── utils/               # Utilities
    ├── logger.py        # Logging
    └── hasher.py        # Query hashing
```

---

## Common Code Patterns

### Adding a New Endpoint

```python
# app/api/routes/my_route.py
from fastapi import APIRouter, Depends

router = APIRouter()

@router.get("/my-endpoint")
async def my_endpoint():
    """Endpoint description."""
    return {"message": "Hello"}
```

```python
# app/main.py - Register the router
from app.api.routes import my_route
app.include_router(my_route.router, prefix="/api/v1", tags=["my-tag"])
```

### Using Configuration

```python
from app.config import config

# Access settings
print(config.redis_host)
print(config.semantic_similarity_threshold)

# Check environment
if config.is_production:
    # Production-specific logic
    pass
```

### Creating a Pydantic Model

```python
from pydantic import BaseModel, Field

class MyRequest(BaseModel):
    """Request model with validation."""

    name: str = Field(..., min_length=1, max_length=100)
    value: int = Field(default=0, ge=0, le=100)
```

### Using the Query Service

```python
from app.services.query_service import QueryService
from app.models.query import QueryRequest

request = QueryRequest(query="What is AI?")
response = await query_service.process(request)

if response.cache_info.cache_hit:
    print(f"Cache hit: {response.cache_info.cache_type}")
```

### Logging

```python
from app.utils.logger import get_logger

logger = get_logger(__name__)

logger.info("Processing query", query_hash="abc123")
logger.error("Failed to process", error=str(e))
```

---

## Testing

### Run All Tests

```bash
pytest
```

### Run with Coverage

```bash
pytest --cov=app --cov-report=html
open htmlcov/index.html
```

### Run Specific Tests

```bash
# Unit tests only
pytest tests/unit/

# Integration tests
pytest tests/integration/

# Specific file
pytest tests/unit/llm/test_openai_provider.py -v

# Specific test
pytest tests/unit/test_config.py::test_redis_url -v
```

### Test Markers

```bash
# Skip slow tests
pytest -m "not slow"

# Only integration tests
pytest -m integration
```

---

## Docker Commands

```bash
# Start all services
docker-compose up -d

# Start specific services
docker-compose up -d redis qdrant

# View logs
docker-compose logs -f app

# Stop services
docker-compose down

# Rebuild images
docker-compose build --no-cache

# Production deployment
docker-compose -f docker-compose.prod.yml up -d
```

---

## Cache Behavior

### Cache Types

| Type | Storage | Latency | Use Case |
|------|---------|---------|----------|
| **Exact** | Redis | ~1ms | Identical queries |
| **Semantic** | Qdrant | ~450ms | Similar questions |
| **Miss** | LLM API | ~8,500ms | New unique queries |

### Cache Flow

```
1. Query received
2. Check exact match (Redis)
   ├── Hit → Return cached (1ms)
   └── Miss → Continue
3. Check semantic match (Qdrant)
   ├── Hit (score >= 0.85) → Return cached (450ms)
   └── Miss → Continue
4. Call LLM (8,500ms)
5. Store in both caches
6. Return response
```

### Disable Caching

```json
{
  "query": "My question",
  "use_cache": false
}
```

---

## Performance Benchmarks

| Metric | Target | Typical |
|--------|--------|---------|
| Exact cache hit | < 5ms | ~1ms |
| Semantic cache hit | < 500ms | ~450ms |
| LLM response | < 15s | ~8.5s |
| Cache hit rate | > 40% | ~58% |

---

## Troubleshooting

### Redis Connection

```bash
# Check Redis
redis-cli ping
# Expected: PONG

# Check from container
docker exec -it ragcache-redis-1 redis-cli ping
```

### Qdrant Connection

```bash
# Check Qdrant health
curl http://localhost:6333/health
# Expected: {"title":"qdrant - vector search engine","version":"..."}
```

### Application Logs

```bash
# View app logs
docker-compose logs -f app

# Check specific error
grep -i error logs/app.log
```

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `OPENAI_API_KEY not set` | Missing env var | Add to `.env` file |
| `Redis connection refused` | Redis not running | `docker-compose up -d redis` |
| `Qdrant unreachable` | Qdrant not running | `docker-compose up -d qdrant` |
| `502 Bad Gateway` | LLM provider error | Check API key/limits |

---

## Code Style

### Sandi Metz Rules

- **Classes < 100 lines**
- **Methods < 5-10 lines**
- **Methods < 4 parameters**
- **Descriptive naming**

### Formatting

```bash
# Format code
black app/ tests/

# Sort imports
isort app/ tests/

# Lint
flake8 app/ tests/

# Type check
mypy app/
```

### Pre-commit

```bash
# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

---

## Useful Links

- **API Docs:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
- **Metrics:** http://localhost:8000/api/v1/metrics
- **Health:** http://localhost:8000/health

## Related Documentation

- [API Documentation](API.md)
- [Architecture](ARCHITECTURE.md)
- [Configuration Guide](CONFIGURATION.md)
- [Module Reference](MODULES.md)
- [Diagrams](DIAGRAMS.md)
- [Testing Guide](TESTING.md)
- [Deployment Guide](DEPLOYMENT.md)
