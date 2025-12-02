# RAG-Cache

A high-performance RAG (Retrieval-Augmented Generation) caching system built with FastAPI, featuring intelligent caching strategies using local Ollama, Qdrant vector database, and Redis.

## Overview

RAG-Cache is designed to optimize LLM query performance by implementing multi-level caching:
- **Exact Cache**: Redis-based exact query matching for instant responses
- **Semantic Cache**: Qdrant-based vector similarity search for semantically similar queries
- **LLM Provider**: Local Ollama integration for cost-effective, privacy-preserving LLM responses

## Architecture

This implementation uses **local services** for complete privacy and cost control:

- **Ollama**: Local LLM inference (default: llama3) - no API keys required
- **Qdrant**: Local vector database for semantic similarity search
- **Redis**: In-memory cache for exact query matching and fast retrieval

## Features

- ✅ **Local LLM Processing**: Uses Ollama for completely local, private LLM inference
- ✅ **Dual Caching Strategy**: Exact match (Redis) + Semantic similarity (Qdrant)
- ✅ **Fast Response Times**: Sub-millisecond responses for cached queries
- ✅ **Cost Effective**: No external API costs with local Ollama
- ✅ **Privacy First**: All processing happens locally
- ✅ **RESTful API**: Clean FastAPI endpoints with OpenAPI documentation
- ✅ **Health Monitoring**: Built-in health checks and metrics endpoints

## Prerequisites

- Docker and Docker Compose
- Ollama installed and running locally (default: `http://localhost:11434`)
- At least one Ollama model downloaded (e.g., `llama3`)

### Installing Ollama

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama
ollama serve

# Pull a model (e.g., llama3)
ollama pull llama3
```

## Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd RAG-Cache
   ```

2. **Configure environment**
   ```bash
   cp .env.example .env  # If available, or create .env with:
   ```
   
   Minimum `.env` configuration:
   ```env
   default_llm_provider=ollama
   ollama_endpoint=http://host.docker.internal:11434
   ollama_model=llama3
   ```

3. **Start services**
   ```bash
   docker-compose up -d
   ```

4. **Verify services are running**
   ```bash
   docker-compose ps
   ```

   All services should show as "healthy" or "running":
   - `ragcache-api` (port 8000)
   - `ragcache-redis` (port 6379)
   - `ragcache-qdrant` (ports 6333, 6334)

5. **Test the API**
   ```bash
   curl -X POST http://localhost:8000/api/v1/query \
     -H "Content-Type: application/json" \
     -d '{"query": "What is the capital of France?"}'
   ```

## API Documentation

Once the services are running, access the interactive API documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Endpoints

### Query Endpoint

**POST** `/api/v1/query`

Process a query with intelligent caching.

**Request:**
```json
{
  "query": "Your question here",
  "provider": "ollama",  // Optional, defaults to config
  "model": "llama3",     // Optional, defaults to config
  "use_cache": true,     // Optional, defaults to true
  "use_semantic_cache": true  // Optional, defaults to true
}
```

**Response:**
```json
{
  "response": "The answer to your question",
  "provider": "ollama",
  "model": "llama3",
  "usage": {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0
  },
  "cache_info": {
    "cache_hit": true,
    "cache_type": "exact",
    "similarity_score": null
  },
  "latency_ms": 0.85
}
```

### Health Endpoints

- **GET** `/health` - Basic health check
- **GET** `/healthz` - Kubernetes-style health check
- **GET** `/ready` - Readiness probe
- **GET** `/live` - Liveness probe

### Metrics Endpoints

- **GET** `/api/v1/metrics` - Application metrics
- **GET** `/api/v1/metrics/prometheus` - Prometheus-formatted metrics

## Configuration

Key configuration options in `.env`:

```env
# LLM Provider
default_llm_provider=ollama
ollama_endpoint=http://host.docker.internal:11434
ollama_model=llama3

# Redis
redis_host=redis
redis_port=6379

# Qdrant
qdrant_host=qdrant
qdrant_port=6333
qdrant_collection_name=query_embeddings

# Cache Settings
cache_ttl_seconds=3600
semantic_similarity_threshold=0.85
enable_semantic_cache=true
enable_exact_cache=true
```

## How It Works

1. **Query Received**: API receives a query request
2. **Exact Cache Check**: Redis is checked for exact query match
   - If found: Return cached response immediately
3. **Semantic Cache Check**: Qdrant searches for semantically similar queries
   - If similarity > threshold: Return cached response
4. **LLM Call**: If no cache hit, query is sent to local Ollama
5. **Cache Storage**: Response is stored in both Redis (exact) and Qdrant (semantic)

## Development

### Running Tests

```bash
docker-compose exec api pytest
```

### Viewing Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f api
```

### Stopping Services

```bash
docker-compose down
```

### Rebuilding After Changes

```bash
docker-compose build api
docker-compose up -d api
```

## Technology Stack

- **API Framework**: FastAPI (Python 3.11)
- **LLM**: Ollama (local inference)
- **Vector Database**: Qdrant
- **Cache**: Redis
- **Embeddings**: sentence-transformers (local)
- **Containerization**: Docker & Docker Compose

## Troubleshooting

### Ollama Connection Issues

If you see 404 errors when calling Ollama:

1. Verify Ollama is running: `curl http://localhost:11434/api/tags`
2. Check the endpoint in `.env`: Should be `http://host.docker.internal:11434`
3. Ensure the model is downloaded: `ollama list`

### Qdrant Health Check Failing

The Qdrant health check may show as "unhealthy" but the service is still functional. This is a known issue with the health check command and doesn't affect functionality.

### Port Conflicts

If ports are already in use, modify `docker-compose.yml` to use different ports.

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]

