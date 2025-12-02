# RAG Cache Module Documentation

This document provides detailed documentation for all modules in the RAG Cache system.

## Table of Contents

- [Overview](#overview)
- [Core Modules](#core-modules)
  - [app/config.py](#appconfigpy)
  - [app/main.py](#appmainpy)
  - [app/exceptions.py](#appexceptionspy)
- [API Layer](#api-layer)
  - [app/api/routes/query.py](#appapioutesquerypy)
  - [app/api/routes/health.py](#appapiouteshealthpy)
  - [app/api/routes/metrics.py](#appapioutesmetricspy)
- [Service Layer](#service-layer)
  - [app/services/query_service.py](#appservicesquery_servicepy)
- [Cache Layer](#cache-layer)
  - [app/cache/redis_cache.py](#appcacheredis_cachepy)
  - [app/cache/qdrant_client.py](#appcacheqdrant_clientpy)
- [LLM Layer](#llm-layer)
  - [app/llm/provider.py](#appllmproviderpy)
  - [app/llm/openai_provider.py](#appllmopenai_providerpy)
  - [app/llm/rate_limiter.py](#appllmrate_limiterpy)
- [Embeddings Layer](#embeddings-layer)
  - [app/embeddings/embedding_generator.py](#appembeddingsembedding_generatorpy)
- [Pipeline Layer](#pipeline-layer)
  - [app/pipeline/semantic_matcher.py](#apppipelinesemantic_matcherpy)
- [Models](#models)
  - [app/models/query.py](#appmodelsquerypy)
  - [app/models/response.py](#appmodelsresponsepy)

---

## Overview

RAG Cache follows a layered architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────┐
│  API Layer (FastAPI routes & middleware)    │
├─────────────────────────────────────────────┤
│  Service Layer (Business logic)             │
├─────────────────────────────────────────────┤
│  Repository Layer (Data access)             │
├─────────────────────────────────────────────┤
│  Provider Layer (External integrations)     │
└─────────────────────────────────────────────┘
```

All modules follow **Sandi Metz's POOD principles**:
- Classes < 100 lines
- Methods < 5-10 lines
- Methods < 4 parameters
- Descriptive naming

---

## Core Modules

### app/config.py

**Purpose:** Application configuration management with validation.

**Class:** `AppConfig`

```python
class AppConfig(BaseSettings):
    """Application configuration with validation."""
```

**Key Configuration Groups:**

| Group | Variables | Description |
|-------|-----------|-------------|
| Application | `app_name`, `app_env`, `log_level`, `debug` | Core app settings |
| API | `api_host`, `api_port`, `api_workers`, `allowed_origins` | HTTP server config |
| Redis | `redis_host`, `redis_port`, `redis_db`, `redis_password` | Exact cache settings |
| Qdrant | `qdrant_host`, `qdrant_port`, `qdrant_collection_name` | Vector DB settings |
| LLM | `openai_api_key`, `anthropic_api_key`, `default_model` | Provider settings |
| Cache | `cache_ttl_seconds`, `semantic_similarity_threshold` | Cache behavior |
| Embedding | `embedding_model`, `embedding_device`, `embedding_batch_size` | Embedding config |

**Properties:**

```python
@property
def redis_url(self) -> str:
    """Build Redis URL from components."""

@property
def qdrant_url(self) -> str:
    """Build Qdrant HTTP URL."""

@property
def is_development(self) -> bool:
    """Check if running in development mode."""

@property
def is_production(self) -> bool:
    """Check if running in production mode."""
```

**Usage:**

```python
from app.config import config

# Access configuration
print(config.redis_host)
print(config.semantic_similarity_threshold)
print(config.is_production)
```

---

### app/main.py

**Purpose:** FastAPI application factory and lifecycle management.

**Class:** `ApplicationState`

```python
class ApplicationState:
    """Manages application-wide state."""

    async def startup(self) -> None:
        """Initialize application resources."""

    async def shutdown(self) -> None:
        """Cleanup application resources."""
```

**Function:** `create_application()`

```python
def create_application() -> FastAPI:
    """Create and configure FastAPI application."""
```

**Middleware Stack (order matters):**

1. `GZipMiddleware` - Response compression
2. `RequestLoggingMiddleware` - Request logging
3. `CORSMiddleware` - Cross-origin requests

**Routers:**

| Router | Prefix | Tags |
|--------|--------|------|
| health | `/` | health |
| query | `/api/v1` | query |
| metrics | `/api/v1` | metrics |

---

## API Layer

### app/api/routes/query.py

**Purpose:** Query processing endpoint.

**Endpoint:**

```python
@router.post("/query", response_model=QueryResponse)
async def process_query(
    request: QueryRequest,
    service: QueryService = Depends(get_query_service),
) -> QueryResponse:
    """Process query with caching."""
```

**Error Handling:**

| Exception | Status Code | Response |
|-----------|-------------|----------|
| `LLMProviderError` | 502 | Bad Gateway |
| `Exception` | 500 | Internal Server Error |

---

### app/api/routes/health.py

**Purpose:** Health check endpoints for monitoring.

**Endpoints:**

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Basic health check |
| `GET /healthz` | Kubernetes liveness probe |
| `GET /ready` | Kubernetes readiness probe |
| `GET /live` | Liveness check |

**Models:**

```python
class ComponentHealth(BaseModel):
    status: Literal["healthy", "unhealthy", "degraded"]
    latency_ms: Optional[float]
    message: Optional[str]

class DetailedHealthResponse(BaseModel):
    status: Literal["healthy", "unhealthy", "degraded"]
    environment: str
    version: str
    components: Dict[str, ComponentHealth]
```

---

### app/api/routes/metrics.py

**Purpose:** Expose application metrics for monitoring.

**Endpoints:**

| Endpoint | Format | Description |
|----------|--------|-------------|
| `GET /api/v1/metrics` | JSON | Application metrics |
| `GET /api/v1/metrics/prometheus` | Text | Prometheus format |

**Metrics Exposed:**

- `ragcache_info` - Application info
- `ragcache_requests_total` - Total requests
- `ragcache_cache_hits_total` - Cache hits
- `ragcache_cache_hit_rate` - Hit rate
- `ragcache_avg_latency_ms` - Average latency
- `ragcache_redis_keys` - Redis key count
- `ragcache_redis_memory_bytes` - Memory usage

---

## Service Layer

### app/services/query_service.py

**Purpose:** Main query processing orchestration.

**Class:** `QueryService`

```python
class QueryService:
    """Coordinates exact cache, semantic cache, and LLM operations."""

    def __init__(
        self,
        cache: RedisCache,
        llm_provider: BaseLLMProvider,
        semantic_matcher: Optional[SemanticMatcher] = None,
    ):
        """Initialize service with dependencies."""
```

**Public Methods:**

```python
async def process(self, request: QueryRequest) -> QueryResponse:
    """
    Process query with caching.

    Order: Exact cache -> Semantic cache -> LLM
    """
```

**Private Methods:**

| Method | Description |
|--------|-------------|
| `_check_exact_cache()` | Check Redis for exact match |
| `_check_semantic_cache()` | Check Qdrant for semantic match |
| `_call_llm()` | Call LLM provider |
| `_store_in_caches()` | Store response in both caches |
| `_build_cached_response()` | Build response from exact cache |
| `_build_semantic_response()` | Build response from semantic cache |
| `_build_response()` | Build response from LLM |

**Processing Flow:**

```
1. Start request context
2. Check exact cache (if enabled)
   └── Return if hit
3. Check semantic cache (if enabled)
   └── Return if hit
4. Call LLM provider
5. Store in caches
6. Return response
```

---

## Cache Layer

### app/cache/redis_cache.py

**Purpose:** Redis cache operations for exact matching.

**Key Operations:**

```python
async def get(self, query: str) -> Optional[CacheEntry]:
    """Get cached entry by query."""

async def set(self, entry: CacheEntry) -> bool:
    """Store cache entry with TTL."""

async def delete(self, query: str) -> bool:
    """Delete cache entry."""

async def get_metrics(self) -> CacheMetrics:
    """Get cache statistics."""
```

**Cache Key Format:**

```python
def generate_cache_key(query: str) -> str:
    """Generate deterministic hash for query."""
    # Uses SHA-256 hash of normalized query
```

---

### app/cache/qdrant_client.py

**Purpose:** Qdrant vector database connection management.

**Functions:**

```python
async def create_qdrant_client() -> AsyncQdrantClient:
    """Create Qdrant async client connection."""
```

**Class:** `QdrantConnectionManager`

```python
class QdrantConnectionManager:
    """Manages Qdrant client connection lifecycle."""

    async def get_client(self) -> AsyncQdrantClient:
        """Get or create Qdrant client."""

    async def close(self) -> None:
        """Close Qdrant client connection."""

    async def health_check(self) -> bool:
        """Check Qdrant server health."""

    async def reconnect(self) -> bool:
        """Reconnect to Qdrant server."""
```

**Context Manager:**

```python
@asynccontextmanager
async def get_pooled_client() -> AsyncIterator[AsyncQdrantClient]:
    """Context manager for acquiring pooled connection."""

# Usage:
async with get_pooled_client() as client:
    await client.upsert(...)
```

---

## LLM Layer

### app/llm/provider.py

**Purpose:** Abstract base class for LLM providers.

**Class:** `BaseLLMProvider`

```python
class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def complete(self, request: QueryRequest) -> LLMResponse:
        """Generate completion for query."""

    @abstractmethod
    def get_name(self) -> str:
        """Get provider name."""
```

---

### app/llm/openai_provider.py

**Purpose:** OpenAI API integration.

**Class:** `OpenAIProvider`

```python
class OpenAIProvider(BaseLLMProvider):
    """OpenAI implementation of LLM provider."""

    def __init__(
        self,
        api_key: str,
        rate_limiter: RateLimiter | None = None,
        retry_handler: RetryHandler | None = None,
        requests_per_minute: int = 500,
    ):
        """Initialize OpenAI provider."""
```

**Features:**

- Rate limiting (default: 500 RPM)
- Automatic retry with exponential backoff
- Async client for non-blocking I/O

**API Call:**

```python
response = await client.chat.completions.create(
    model=request.get_model(config.default_model),
    messages=[{"role": "user", "content": request.query}],
    max_tokens=request.get_max_tokens(config.default_max_tokens),
    temperature=request.get_temperature(config.default_temperature),
)
```

---

### app/llm/rate_limiter.py

**Purpose:** Rate limiting for LLM API calls.

**Class:** `RateLimiter`

```python
class RateLimiter:
    """Token bucket rate limiter."""

    async def acquire(self) -> None:
        """Acquire permission to make a request."""

    def update_limit(self, new_limit: int) -> None:
        """Update the rate limit."""
```

**Configuration:**

```python
@dataclass
class RateLimitConfig:
    requests_per_minute: int = 500
    burst_multiplier: float = 1.5
```

---

## Embeddings Layer

### app/embeddings/embedding_generator.py

**Purpose:** Generate vector embeddings from text.

**Class:** `EmbeddingGenerator`

```python
class EmbeddingGenerator:
    """Service to generate vector embeddings from text."""

    def __init__(self, model_name: str | None = None):
        """Initialize with SentenceTransformer model."""

    async def generate(self, text: str) -> List[float]:
        """Generate embedding for single text."""

    async def generate_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for batch of texts."""
```

**Default Model:** `sentence-transformers/all-MiniLM-L6-v2`

**Vector Dimensions:** 384

---

## Pipeline Layer

### app/pipeline/semantic_matcher.py

**Purpose:** Semantic similarity matching using vector search.

**Class:** `SemanticMatcher`

```python
class SemanticMatcher:
    """Finds semantically similar queries in cache."""

    async def find_match(self, query: str) -> Optional[SemanticMatch]:
        """Find semantically similar cached response."""

    async def store_for_matching(
        self,
        query: str,
        query_hash: str,
        response: str,
        provider: str,
        model: str,
    ) -> None:
        """Store query for future semantic matching."""
```

**Model:** `SemanticMatch`

```python
@dataclass
class SemanticMatch:
    original_query: str
    cached_response: str
    similarity_score: float
    provider: str
    model: str
```

**Similarity Threshold:** Configurable (default: 0.85)

---

## Models

### app/models/query.py

**Purpose:** Query request validation models.

**Class:** `QueryRequest`

```python
class QueryRequest(BaseModel):
    """Incoming query request with validation."""

    query: str = Field(..., min_length=1, max_length=10000)
    provider: Optional[Literal["openai", "anthropic"]] = None
    model: Optional[str] = None
    max_tokens: Optional[int] = Field(None, ge=1, le=4000)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    use_cache: bool = True
    use_semantic_cache: bool = True
```

**Validation:**

```python
@field_validator("query")
def validate_query(cls, v: str) -> str:
    """Validate and normalize query."""
    v = v.strip()
    if not v:
        raise ValueError("Query cannot be empty")
    return v
```

---

### app/models/response.py

**Purpose:** Query response models.

**Class:** `QueryResponse`

```python
class QueryResponse(BaseModel):
    """Query processing response."""

    response: str
    provider: str
    model: str
    usage: UsageMetrics
    cache_info: CacheInfo
    latency_ms: float
```

**Class:** `UsageMetrics`

```python
class UsageMetrics(BaseModel):
    """Token usage metrics."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    @classmethod
    def create(cls, prompt_tokens: int, completion_tokens: int) -> "UsageMetrics":
        """Create with calculated total."""
```

**Class:** `CacheInfo`

```python
class CacheInfo(BaseModel):
    """Cache hit information."""

    cache_hit: bool
    cache_type: Optional[Literal["exact", "semantic"]]
    similarity_score: Optional[float]

    @classmethod
    def miss(cls) -> "CacheInfo":
        """Create cache miss info."""

    @classmethod
    def exact_hit(cls) -> "CacheInfo":
        """Create exact cache hit info."""

    @classmethod
    def semantic_hit(cls, similarity_score: float) -> "CacheInfo":
        """Create semantic cache hit info."""
```

---

## Utility Modules

### app/utils/logger.py

**Purpose:** Structured logging configuration.

**Functions:**

```python
def setup_logging(level: str = "INFO") -> None:
    """Configure structured logging."""

def get_logger(name: str) -> BoundLogger:
    """Get logger for module."""

def log_llm_call(provider: str, model: str, tokens: int) -> None:
    """Log LLM API call."""
```

### app/utils/hasher.py

**Purpose:** Query hashing for cache keys.

**Functions:**

```python
def generate_cache_key(query: str) -> str:
    """Generate deterministic cache key from query."""
```

---

## Design Patterns Used

| Pattern | Location | Purpose |
|---------|----------|---------|
| Factory | `app/llm/factory.py` | Create LLM providers |
| Strategy | `CacheInfo.miss/exact_hit/semantic_hit` | Cache type handling |
| Repository | `app/repositories/` | Data access abstraction |
| Dependency Injection | FastAPI `Depends()` | Loose coupling |
| Builder | Pydantic models | Object construction |
| Context Manager | `get_pooled_client()` | Resource management |

---

## Related Documentation

- [API Documentation](API.md)
- [Architecture](ARCHITECTURE.md)
- [Diagrams](DIAGRAMS.md)
- [Configuration](CONFIGURATION.md)
