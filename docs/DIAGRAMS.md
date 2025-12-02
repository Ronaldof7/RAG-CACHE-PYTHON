# RAG Cache Architecture Diagrams

This document contains visual representations of the RAG Cache system architecture using Mermaid diagrams.

## System Architecture Overview

```mermaid
graph TB
    subgraph "Client Layer"
        Client[Client Application]
        CLI[CLI Tools]
    end

    subgraph "API Gateway"
        API[FastAPI Service<br/>Port 8000]
        MW[Middleware Stack]
    end

    subgraph "Service Layer"
        QS[Query Service]
        CM[Cache Manager]
        SM[Semantic Matcher]
    end

    subgraph "Provider Layer"
        LLM[LLM Factory]
        OAI[OpenAI Provider]
        ANT[Anthropic Provider]
    end

    subgraph "Data Layer"
        Redis[(Redis Cache<br/>Exact Match)]
        Qdrant[(Qdrant Vector DB<br/>Semantic Match)]
    end

    subgraph "External Services"
        OpenAI[OpenAI API]
        Anthropic[Anthropic API]
    end

    Client --> API
    CLI --> API
    API --> MW
    MW --> QS
    QS --> CM
    QS --> SM
    QS --> LLM
    CM --> Redis
    SM --> Qdrant
    LLM --> OAI
    LLM --> ANT
    OAI --> OpenAI
    ANT --> Anthropic
```

## Query Processing Flow

```mermaid
sequenceDiagram
    participant C as Client
    participant API as FastAPI
    participant QS as QueryService
    participant Redis as Redis Cache
    participant SM as SemanticMatcher
    participant Qdrant as Qdrant
    participant LLM as LLM Provider
    participant OpenAI as OpenAI API

    C->>API: POST /api/v1/query
    API->>QS: process(request)

    Note over QS: Step 1: Exact Cache
    QS->>Redis: get(query_hash)

    alt Exact Cache Hit
        Redis-->>QS: cached_entry
        QS-->>API: QueryResponse (cache_type: exact)
        API-->>C: Response (~1ms)
    else Exact Cache Miss
        Redis-->>QS: null

        Note over QS: Step 2: Semantic Cache
        QS->>SM: find_match(query)
        SM->>Qdrant: search_similar(embedding)

        alt Semantic Cache Hit
            Qdrant-->>SM: similar_results
            SM-->>QS: SemanticMatch
            QS-->>API: QueryResponse (cache_type: semantic)
            API-->>C: Response (~450ms)
        else Semantic Cache Miss
            Qdrant-->>SM: no_match
            SM-->>QS: null

            Note over QS: Step 3: LLM Call
            QS->>LLM: complete(request)
            LLM->>OpenAI: chat/completions
            OpenAI-->>LLM: response
            LLM-->>QS: LLMResponse

            Note over QS: Step 4: Store in Caches
            QS->>Redis: set(entry)
            QS->>SM: store_for_matching()
            SM->>Qdrant: upsert(embedding)

            QS-->>API: QueryResponse (cache_type: null)
            API-->>C: Response (~8500ms)
        end
    end
```

## Component Architecture

```mermaid
graph LR
    subgraph "app/api"
        Routes[routes/]
        Middleware[middleware/]
        Deps[deps.py]
    end

    subgraph "app/services"
        QueryService[query_service.py]
    end

    subgraph "app/cache"
        RedisCache[redis_cache.py]
        QdrantClient[qdrant_client.py]
        QdrantPool[qdrant_pool.py]
    end

    subgraph "app/llm"
        Provider[provider.py]
        OpenAIProvider[openai_provider.py]
        AnthropicProvider[anthropic_provider.py]
        Factory[factory.py]
        RateLimiter[rate_limiter.py]
        Retry[retry.py]
    end

    subgraph "app/embeddings"
        EmbeddingGenerator[embedding_generator.py]
        EmbeddingCache[embedding_cache.py]
    end

    subgraph "app/pipeline"
        SemanticMatcher[semantic_matcher.py]
        QueryNormalizer[query_normalizer.py]
        PerfMonitor[performance_monitor.py]
    end

    subgraph "app/models"
        QueryModels[query.py]
        ResponseModels[response.py]
        CacheModels[cache_entry.py]
        LLMModels[llm.py]
    end

    Routes --> QueryService
    QueryService --> RedisCache
    QueryService --> SemanticMatcher
    QueryService --> Factory
    SemanticMatcher --> EmbeddingGenerator
    SemanticMatcher --> QdrantClient
    Factory --> OpenAIProvider
    Factory --> AnthropicProvider
    OpenAIProvider --> RateLimiter
    OpenAIProvider --> Retry
```

## Data Models

```mermaid
classDiagram
    class QueryRequest {
        +str query
        +str? provider
        +str? model
        +int? max_tokens
        +float? temperature
        +bool use_cache
        +bool use_semantic_cache
        +get_provider(default) str
        +get_model(default) str
    }

    class QueryResponse {
        +str response
        +str provider
        +str model
        +UsageMetrics usage
        +CacheInfo cache_info
        +float latency_ms
        +from_cache() bool
        +is_exact_match() bool
        +is_semantic_match() bool
    }

    class UsageMetrics {
        +int prompt_tokens
        +int completion_tokens
        +int total_tokens
        +create(prompt, completion)$ UsageMetrics
    }

    class CacheInfo {
        +bool cache_hit
        +str? cache_type
        +float? similarity_score
        +miss()$ CacheInfo
        +exact_hit()$ CacheInfo
        +semantic_hit(score)$ CacheInfo
    }

    class CacheEntry {
        +str query_hash
        +str original_query
        +str response
        +str provider
        +str model
        +int prompt_tokens
        +int completion_tokens
        +list[float]? embedding
    }

    class LLMResponse {
        +str content
        +int prompt_tokens
        +int completion_tokens
        +str model
        +total_tokens() int
    }

    QueryResponse --> UsageMetrics
    QueryResponse --> CacheInfo
```

## Cache Strategy

```mermaid
flowchart TD
    Start([Query Received]) --> ExactCheck{Exact Match<br/>in Redis?}

    ExactCheck -->|Yes| ExactHit[Return Cached Response<br/>~1ms latency]
    ExactCheck -->|No| SemanticCheck{Semantic Match<br/>in Qdrant?<br/>score >= 0.85}

    SemanticCheck -->|Yes| SemanticHit[Return Similar Response<br/>~450ms latency]
    SemanticCheck -->|No| LLMCall[Call LLM Provider<br/>~8500ms latency]

    LLMCall --> StoreExact[Store in Redis<br/>TTL: 3600s]
    StoreExact --> StoreSemantic[Store in Qdrant<br/>with Embedding]
    StoreSemantic --> Return[Return Response]

    ExactHit --> End([Response Sent])
    SemanticHit --> End
    Return --> End

    style ExactHit fill:#4ade80
    style SemanticHit fill:#60a5fa
    style LLMCall fill:#f97316
```

## Deployment Architecture

### Development Environment

```mermaid
graph TB
    subgraph "Docker Compose"
        App[RAGCache App<br/>:8000]
        Redis[(Redis<br/>:6379)]
        Qdrant[(Qdrant<br/>:6333/:6334)]
    end

    Dev[Developer] --> App
    App --> Redis
    App --> Qdrant
```

### Production Environment

```mermaid
graph TB
    subgraph "Load Balancer"
        LB[Nginx/ALB]
    end

    subgraph "Application Tier"
        App1[RAGCache #1]
        App2[RAGCache #2]
        AppN[RAGCache #N]
    end

    subgraph "Cache Tier"
        Redis[(Redis Cluster)]
        Qdrant[(Qdrant Cluster)]
    end

    subgraph "External"
        OpenAI[OpenAI API]
        Anthropic[Anthropic API]
    end

    subgraph "Monitoring"
        Prometheus[Prometheus]
        Grafana[Grafana]
    end

    LB --> App1
    LB --> App2
    LB --> AppN

    App1 --> Redis
    App2 --> Redis
    AppN --> Redis

    App1 --> Qdrant
    App2 --> Qdrant
    AppN --> Qdrant

    App1 --> OpenAI
    App1 --> Anthropic

    Prometheus --> App1
    Prometheus --> App2
    Prometheus --> AppN
    Grafana --> Prometheus
```

## LLM Provider Architecture

```mermaid
graph TB
    subgraph "LLM Layer"
        Factory[LLMProviderFactory]
        Base[BaseLLMProvider<br/>Abstract]
        OpenAI[OpenAIProvider]
        Anthropic[AnthropicProvider]
    end

    subgraph "Resilience"
        RateLimiter[RateLimiter]
        RetryHandler[RetryHandler]
        CircuitBreaker[CircuitBreaker]
    end

    subgraph "Request Flow"
        Request[QueryRequest]
        Response[LLMResponse]
    end

    Factory --> Base
    Base --> OpenAI
    Base --> Anthropic

    OpenAI --> RateLimiter
    OpenAI --> RetryHandler
    Anthropic --> RateLimiter
    Anthropic --> RetryHandler

    RateLimiter --> CircuitBreaker

    Request --> Factory
    Factory --> Response
```

## Pipeline Components

```mermaid
graph LR
    subgraph "Query Pipeline"
        Validator[QueryValidator]
        Normalizer[QueryNormalizer]
        Preprocessor[QueryPreprocessor]
    end

    subgraph "Cache Pipeline"
        ExactLookup[Exact Cache Lookup]
        SemanticLookup[Semantic Cache Lookup]
        CacheStore[Cache Storage]
    end

    subgraph "Processing"
        LLMProcess[LLM Processing]
        ResponseBuild[Response Builder]
    end

    subgraph "Monitoring"
        PerfMonitor[Performance Monitor]
        RequestContext[Request Context]
    end

    Validator --> Normalizer --> Preprocessor
    Preprocessor --> ExactLookup
    ExactLookup -->|Miss| SemanticLookup
    SemanticLookup -->|Miss| LLMProcess
    LLMProcess --> CacheStore
    CacheStore --> ResponseBuild

    PerfMonitor -.-> ExactLookup
    PerfMonitor -.-> SemanticLookup
    PerfMonitor -.-> LLMProcess
    RequestContext -.-> Validator
```

## Error Handling Flow

```mermaid
flowchart TD
    Request([Incoming Request]) --> Validation{Valid Request?}

    Validation -->|No| ValidationError[400 Bad Request]
    Validation -->|Yes| CacheCheck{Cache Available?}

    CacheCheck -->|Error| CacheError[Log & Continue]
    CacheCheck -->|OK| CacheLookup[Cache Lookup]
    CacheError --> LLMCall

    CacheLookup -->|Hit| Success[Return Cached]
    CacheLookup -->|Miss| LLMCall[Call LLM]

    LLMCall --> LLMError{LLM Error?}

    LLMError -->|Yes| RetryCheck{Retry Available?}
    LLMError -->|No| StoreCache[Store in Cache]

    RetryCheck -->|Yes| LLMCall
    RetryCheck -->|No| ProviderError[502 Bad Gateway]

    StoreCache -->|Error| LogError[Log Error]
    StoreCache -->|OK| Success
    LogError --> Success

    Success --> Response([Return Response])
    ValidationError --> Response
    ProviderError --> Response

    style ValidationError fill:#ef4444
    style ProviderError fill:#ef4444
    style Success fill:#22c55e
```

---

## Related Documentation

- [API Documentation](API.md)
- [Architecture](ARCHITECTURE.md)
- [Deployment Guide](DEPLOYMENT.md)
- [Configuration](CONFIGURATION.md)
