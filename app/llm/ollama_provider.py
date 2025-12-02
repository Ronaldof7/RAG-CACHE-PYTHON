"""
Ollama LLM provider implementation.

Sandi Metz Principles:
- Single Responsibility: Ollama API interaction
- Small methods: Each method < 10 lines
- Dependency Injection: Endpoint and model injected
"""

import httpx

from app.config import config
from app.exceptions import LLMProviderError
from app.llm.provider import BaseLLMProvider
from app.models.llm import LLMResponse
from app.models.query import QueryRequest
from app.utils.logger import get_logger

logger = get_logger(__name__)


class OllamaProvider(BaseLLMProvider):
    """
    Ollama implementation of LLM provider.

    Handles communication with local Ollama API.
    """

    def __init__(
        self,
        endpoint: str = None,
        model: str = None,
        timeout: float = 60.0,
    ):
        """
        Initialize Ollama provider.

        Args:
            endpoint: Ollama API endpoint (defaults to config)
            model: Model name (defaults to config)
            timeout: Request timeout in seconds (default: 60.0)
        """
        self.endpoint = endpoint or config.ollama_endpoint
        self.model = model or config.ollama_model
        self.timeout = timeout

    async def complete(self, request: QueryRequest) -> LLMResponse:
        """
        Generate completion using Ollama.

        Args:
            request: Query request

        Returns:
            LLM response

        Raises:
            LLMProviderError: If API call fails
        """
        data = {
            "model": self.model,
            "prompt": request.query,
            "stream": False,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.endpoint}/api/generate", json=data
                )
                response.raise_for_status()
                result = response.json()

            content = result.get("response", "")
            if not content:
                error_msg = f"Ollama returned empty response for model {self.model}"
                logger.error("Empty response from Ollama", model=self.model)
                raise LLMProviderError(error_msg)

            return LLMResponse(
                content=content,
                prompt_tokens=0,  # Ollama doesn't provide token counts
                completion_tokens=0,
                model=self.model,
            )

        except httpx.TimeoutException as e:
            error_msg = (
                f"Ollama request timed out after {self.timeout}s. "
                f"Endpoint: {self.endpoint}, Model: {self.model}"
            )
            logger.error("Ollama timeout", error=str(e), endpoint=self.endpoint)
            raise LLMProviderError(error_msg) from e

        except httpx.HTTPStatusError as e:
            error_msg = (
                f"Ollama API returned error {e.response.status_code}: "
                f"{e.response.text[:200]}. Endpoint: {self.endpoint}"
            )
            logger.error(
                "Ollama HTTP error",
                status_code=e.response.status_code,
                error=str(e),
            )
            raise LLMProviderError(error_msg) from e

        except httpx.RequestError as e:
            error_msg = (
                f"Failed to connect to Ollama at {self.endpoint}. "
                f"Ensure Ollama is running and accessible."
            )
            logger.error("Ollama connection error", error=str(e), endpoint=self.endpoint)
            raise LLMProviderError(error_msg) from e

        except Exception as e:
            error_msg = f"Unexpected error in Ollama provider: {str(e)}"
            logger.error("Unexpected Ollama error", error=str(e))
            raise LLMProviderError(error_msg) from e

    def get_name(self) -> str:
        """Get provider name."""
        return "ollama"
