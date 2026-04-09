import os
import concurrent.futures
from abc import ABC, abstractmethod
from typing import List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class EmbeddingSettings(BaseSettings):
    """
    Configuration settings for embedding providers.
    """

    model_config = SettingsConfigDict(
        env_prefix="EMBED_", env_file=".env", extra="ignore"
    )

    transformers_offline: str = Field(default="1", alias="TRANSFORMERS_OFFLINE")
    hf_hub_offline: str = Field(default="1", alias="HF_HUB_OFFLINE")
    provider: str = Field(default="hf")
    hf_model_name: str = Field(default="all-MiniLM-L6-v2")
    openai_api_key: Optional[str] = Field(default=None)
    openai_model_name: str = Field(default="text-embedding-3-small")
    timeout_seconds: int = Field(default=5)


class EmbeddingProvider(ABC):
    """
    Abstract interface defining the contract for embedding generation.
    """

    @abstractmethod
    def get_embedding(self, text: str) -> List[float]:
        """
        Generates a vector representation for a single string.
        """
        pass

    @abstractmethod
    def get_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generates vector representations for a list of strings.
        """
        pass


class HuggingFaceProvider(EmbeddingProvider):
    """
    Local embedding provider using Sentence-Transformers.
    """

    def __init__(self, settings: EmbeddingSettings):
        """
        Initializes the model ensuring local files are prioritized.
        """
        from sentence_transformers import SentenceTransformer

        os.environ["TRANSFORMERS_OFFLINE"] = settings.transformers_offline
        os.environ["HF_HUB_OFFLINE"] = settings.hf_hub_offline

        self._model = SentenceTransformer(
            settings.hf_model_name,
            trust_remote_code=True,
            local_files_only=(settings.hf_hub_offline == "1"),
        )
        self._timeout = settings.timeout_seconds
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    def _encode_sync(self, texts: List[str]) -> List[List[float]]:
        """
        Executes encoding synchronously.
        """
        return self._model.encode(texts).tolist()

    def _run_with_timeout(self, texts: List[str]) -> List[List[float]]:
        """
        Manages execution lifecycle for inference timeout protection.
        """
        future = self._executor.submit(self._encode_sync, texts)
        try:
            return future.result(timeout=self._timeout)
        except concurrent.futures.TimeoutError as e:
            raise TimeoutError("HuggingFace inference timed out.") from e
        except Exception as e:
            raise RuntimeError(f"HuggingFace inference failed: {str(e)}") from e

    def get_embedding(self, text: str) -> List[float]:
        """
        Generates a single embedding vector.
        """
        return self._run_with_timeout([text])[0]

    def get_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generates multiple embedding vectors.
        """
        return self._run_with_timeout(texts)


class OpenAIProvider(EmbeddingProvider):
    """
    Cloud embedding provider using OpenAI API.
    """

    def __init__(self, settings: EmbeddingSettings):
        """
        Initializes the OpenAI client with provided credentials.
        """
        from openai import OpenAI

        self._client = OpenAI(api_key=settings.openai_api_key)
        self._model = settings.openai_model_name
        self._timeout = settings.timeout_seconds

    def get_embedding(self, text: str) -> List[float]:
        """
        Fetches a single vector from the remote API.
        """
        response = self._client.embeddings.create(
            input=text, model=self._model, timeout=self._timeout
        )
        return response.data[0].embedding

    def get_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Fetches multiple vectors from the remote API.
        """
        response = self._client.embeddings.create(
            input=texts, model=self._model, timeout=self._timeout
        )
        return [item.embedding for item in response.data]


class EmbeddingProviderFactory:
    """
    Assembles embedding providers based on typed settings.
    """

    @staticmethod
    def create(settings: EmbeddingSettings) -> EmbeddingProvider:
        """
        Instantiates the concrete provider based on the settings configuration.
        """
        if settings.provider.lower() == "openai":
            if not settings.openai_api_key:
                raise ValueError("OpenAI API key is missing in environment.")
            return OpenAIProvider(settings)

        return HuggingFaceProvider(settings)
