"""
app/core/embeddings.py
Embeddings factory.

Centralizing embedding creation here means you swap providers by
changing one env var, not by hunting through the codebase.

Supported:
  - OpenAI text-embedding-3-small / text-embedding-3-large
  - (Future) Anthropic Voyage via voyageai package
"""
from functools import lru_cache

from langchain_openai import OpenAIEmbeddings
from loguru import logger

from app.core.config import get_settings


@lru_cache(maxsize=1)
def get_embeddings():
    """
    Return a cached embedding model instance.
    Cached so we don't re-initialize on every call — the model object
    itself is stateless; it just holds config + API key.
    """
    settings = get_settings()

    logger.info(f"Initializing embeddings: {settings.embedding_model}")

    # OpenAI embeddings (default — works with both openai and anthropic LLM)
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key or None,
    )

    return embeddings
