"""
app/core/rag_chain.py
The RAG chain: retriever → context assembly → LLM → cited answer.

Architecture:
  1. MMR retrieval (balances relevance vs diversity, reduces redundancy)
  2. Context window assembly with source metadata
  3. LLM call with structured prompt enforcing citation format
  4. Response parsing with source extraction

The chain is built fresh per-query to keep it stateless and testable.
Conversation history is passed explicitly rather than stored in chain state.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import Document
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from loguru import logger

from app.core.config import get_settings
from app.core.embeddings import get_embeddings


# ── Prompt templates ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert document analyst. Your job is to answer questions \
accurately using ONLY the context provided below. 

Rules you must follow:
1. Base your answer exclusively on the provided context. Do not use outside knowledge.
2. If the context does not contain enough information to answer confidently, say:
   "I don't have enough information in the provided documents to answer this question."
3. Always cite your sources using [Source: filename, chunk N] format inline.
4. Be precise and concise. Avoid padding or filler language.
5. If multiple sources agree, cite all of them. If they conflict, note the conflict.

Context:
{context}

Conversation history (for follow-up questions):
{chat_history}
"""

HUMAN_PROMPT = "{question}"


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class RAGResponse:
    answer: str
    sources: list[dict]
    retrieved_chunks: list[Document]
    query: str
    retrieval_scores: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "sources": self.sources,
            "query": self.query,
            "chunk_count": len(self.retrieved_chunks),
            "retrieved_context": [
                {
                    "content": doc.page_content[:500],
                    "metadata": doc.metadata,
                }
                for doc in self.retrieved_chunks
            ],
        }


# ── LLM factory ──────────────────────────────────────────────────────────────

def _get_llm():
    settings = get_settings()
    if settings.llm_provider == "anthropic":
        return ChatAnthropic(
            model=settings.llm_model,
            api_key=settings.anthropic_api_key,
            temperature=0.1,  # Low temp for factual Q&A
            max_tokens=2048,
        )
    else:
        return ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.openai_api_key,
            temperature=0.1,
            max_tokens=2048,
        )


# ── Retriever factory ─────────────────────────────────────────────────────────

def _get_retriever():
    settings = get_settings()
    embeddings = get_embeddings()

    vector_store = Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=embeddings,
        persist_directory=str(settings.chroma_persist_dir),
    )

    # MMR = Maximum Marginal Relevance
    # Balances similarity to the query with diversity among retrieved docs.
    # This reduces redundant chunks that say the same thing slightly differently.
    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": settings.retriever_top_k,
            "fetch_k": settings.retriever_top_k * 3,  # MMR candidate pool
            "lambda_mult": 1 - settings.mmr_diversity_score,  # 1=similarity, 0=diversity
        },
    )


# ── Source extraction ─────────────────────────────────────────────────────────

def _extract_sources(docs: list[Document]) -> list[dict]:
    """Deduplicate and format source references from retrieved chunks."""
    seen = set()
    sources = []
    for doc in docs:
        meta = doc.metadata
        key = (meta.get("source_file", "unknown"), meta.get("chunk_index", 0))
        if key not in seen:
            seen.add(key)
            sources.append({
                "file": meta.get("source_file", "unknown"),
                "chunk_index": meta.get("chunk_index"),
                "total_chunks": meta.get("total_chunks"),
                "page": meta.get("page"),
                "preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
            })
    return sources


# ── Main query function ───────────────────────────────────────────────────────

def query_rag(
    question: str,
    chat_history: Optional[list[tuple[str, str]]] = None,
) -> RAGResponse:
    """
    Run a single RAG query.

    Args:
        question: The user's question.
        chat_history: List of (human_msg, ai_msg) tuples for conversation context.

    Returns:
        RAGResponse with answer, sources, and retrieved chunks.
    """
    settings = get_settings()
    chat_history = chat_history or []

    logger.info(f"RAG query: {question[:100]}...")

    llm = _get_llm()
    retriever = _get_retriever()

    # ── Retrieve relevant chunks ──────────────────────────────────
    retrieved_docs = retriever.invoke(question)
    logger.debug(f"Retrieved {len(retrieved_docs)} chunks")

    if not retrieved_docs:
        return RAGResponse(
            answer="No relevant documents found. Please upload documents first.",
            sources=[],
            retrieved_chunks=[],
            query=question,
        )

    # ── Assemble context string ───────────────────────────────────
    context_parts = []
    for i, doc in enumerate(retrieved_docs):
        file_name = doc.metadata.get("source_file", "unknown")
        chunk_idx = doc.metadata.get("chunk_index", i)
        context_parts.append(f"[Source: {file_name}, chunk {chunk_idx}]\n{doc.page_content}")
    context = "\n\n---\n\n".join(context_parts)

    # ── Format chat history ───────────────────────────────────────
    history_text = ""
    for human, ai in chat_history[-5:]:  # Last 5 turns max
        history_text += f"Human: {human}\nAssistant: {ai}\n\n"

    # ── Build and invoke prompt ───────────────────────────────────
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
        HumanMessagePromptTemplate.from_template(HUMAN_PROMPT),
    ])

    chain = prompt | llm
    response = chain.invoke({
        "context": context,
        "chat_history": history_text,
        "question": question,
    })

    answer = response.content
    sources = _extract_sources(retrieved_docs)

    logger.info(f"Answer generated ({len(answer)} chars, {len(sources)} sources)")

    return RAGResponse(
        answer=answer,
        sources=sources,
        retrieved_chunks=retrieved_docs,
        query=question,
    )
