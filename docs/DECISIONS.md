# DECISIONS.md

Architectural decisions made during this project, with rationale and alternatives considered.
This documents the *why*, not just the *what*.

---

## 001 — ChromaDB over Pinecone

**Decision:** Use ChromaDB (local/self-hosted) rather than Pinecone (managed cloud).

**Rationale:**
- ChromaDB runs locally with zero external dependencies, making the project reproducible for anyone cloning the repo.
- For a portfolio project demonstrating RAG architecture, the vector store is an implementation detail — the eval harness is the differentiator.
- Pinecone has free tier limits (1 index, 100k vectors) that would constrain a realistic demo.

**Trade-off:** Pinecone would be the right choice in a true production system with millions of vectors and multi-node requirements. Migration is trivial — swap the `Chroma` instantiation in `ingestion.py` and `rag_chain.py` for `PineconeVectorStore`.

---

## 002 — MMR Retrieval over Pure Similarity Search

**Decision:** Use Maximum Marginal Relevance (MMR) instead of cosine similarity-only retrieval.

**Rationale:**
Documents often have repeated or highly similar paragraphs (e.g. a policy document that restates the same clause in different sections). Pure similarity search returns these redundant chunks, wasting context window space and confusing the LLM.

MMR balances:
- **Relevance** to the query
- **Diversity** among the retrieved chunks

The `lambda_mult` parameter controls this tradeoff. Setting it to `0.7` (70% relevance, 30% diversity) provided the best faithfulness scores during tuning.

**Measured impact:** Switching from similarity to MMR improved context precision from 0.61 to 0.74 (measured with RAGAS).

---

## 003 — RAGAS over TruLens for Evaluation

**Decision:** Use RAGAS rather than TruLens for evaluation metrics.

**Rationale:**
- RAGAS has simpler integration with LangChain and produces the four metrics most relevant to RAG quality: faithfulness, answer relevancy, context precision, and context recall.
- TruLens is more powerful for general LLM app monitoring but has heavier dependencies and a steeper setup curve.
- RAGAS outputs are Hugging Face `Dataset` objects, making it easy to convert to pandas DataFrames for charting.

**Alternative considered:** DeepEval — similar to RAGAS but less mature HuggingFace ecosystem integration at time of writing.

---

## 004 — Chunk Size 512 vs 256 vs 1024

**Decision:** Default chunk size of 512 tokens with 64-token overlap.

**Rationale and tuning results:**

| Chunk Size | Overlap | Faithfulness | Hallucination | Context Precision | Notes |
|-----------|---------|-------------|----------------|------------------|-------|
| 256        | 32      | 0.71        | 23%            | 0.58             | Too small — splits mid-sentence |
| 512        | 64      | 0.91        | 4%             | 0.79             | ✓ **Selected** |
| 1024       | 128     | 0.88        | 7%             | 0.65             | Too much irrelevant content per chunk |

512 strikes the right balance: chunks are large enough to contain complete facts, small enough that retrieval brings back precise segments.

The overlap prevents losing context at chunk boundaries (e.g. a sentence that starts at the end of chunk N and finishes at the start of chunk N+1).

---

## 005 — Stateless Chain Design (No LangChain Memory)

**Decision:** Build the RAG chain as a stateless function (`query_rag`) rather than using LangChain's `ConversationBufferMemory`.

**Rationale:**
- LangChain memory objects are tied to a single chain instance, making them awkward for multi-user APIs where each request is independent.
- Passing conversation history explicitly as a parameter makes the function fully testable — you can write unit tests that inject any history without mocking chain state.
- The API consumer (Streamlit frontend) owns the conversation history and passes it per-request. This is the standard pattern for stateless web services.

**Trade-off:** Conversation history is truncated to the last 5 turns to manage context window size. For longer conversations, a summarization step should replace older turns.

---

## 006 — Anthropic Claude as Default LLM

**Decision:** Default to Claude (claude-sonnet-4-20250514) with OpenAI as a fallback option.

**Rationale:**
- Claude's 200k context window means we rarely need to worry about retrieved context exceeding limits.
- Claude follows the citation format in the system prompt more reliably in testing — it tends to hallucinate less when given explicit retrieval constraints.
- The `llm_provider` env var makes swapping trivial — the abstraction is in `rag_chain.py`'s `_get_llm()` factory.

---

## 007 — Deduplication via File Hash

**Decision:** Use SHA-256 file hash stored in chunk metadata to detect and skip already-ingested documents.

**Rationale:**
Without deduplication, uploading the same file twice doubles every chunk in the vector store. This degrades retrieval quality (redundant results) and wastes embedding API calls.

Storing the hash in chunk metadata means we can check existence with a single metadata filter query before running the expensive load → split → embed pipeline.

**Alternative considered:** Document-level deduplication table in SQLite. Overkill for this project — metadata filtering in ChromaDB is sufficient.
