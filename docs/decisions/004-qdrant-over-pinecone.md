# ADR 004: Self-Hosted Qdrant for RAG Retrieval

**Status**: Accepted  
**Date**: 2024-06-01

### Context
Phase 6 of this project introduced a modular Retreival-Augmented Generation (RAG) pipeline capable of Dense (vector) and Hybrid retrieval. We needed a vector database to store document embeddings.

Hosted solutions like Pinecone or Weaviate Cloud offer generous free tiers but require creating external accounts, managing separate sets of API keys, and eventually hitting index limits or inactivity pauses.

### Decision
We use **Qdrant** running locally via Docker (or deployed alongside the backend container).

### Consequences
- **Positive:** Complete data control.
- **Positive:** No vendor lock-in or external API dependencies. Qdrant's Rust-based engine is highly optimized and consumes effectively zero overhead on idle.
- **Negative:** We must manage the persistent volume storage ourselves in the deployment environment.
- **Negative:** Not inherently globally distributed (unlike Pinecone), though latency is irrelevant since the vector DB sits directly next to the backend FastAPI service.
