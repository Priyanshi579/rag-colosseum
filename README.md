---
title: RAG Colosseum
emoji: 🏛️
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# RAG Colosseum

FastAPI retrieval-augmented question answering service for the 50-passage knowledge base.

## Endpoint

`POST /query`

```json
{"question":"How many nodes are required for a Raft quorum?","top_k":3}
```

The service returns JSON with `answer`, `sources`, `scores`, and `confidence`. The embedding model is downloaded at startup and cached in `/data`. The first cold start may take several minutes.

Health check: `GET /health`

