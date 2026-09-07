from __future__ import annotations

from elasticsearch import Elasticsearch

from src.llm_client import get_embedding

import streamlit as st


client = Elasticsearch(hosts=st.secrets["ELASTICSEARCH_HOST"])


def _serialize_hit(hit: dict) -> dict:
    source = hit.get("_source", {})
    return {
        "question": source.get("q", ""),
        "answer": source.get("a", ""),
        "score": hit.get("_score"),
    }


def vector_search(query: str, top_k: int = 3) -> list[dict]:
    """Embed the query and use cosine similarity to retrieve the nearest Q&A records."""
    embedding = get_embedding(query)
    response = client.search(
        index=st.secrets["INDEX_NAME"],
        knn={
            "field": "q_vector",
            "query_vector": embedding,
            "k": top_k,
            "num_candidates": max(top_k * 10, 20),
        },
        source=["q", "a"],
    )
    return [_serialize_hit(hit) for hit in response.get("hits", {}).get("hits", [])]

