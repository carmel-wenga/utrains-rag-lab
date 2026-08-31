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


def full_text_search(query: str, top_k: int = 3) -> list[dict]:
    """Search both the question and answer text using Elasticsearch match queries."""
    body = {
        "size": top_k,
        "query": {
            "bool": {
                "should": [
                    {"match": {"q": query}},
                    {"match": {"a": query}},
                ],
                "minimum_should_match": 1,
            }
        },
    }
    response = client.search(index=st.secrets["INDEX_NAME"], body=body)
    return [_serialize_hit(hit) for hit in response.get("hits", {}).get("hits", [])]


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


def search(query: str, mode: str = "Vector Search", top_k: int = 3) -> list[dict]:
    """Return the relevant Q&A records in the requested search mode."""
    if mode == "Full-Text Search":
        return full_text_search(query, top_k=top_k)
    return vector_search(query, top_k=top_k)
