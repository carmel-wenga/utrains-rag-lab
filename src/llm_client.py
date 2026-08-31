from __future__ import annotations

from openai import OpenAI
from openai.types.responses import EasyInputMessageParam

import streamlit as st

SYSTEM_PROMPT = """You are a helpful support assistant for Utrains training programs.
Answer the user's question using only the context provided below.

Rules:
- Use the retrieved context as your only source of truth.
- If the context contains the answer, respond clearly and concisely.
- If the answer is not present in the context, say you do not have enough information from the provided course material.
- Do not invent program details, policies, or timing.
- Prefer a direct answer, with a short note when the response is based on retrieved support context.

Context:
{context}
"""

def create_openai_client() -> OpenAI:
    """Create and return an OpenAI client using the API key from Streamlit secrets."""
    return OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


def get_embedding(text: str, client:OpenAI | None = None) -> list[float]:
    """Generate an embedding for a single input string using the configured OpenAI model."""
    active_client = client or create_openai_client()
    print(f"[EMBEDDINGS] Get Embeddings For: {text}")
    response = active_client.embeddings.create(model=st.secrets["EMBEDDING_MODEL"], input=text)
    return response.data[0].embedding


def call_llm(
        user_question: str,
        context_chunks: list[dict],
        llm_model: str,
        client:OpenAI | None = None
    ) -> str:
    """Generate a grounded answer using the retrieved context chunks."""

    # Prepare the retrieved Q&A records as context
    context = "\n\n---\n\n".join([f"Q: {chunk['question']}\nA: {chunk['answer']}" for chunk in context_chunks])

    # create openai client if None
    active_client = client or create_openai_client()

    # ask the LLM to answer the question.
    messages: list[EasyInputMessageParam] = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT.format(context=context),
        },
        {
            "role": "user",
            "content": user_question,
        },
    ]

    response = active_client.responses.create(
        model=llm_model,
        input=messages,
    )

    return response.output_text
