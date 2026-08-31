from __future__ import annotations

import json
import argparse

from src.config import settings

from pathlib import Path
from elasticsearch import Elasticsearch, helpers
from src.llm_client import get_embedding

from openai import OpenAI

openai_client = OpenAI(api_key=settings.openai_api_key)

def create_index(es: Elasticsearch, vector_dimension) -> None:
    """Create the Elasticsearch index and mapping for the Q&A dataset."""
    if es.indices.exists(index=settings.index_name):
        es.indices.delete(index=settings.index_name)

    mapping = {
        "mappings": {
            "properties": {
                "q": {"type": "text"},
                "a": {"type": "text"},
                "q_vector": {
                    "type": "dense_vector",
                    "dims": vector_dimension,
                    "index": True,
                    "similarity": "cosine",
                }
            }
        }
    }
    es.indices.create(index=settings.index_name, body=mapping)


def load_dataset(dataset_path: Path) -> dict:
    with dataset_path.open("r", encoding="utf-8") as dataset:
        return json.load(dataset)


def index_documents(src_dataset) -> None:
    """Read Q&A pairs, embed each question, and bulk index them into Elasticsearch."""
    es = Elasticsearch(settings.elasticsearch_host)

    create_index(es, settings.vector_dimension)

    dataset = load_dataset(src_dataset)
    questions = dataset.get("questions", [])
    actions = []

    for item in questions:
        question_text = item["q"]
        answer_text = item["a"]

        vector = get_embedding(question_text, client=openai_client)
        actions.append(
            {
                "_index": settings.index_name,
                "_source": {
                    "q": question_text,
                    "a": answer_text,
                    "q_vector": vector,
                },
            }
        )

    if actions:
        helpers.bulk(es, actions)
        print("[INDEXING] Completed")


def parse_args() -> argparse.Namespace:
    default_dataset = Path(__file__).resolve().parents[1] / "data" / "qa_data.json"
    parser = argparse.ArgumentParser(description="Index a Q&A dataset into Elasticsearch.")
    parser.add_argument(
        "src_dataset", nargs="?", type=Path, default=default_dataset,
        help="Path to the JSON dataset to index."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    index_documents(args.src_dataset)

