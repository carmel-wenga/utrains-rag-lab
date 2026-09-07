from __future__ import annotations

import json

from pathlib import Path

from src.config import settings
from src.llm_client import create_embedding_model, get_embedding


def main() -> None:
    dataset_path = Path(__file__).resolve().parents[1] / "data" / "qa_data.json"
    embedding_model = create_embedding_model()

    with dataset_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    for index, item in enumerate(payload.get("questions", []), start=1):
        question = item["q"]
        vector = get_embedding(question, embedding_model=embedding_model)
        print(f"{index}. {question}")
        print(f"   model: {settings.embedding_model}")
        print(f"   dimensions: {len(vector)}")
        print()


if __name__ == "__main__":
    main()
