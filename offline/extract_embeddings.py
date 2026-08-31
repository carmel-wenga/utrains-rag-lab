from __future__ import annotations

import json
import streamlit as st

from pathlib import Path
from src.llm_client import get_embedding


def main() -> None:
    dataset_path = Path(__file__).resolve().parents[1] / "data" / "qa_data.json"
    with dataset_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    for index, item in enumerate(payload.get("questions", []), start=1):
        question = item["q"]
        vector = get_embedding(question)
        print(f"{index}. {question}")
        print(f"   model: {st.secrets['EMBEDDING_MODEL']}")
        print(f"   dimensions: {len(vector)}")
        print()


if __name__ == "__main__":
    main()
