# Utrains RAG Lab

This lab is a hands-on project that demonstrates a beginner-friendly Retrieval-Augmented Generation (RAG) workflow for a UTrains support chatbot. Students can index a small Q&A dataset into Elasticsearch, use full-text or vector search, and ask an LLM to answer using only the retrieved context.

## Project goals

- Store support questions and answers in Elasticsearch.
- Generate embeddings for the question text with OpenAI.
- Search using either full-text search or vector similarity.
- Pass the retrieved context to an LLM to produce a grounded answer.
- Run the whole system locally with Docker Compose.

## Project structure

```text
utrains-rag-lab/
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
├── .env.example
├── README.md
├── data/
│   └── qa_data.json
├── offline/
│   ├── index_data.py
│   └── extract_embeddings.py
├── app.py
├── .streamlit/
│   ├── config.toml
│   └── secrets.toml
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── llm_client.py
│   ├── llm-client.py
│   ├── prompts.py
│   ├── rag.py
│   ├── streamlit_ui.py
│   ├── streamlit-ui.py
│   └── ui.py
```

## 1. Install dependencies with UV

From the project root:

```bash
cd utrains-rag-lab
uv sync
```

If you do not have `uv` installed yet, install it first:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 2. Configure the project secrets

Create a local secrets file in `.streamlit/secrets.toml` and add your OpenAI API key and connection settings:

```toml
openai_api_key = "your_openai_api_key_here"
elasticsearch_host = "http://localhost:9200"
elasticsearch_username = ""
elasticsearch_password = ""
index_name = "utrains-qa"
embedding_model = "text-embedding-3-small"
llm_model = "gpt-4o-mini"
vector_dimension = 1536
streamlit_port = 8501
```

In the app code, values are read with `st.secrets["variable_name"]` and the rest of the project falls back to this file when running outside Streamlit.

## 3. Start Elasticsearch locally

### Option A: Using Docker Compose

```bash
docker compose up -d elasticsearch
```

This launches a single-node Elasticsearch instance in no-security mode for the lab.

### Option B: Local Elasticsearch

If you have Elasticsearch running already, make sure it is reachable at `http://localhost:9200`.

## 4. Index the sample Q&A data

The offline script reads `data/qa_data.json`, embeds each question using `text-embedding-3-small`, and bulk-loads the records into Elasticsearch.

```bash
docker compose exec streamlit uv run python -m offline.index_data data/source_dataset.json
```

You should see logs similar to:

- creating the `utrains-qa` index,
- embedding each question,
- bulk indexing each document,
- a total count of indexed records.

## 5. Run the Streamlit app

From the root folder:

```bash
uv run streamlit run app.py
```

The app will open in your browser. Use the sidebar to switch between:

- Vector Search (Embeddings)
- Full-Text Search (BM25)

Then ask questions such as:

- "What is the duration of the DevOps training program?"
- "Does UTrains provide internship placement support?"
- "How do I reset my learner account password?"

The app shows both the chatbot response and the retrieved context in an expandable section so students can inspect what the model saw.

## 6. Run the full Docker stack

To start both Elasticsearch and the Streamlit app in one command:

```bash
docker compose up --build
```

This is the easiest setup for the classroom environment. The app waits for Elasticsearch health checks to pass before starting.

## 7. Optional: inspect embeddings

You can preview the generated embeddings for the dataset:

```bash
uv run python offline/extract_embeddings.py
```

This script prints the question and the number of dimensions in the embedding vector.

## 8. How the RAG flow works

1. The user sends a question to the app.
2. The app chooses either full-text or vector search.
3. Elasticsearch retrieves the most relevant Q&A chunks.
4. Those chunks are added to the prompt as context.
5. The OpenAI model answers using only the context it has received.
6. The UI displays the response and the retrieved context for transparency.

## Troubleshooting

### OpenAI API key error

- Make sure `.streamlit/secrets.toml` contains a valid `openai_api_key` value.
- Ensure the key is not blank and is active for your OpenAI account.

### Elasticsearch connection error

- Confirm Elasticsearch is running on `http://localhost:9200`.
- If using Docker Compose, check `docker compose ps` and `docker compose logs elasticsearch`.

### No results returned

- Re-run the indexing script after checking the dataset file.
- Confirm the `utrains-qa` index exists in Elasticsearch.

## Notes for students

This project is intentionally simple and easy to read. The focus is on understanding the RAG pattern:

- retrieve relevant documents,
- add them to the prompt,
- let the model answer with those facts only.

This is a great foundation for more advanced RAG experiments such as chunking, reranking, hybrid search, and production-grade retrieval workflows.
