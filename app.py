from __future__ import annotations

import streamlit as st

from src.llm_client import create_openai_client, call_llm
from src.rag import search


def _normalize_search_mode(label: str) -> str:
    if label == "Vector Search (Embeddings)":
        return "vector"
    return "full_text"


def main() -> None:
    st.set_page_config(
        page_title="Utrains Support Chatbot",
        page_icon="🎓",
        layout="centered"
    )

    st.title("Utrains Support Chatbot")
    st.caption("Ask questions about training programs, support, schedules, and platform access.")

    # Sidebar
    with st.sidebar:
        st.title("Support Assistant")
        st.markdown("Powered by **OpenAI**")

        openai_api_key = st.secrets["OPENAI_API_KEY"]
        st.markdown(f"OpenAI key loaded: {'**YES**' if openai_api_key else '**NO**'}")
        st.divider()

        # display Elasticsearch Instance
        elasticsearch_host = st.secrets["ELASTICSEARCH_HOST"]
        st.markdown("Vector Database Instance")
        st.markdown(f"Elasticsearch: {elasticsearch_host}")

        st.divider()
        llm_model_name = st.radio(
            "Choose LLM Model",
            ["gpt-4o-mini", "gpt-4o"],
            index=0,
        )

        st.divider()
        search_mode_label = st.radio(
            "Search Mode",
            ["Vector Search", "Full-Text Search"],
            index=0,
        )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "openai_client" not in st.session_state:
        # Init and store llm interaction and session to the streamlit session
        openai_client = create_openai_client()
        if openai_client:
            st.session_state.openai_client = openai_client
        else:
            st.error("Failed to initialize OpenAI client. Please check your API key.")
            st.stop()

    for item in st.session_state.chat_history:
        if item["role"] == "user":
            with st.chat_message("user"):
                st.write(item["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(item["content"])
                if item.get("context"):
                    with st.expander("Show retrieved context", expanded=False):
                        for index, result in enumerate(item["context"], start=1):
                            st.markdown(f"**Result {index}**")
                            st.write(f"Question: {result['question']}")
                            st.write(f"Answer: {result['answer']}")
                            if result.get("score") is not None:
                                st.caption(f"Score: {result['score']:.4f}")

    # 1. Get user prompt from the chat input box
    user_question = st.chat_input("Ask a question about a Utrains training program")

    if user_question:
        mode = _normalize_search_mode(search_mode_label)

        # add user question to the chat history and display on the UI
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)


        with st.spinner("Executing…"):
            # 2. search relevant context from the indexed Q&A data using the selected search mode
            retrieved_chunks = search(str(user_question), mode=mode, top_k=3)

            # 3. Use the retrieved context to call the OpenAI. If the context is empty, return a message indicating that
            # the answer cannot be provided.
            if retrieved_chunks:
                answer = call_llm(
                    user_question=str(user_question), # User Prompt
                    context_chunks=retrieved_chunks,  # The retrieved context
                    llm_model=llm_model_name,
                    client=st.session_state.openai_client
                )
            else:
                answer = "I couldn't find enough context in the indexed Q&A data to answer that question confidently."

        answer = {
            "role": "assistant",
            "content": answer,
            "context": retrieved_chunks,
        }
        st.session_state.chat_history.append(answer)

        with st.chat_message("assistant"):
            st.markdown(answer["content"])
            if answer.get("context"):
                with st.expander("Show retrieved context", expanded=False):
                    for index, result in enumerate(answer["context"], start=1):
                        st.markdown(f"**Result {index}**")
                        st.write(f"Question: {result['question']}")
                        st.write(f"Answer: {result['answer']}")
                        if result.get("score") is not None:
                            st.caption(f"Score: {result['score']:.4f}")

if __name__ == "__main__":
    main()
