import streamlit as st
import openai
from openai import OpenAI as OpenAIClient
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
import os

st.set_page_config(page_title="Nomads Co-Pilot", page_icon="💡", layout="centered")

# Simple password gate
st.sidebar.title("Admin Access")
password = st.sidebar.text_input("Enter password", type="password")
if password != "nomads2025":
    st.warning("Please enter a valid password to continue.")
    st.stop()

# Set up OpenAI API key
try:
    api_key = st.secrets["OPENAI_API_KEY"]
except (FileNotFoundError, KeyError):
    api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("OPENAI_API_KEY not found. Please set it in Streamlit secrets or as an environment variable.")
    st.stop()

# Instantiate the OpenAI client for potential direct calls (though RAG is primary)
# This client is different from the LLM used by LlamaIndex
# direct_openai_client = OpenAIClient(api_key=api_key)

# Load documents and create index
@st.cache_resource
def load_index():
    reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
    docs = reader.load_data()
    # Configure LlamaIndex Settings
    Settings.llm = LlamaOpenAI(model="gpt-4", api_key=api_key)
    Settings.embed_model = OpenAIEmbedding(api_key=api_key)
    index = VectorStoreIndex.from_documents(docs)
    return index

index = load_index()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("Nomads Co-Pilot")
st.caption("Your AI assistant for Nomads strategy and insights.")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if user_query := st.chat_input("Ask about Nomads strategy, regulations, or insights..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # This is the primary interaction via LlamaIndex RAG
                query_engine = index.as_query_engine(streaming=True)
                response_stream = query_engine.query(user_query)
                response_content = st.write_stream(response_stream.response_gen)
                # The problematic line `response = openai.ChatCompletion.create(...)` has been removed.
                # The RAG output `response_content` is now directly used.

            except openai.APIError as e: # Catch specific OpenAI errors
                st.error(f"OpenAI API Error: {e}. Please check your API key and model access.")
                response_content = "Sorry, I encountered an API error with OpenAI."
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
                response_content = "Sorry, I encountered an unexpected error."
        st.session_state.messages.append({"role": "assistant", "content": response_content})
