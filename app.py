import streamlit as st
from openai import OpenAI as OpenAIClient
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
import os

st.set_page_config(page_title="Nomads Co-Pilot", page_icon="💡", layout="centered")

# Simple password gate
st.sidebar.title("🔒 Admin Access")
password = st.sidebar.text_input("Enter password", type="password")
if password != "nomads2025":
    st.warning("Please enter a valid password to continue.")
    st.stop()

# Set up OpenAI API key
api_key_val = None
try:
    api_key_val = st.secrets.get("OPENAI_API_KEY")
    if not api_key_val:
        api_key_val = os.getenv("OPENAI_API_KEY")
except (FileNotFoundError, KeyError):
    api_key_val = os.getenv("OPENAI_API_KEY")

if not api_key_val:
    st.error("OPENAI_API_KEY not found. Please set it in Streamlit secrets (e.g., .streamlit/secrets.toml) or as an environment variable.")
    st.stop()

# Instantiate the OpenAI client for direct API calls
client = OpenAIClient(api_key=api_key_val)

# Load documents and create index (assumes there's a "data" folder with PDFs)
@st.cache_resource
def load_index():
    reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
    docs = reader.load_data()
    # Configure LlamaIndex Settings globally
    Settings.llm = LlamaOpenAI(model="gpt-4", api_key=api_key_val)
    Settings.embed_model = OpenAIEmbedding(api_key=api_key_val)
    index = VectorStoreIndex.from_documents(docs)
    return index

index = load_index()
query_engine = index.as_query_engine()

# Define system prompts for personas
personas = {
    "Sanne de Vries (Municipality)": "You are Sanne de Vries, Sustainability and Waste Innovation Lead at Gemeente Utrecht. Your priorities are public trust, circular systems, and regulatory compliance. Be cautious but curious, and incorporate the provided documents into your reasoning.",
    "Leon Müller (E-Retailer)": "You are Leon Müller, Category Manager for Baby & Health at Zalando Green Hub. You are data-driven, trend-sensitive, and pragmatic. Use both data from the uploaded documents and your own commercial lens.",
    "Isabelle Fournier (Retail Buyer)": "You are Isabelle Fournier, Senior Buyer for Baby & Family at Carrefour France. Your focus is retail success, pricing logic, and logistics. Reference the documents when appropriate.",
    "Daniela Rossi (Competitor)": "You are Daniela Rossi, Sustainability and Innovation Director at Pampers Italy. You think long-term and strategically. If any of the data could affect your market position, reflect that.",
    "Jeroen Bakker (Waste Expert)": "You are Jeroen Bakker, Technical Director at AVR Waste-to-Energy in Rotterdam. Focus on practicality and technical integration. Use any technical or regulatory data from the documents to support your thinking."
}

# Streamlit UI
st.title("Nomads Synthetic Stakeholder Chat with RAG")

selected_persona_name = st.selectbox("Choose a synthetic stakeholder to chat with:", list(personas.keys()))
user_input = st.text_area("Your message:", height=150)

if st.button("Send"):
    if user_input:
        persona_system_prompt = personas[selected_persona_name]
        
        # Get context from RAG query engine
        try:
            context_response_obj = query_engine.query(user_input)
            context_text = str(context_response_obj)
        except Exception as e:
            st.error(f"Error querying LlamaIndex: {e}")
            st.stop()

        # Construct the full prompt for the system role, similar to the old logic
        full_prompt_for_system_role = f"{persona_system_prompt}\n\nRelevant info from documents: {context_text}\n\nUser's query: {user_input}"

        try:
            # Use the new OpenAI client an_d chat completions create method
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": full_prompt_for_system_role}
                ]
            )
            assistant_response = response.choices[0].message.content
            st.markdown(f"**{selected_persona_name}**: {assistant_response}")

        except Exception as e:
            st.error(f"Error communicating with OpenAI: {e}")
            # For more detailed debugging locally:
            # import traceback
            # st.error(traceback.format_exc())

    else:
        st.warning("Please enter a message.")
