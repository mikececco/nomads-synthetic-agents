
import streamlit as st
import openai
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, ServiceContext
from llama_index.llms import OpenAI

# Set up OpenAI API
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Simple password gate
st.sidebar.title("🔒 Access")
password = st.sidebar.text_input("Enter password", type="password")
if password != "nomads2025":
    st.warning("Please enter a valid password to continue.")
    st.stop()

# Load documents and create index (assumes there's a "data" folder with PDFs)
@st.cache_resource
def load_index():
    reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
    docs = reader.load_data()
    service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-4"))
    index = VectorStoreIndex.from_documents(docs, service_context=service_context)
    return index

index = load_index()
query_engine = index.as_query_engine()

# Define system prompts
personas = {
    "Sanne de Vries (Municipality)": "You are Sanne de Vries, Sustainability and Waste Innovation Lead at Gemeente Utrecht. Your priorities are public trust, circular systems, and regulatory compliance. Be cautious but curious, and incorporate the provided documents into your reasoning.",
    "Leon Müller (E-Retailer)": "You are Leon Müller, Category Manager for Baby & Health at Zalando Green Hub. You are data-driven, trend-sensitive, and pragmatic. Use both data from the uploaded documents and your own commercial lens.",
    "Isabelle Fournier (Retail Buyer)": "You are Isabelle Fournier, Senior Buyer for Baby & Family at Carrefour France. Your focus is retail success, pricing logic, and logistics. Reference the documents when appropriate.",
    "Daniela Rossi (Competitor)": "You are Daniela Rossi, Sustainability and Innovation Director at Pampers Italy. You think long-term and strategically. If any of the data could affect your market position, reflect that.",
    "Jeroen Bakker (Waste Expert)": "You are Jeroen Bakker, Technical Director at AVR Waste-to-Energy in Rotterdam. Focus on practicality and technical integration. Use any technical or regulatory data from the documents to support your thinking."
}

# Streamlit UI
st.title("Nomads Synthetic Stakeholder Chat with RAG")

persona = st.selectbox("Choose a synthetic stakeholder to chat with:", list(personas.keys()))
user_input = st.text_area("Your message:", height=150)

if st.button("Send"):
    if user_input:
        system_prompt = personas[persona]
        context_response = query_engine.query(user_input)
        full_prompt = f"{system_prompt}\n\nRelevant info: {context_response}\n\nUser: {user_input}"

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": full_prompt}
            ]
        )
        st.markdown(f"**{persona}**: {response['choices'][0]['message']['content']}")
    else:
        st.warning("Please enter a message.")
