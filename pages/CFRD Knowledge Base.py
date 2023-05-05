import streamlit as st

st.title("CFRD Knowledge Base")
st.info("""I am a knowledgeable chatbot with extensive information about Concrete Face Rockfill Dams (CFRDs). 
        I can answer questions about CFRDs, and I can also summarize the contents of a PDF file.""")
# set up pinecone
import pinecone
import os
from tqdm.auto import tqdm
from langchain.embeddings.openai import OpenAIEmbeddings

pinecone.init(api_key='48640420-7e79-46d4-b71d-d07286818fef', environment='us-central1-gcp')

index_name = 'icold'

embeddings = OpenAIEmbeddings()

index = pinecone.Index(index_name, embeddings)

query = st.text_input("Ask me a question about CFRDs", "")
query = "how to estimate leakage through a cfrd?"
if query != "":
    results = index.query(queries=[query], top_k=1)
    st.write(results[0].text)