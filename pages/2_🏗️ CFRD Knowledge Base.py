import streamlit as st
import pandas as pd
import glob
import os

# list csv files in parent directory


summaries = pd.read_csv("summary2.csv")
# get the base name of file_name

st.title("CFRD Knowledge Base")
st.info("""
    I am a knowledge base with extensive information about Concrete Face Rockfill Dams (CFRDs).
    Based on a series of scientific papers spanning the topics of 
    * Numerical Modelling
    * Dam Behavior
    * Seismic Response
    * Rockfill Behavior
    * Structural
        """)
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.document_loaders import TextLoader
from langchain.document_loaders import UnstructuredPDFLoader
import os
embeddings = OpenAIEmbeddings()
import pinecone 
import os
from tqdm.autonotebook import tqdm
# initialize pinecone
pinecone.init(api_key="48640420-7e79-46d4-b71d-d07286818fef",
              environment="us-central1-gcp")

def pretty_print(response):
    import textwrap
    # Split the response by lines of max 80 characters
    return '\n'.join(textwrap.wrap(response, 80))

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

temperature = 0.5

llm=ChatOpenAI(temperature=temperature, max_tokens=2000)

docsearch = Pinecone.from_existing_index('icold', embeddings, 
                                         )
qa = RetrievalQA.from_chain_type(
    llm=llm,
#     chain_type="map_reduce",
    chain_type="stuff",
    retriever=docsearch.as_retriever(),
    return_source_documents=True,
    verbose=True,
    )

query = st.text_input("Ask me a question about CFRDs", "What type of damage did the Zipingpu dam endure during the 2008 earthquake?")
@st.cache_data()
def run_query(query):
    result = qa({"query": query, "top_k": 5, "max_tokens": 2000})
    return result["result"], result["source_documents"]

if st.button("Ask"):
        result, sources = run_query(query)
        st.success(result)
        # Print sources
        for isource, source in enumerate(sources):
            file_name = source.metadata['source']
            file_name = os.path.basename(file_name)
            base_name = file_name.split("\\")[-1]
            i = [i for i in range(len(summaries)) if base_name in summaries.iloc[i, 0]][0]
            title = summaries.iloc[i, 1]

            with st.expander(f"[{isource+1}] " + title):
                st.write(source.page_content.replace("\n", " "))

