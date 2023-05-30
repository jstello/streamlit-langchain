import streamlit as st
import pandas as pd
import glob
import os

# list csv files in parent directory


summaries = pd.read_csv("summary2.csv")

# st.dataframe(summaries)
# get the base name of file_name

st.title("CFRD Knowledge Base")
st.info("""
    I am a knowledge base with extensive information about Concrete Face Rockfill Dams (CFRDs).
    Based on a series of scientific papers spanning the topics of 
    * Numerical Modelling
    * Dam Behavior
    * Seismic Response
    * Rockfill Behavior
    * Structural Analysis
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

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pinecone.init(environment="us-central1-gcp", api_key=PINECONE_API_KEY)

def pretty_print(response):
    import textwrap
    # Split the response by lines of max 80 characters
    return '\n'.join(textwrap.wrap(response, 80))

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

temperature = 0.5

llm=ChatOpenAI(temperature=temperature, max_tokens=1000)

docsearch = Pinecone.from_existing_index('icold', embeddings, 
                                         )

# index = pinecone.Index(index_name, embeddings)

# st.info(info)

qa = RetrievalQA.from_chain_type(
    llm=llm,
#     chain_type="map_reduce",
    chain_type="stuff",
    retriever=docsearch.as_retriever(),
    return_source_documents=True,
    verbose=True,
    )

query = st.text_input("Ask me a question about CFRDs",
                      "What type of damage did the Zipingpu dam endure during the 2008 earthquake?")
@st.cache_data()
def run_query(query):
    result = qa({"query": query, "top_k": 4, "max_tokens": 1000})
    return result["result"], result["source_documents"]

# st.dataframe(summaries.head())

if st.button("Ask"):
        result, sources = run_query(query)
        st.success(result)
        # Print sources
        for isource, source in enumerate(sources):
            file_name = source.metadata['source']
            file_name = os.path.basename(file_name)
            base_name = file_name.split("\\")[-1]
            try:
                i = [i for i in range(len(summaries)) if base_name in summaries.iloc[i, 0]][0]
                title = summaries.iloc[i, 1]
            except:
                title = base_name

            with st.expander(f"Source fragment [{isource+1}] from file" + title):
                st.write(source.page_content.replace("\n", " "))

