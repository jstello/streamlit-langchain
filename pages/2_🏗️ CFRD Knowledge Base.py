import streamlit as st
import pandas as pd
import glob
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import pinecone 

def initialize_pinecone():
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    pinecone.init(environment="us-central1-gcp", api_key=PINECONE_API_KEY)
    return pinecone

def get_index(index_name):
    return pinecone.Index(index_name)

def get_embeddings():
    return OpenAIEmbeddings()

def get_vectorstore(index, embed):
    return Pinecone.from_existing_index('icold', embed)

def get_chat_model():
    return ChatOpenAI(temperature=0.5, max_tokens=1000)

def get_retrieval_qa(llm, vectorstore):
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=True,
        verbose=True,
    )

# @st.cache
def run_query(qa, query):
    result = qa({"query": query, "top_k": 4, "max_tokens": 1000})
    return result["result"], result["source_documents"]

def main():
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

    summaries = pd.read_csv("summary2.csv")

    initialize_pinecone()
    index = get_index('icold')
    embed = get_embeddings()
    vectorstore = get_vectorstore(index, embed)
    llm = get_chat_model()
    qa = get_retrieval_qa(llm, vectorstore)

    query = st.text_input("Ask me a question about CFRDs",
                          "What type of damage did the Zipingpu dam endure during the 2008 earthquake?")

    if st.button("Ask"):
        result, sources = run_query(qa, query)
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

if __name__ == "__main__":
    main()
