import streamlit as st

st.title("CFRD Knowledge Base")
st.info("""I am a knowledgeable chatbot with extensive information about Concrete Face Rockfill Dams (CFRDs). 
        I can answer questions about CFRDs, and I can also summarize the contents of a PDF file.""")
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

temperature = st.slider("Temperature", 0.0, 1.0, 0.5, 0.1)
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

query = st.text_input("Ask me a question about CFRDs", "What is a CFRD?")
@st.cache_data()
def run_query(query):
    result = qa({"query": query, "top_k": 5, "max_tokens": 2000})
    return result["result"], result["source_documents"]

if st.button("Ask"):
        result, sources = run_query(query)
        st.success(result)
        # Print sources
        for source in sources:
            file_name = source.metadata['source']
            file_name = os.path.basename(file_name)
            with st.expander("Source: " + file_name):
                st.write(source.page_content.replace("\n", " "))
