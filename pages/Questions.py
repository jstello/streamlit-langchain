import sys
print(sys.executable)
import os
from langchain.chat_models import ChatOpenAI
import streamlit as st
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.vectorstores import Pinecone
import pinecone

os.environ['PINECONE_API_KEY'] = "48640420-7e79-46d4-b71d-d07286818fef"



llm = ChatOpenAI(temperature=0, openai_api_key="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", max_tokens=800)
chain = load_qa_chain(llm, chain_type="stuff")

embeddings = OpenAIEmbeddings(openai_api_key="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")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=20,
    # separators=[
    #     'Abstract',
    #     'Introduction',
    #     'Conclusions',
    #     '\n\n',
    #     '\n',
    #     ' ',
    #     '']
    )

if "pdf_file" not in st.session_state:
    st.session_state["pdf_file"] = files[0]
    pdf_file = st.session_state["pdf_file"]
else:
    pdf_file = st.session_state["pdf_file"]

if 'title' not in st.session_state:
    st.session_state["title"] = "No title"
    title = st.session_state["title"]
else:
    title = st.session_state["title"]
    
st.title(title)

loader = UnstructuredPDFLoader(pdf_file)
@st.cache_data()
def load_data():
    return loader.load()

data = load_data()

@st.cache_data()
def split_documents(_data):
    return text_splitter.split_documents(data)

texts = split_documents(data)

# initialize pinecone
pinecone.init(environment='us-central1-gcp', api_key=os.environ['PINECONE_API_KEY'])
index_name = "icold"

# Check if the index already exists
if not pinecone.index_exists(index_name):
    # Create the Pinecone index
    pinecone.create_index(index_name, embeddings_dim=embeddings.shape[1])
    pinecone.index(index_name, [t.page_content for t in texts], embeddings)

# Retrieve the Pinecone index
docsearch = pinecone.Index(index_name)


def pretty_print(response):
    import textwrap
    # Split the response by lines of max 80 characters
    return '\n'.join(textwrap.wrap(response, 80))



summary = st.session_state["summary"]

st.markdown("## Summary")
st.success(summary)

query = st.text_input("Ask a question of this PDF: ", "")
# @st.cache_data()
def run_chain(query, docs):
    return chain.run(input_documents=docs, question=query)

if query != "":
    docs = docsearch.similarity_search(query, include_metadata=True)
    # response = run_chain(query, docs)
    # st.success(pretty_print(response))
    e1 = st.expander("Relevant documents")    
    for i, doc in enumerate(docs):
        e1.write(f"Relevant document # {i}:")
        e1.write(doc.page_content.replace("\n\n", "\n "))

    
e2 = st.expander("Full text")
e2.write(data[0].page_content.replace("\n\n", "\n "))
    