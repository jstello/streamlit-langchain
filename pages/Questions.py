import sys
print(sys.executable)
import os
from langchain.chat_models import ChatOpenAI
import streamlit as st
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.vectorstores import Chroma
import glob

files = glob.glob("ICOLD - CFRD New Bulletin 2023/**/*.pdf", recursive=True)

llm = ChatOpenAI(temperature=0, max_tokens=2000)
chain = load_qa_chain(llm, chain_type="stuff")

embeddings = OpenAIEmbeddings()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000, 
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
    
st.title(title.replace('"', ''))
# Provide a link to the file on github
file_path = r"https://github.com/jstello/streamlit-langchain/blob/ask-questions-3.5-turbo/" + pdf_file.replace('\\', os.sep)

with st.expander("File Name:"):
    st.write(file_path)

loader = UnstructuredPDFLoader(pdf_file.replace('\\', os.sep))
def load_data():
    return loader.load()

data = load_data()
# Remove the references section
data[0].page_content = data[0].page_content.split('References')[0]

def split_documents(_data):
    return text_splitter.split_documents(data)

texts = split_documents(data)

# initialize pinecone
# pinecone.init(environment='us-central1-gcp', api_key=os.environ['PINECONE_API_KEY'])
# index_name = "icold"

# # Check if the index already exists
# if not pinecone.index_exists(index_name):
#     # Create the Pinecone index
#     pinecone.create_index(index_name, embeddings_dim=embeddings.shape[1])
#     pinecone.index(index_name, [t.page_content for t in texts], embeddings)

# Retrieve the Pinecone index
# docsearch = pinecone.Index(index_name)

db = Chroma.from_documents(texts, embeddings)

def pretty_print(response):
    import textwrap
    # Split the response by lines of max 80 characters
    return '\n'.join(textwrap.wrap(response, 80))

if "summary" not in st.session_state:
    st.session_state["summary"] = "No summary"
    summary = st.session_state["summary"]
else:
    summary = st.session_state["summary"]

st.markdown("## Summary")
st.success(summary)



query = st.text_input("Ask a question of this PDF: (Puede ser en español!)", "")
if query != "":
    docs = db.similarity_search(query)
    # docs = docsearch.similarity_search(query, include_metadata=True)
    response = chain.run(input_documents=docs, question=query)
    st.success(pretty_print(response))
    e1 = st.expander("Relevant fragments from the PDF:")    
    for i, doc in enumerate(docs):
        e1.write(f"Relevant document # {i}:")
        e1.write(doc.page_content.replace("\n\n", "\n "))

    
e2 = st.expander("Full text head")
e2.write(data[0].page_content.replace("\n\n", "\n ")[0:10000])
    