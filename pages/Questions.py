import PyPDF2
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
import firebase_admin
from firebase_admin import credentials, db



if 1==1:  # Authenticate firebase
        
    @st.cache(allow_output_mutation=True)
    def init_firebase():
        cred = credentials.Certificate('cfrd-questions-firebase-adminsdk-saer6-926a4198fb.json')
        firebase_app = firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://cfrd-questions.firebaseio.com'
        })
        return firebase_app

    firebase_app = init_firebase()
    database_ref = db.reference('interactions', app=firebase_app)



files = glob.glob("ICOLD - CFRD New Bulletin 2023/**/*.pdf", recursive=True)

llm = ChatOpenAI(temperature=0.5, max_tokens=1000)
chain = load_qa_chain(llm, chain_type="stuff", verbose=True)

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

if 1==1:  # Session state variables

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
    if "summary" not in st.session_state:
        st.session_state["summary"] = "No summary"
        summary = st.session_state["summary"]
    else:
        summary = st.session_state["summary"]

if 1==1:  # Link to file    
    st.title(title.replace('"', ''))
    # Provide a link to the file on github
    file_path = r"https://github.com/jstello/streamlit-langchain/blob/master/" + pdf_file.replace('\\', '/')
    # Replace spaces with %20
    file_path = file_path.replace(' ', '%20')

    with st.expander("Link to file:"):
        st.markdown(f"The original file can be found [here]({file_path})")

if 1==1:  # Load and split text
    loader = UnstructuredPDFLoader(pdf_file.replace('\\', os.sep))
    def load_data():
        return loader.load()

    data = load_data()
    # Remove the references section
    data[0].page_content = data[0].page_content.split('References')[0]

    def split_documents(_data):
        return text_splitter.split_documents(data)

    texts = split_documents(data)


db = Chroma.from_documents(texts, embeddings)

def pretty_print(response):
    import textwrap
    # Split the response by lines of max 80 characters
    return '\n'.join(textwrap.wrap(response, 80))


with st.expander("Summary"):
    st.write(summary)

if 1==1:  # Sample questions
    query = """
    Think of 10 technical questions relevant to dam engineering one could ask about this context, 
    meaning that the answer is contained within it, and return them with 
    double spaces between them in a bullet list.
    """
    docs = db.similarity_search(query)
    sample_questions = chain.run(input_documents=docs, question=query)
    with st.expander("Sample questions"):
        st.write(sample_questions)

if 1==1:  # Display images
    # display the image in streamlit
    folder = pdf_file.split("\\")[-2]
    file_name = pdf_file.split("\\")[-1].replace(".pdf", "")

    image_files = glob.glob(f"images/{folder}/{file_name}/*.png")

    # Filter image files to those with size > 100 KB

    # Sort these files from largest to smallest
    image_files.sort(key=os.path.getsize, reverse=True)

    # Display 9 images in a 3 x 3 grid
    with st.expander("Images from the paper"):
        for i in range(3):
            cols = st.columns(3)
            for j in range(3):
                image_i= i*3 + j
                try:
                    cols[j].image(image_files[image_i])
                except: 
                    pass
            


query = st.text_input("Ask a question of this PDF: (Puede ser en español!)", "")
if query != "":  # Ask questions of the paper
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
    