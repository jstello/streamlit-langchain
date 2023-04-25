if 1==1:  # Imports
    import PyPDF2
    import sys
    print(sys.executable)
    import os
    from langchain.chat_models import ChatOpenAI
    import streamlit as st
    from langchain.chains.question_answering import load_qa_chain
    from langchain.embeddings.openai import OpenAIEmbeddings
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.text_splitter import CharacterTextSplitter
    from langchain.document_loaders import UnstructuredPDFLoader
    from langchain.vectorstores import Chroma
    import glob
    from io import StringIO
    from langchain.chains.summarize import load_summarize_chain

st.title("Chat with your pdf")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
text_splitter = CharacterTextSplitter(
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


if uploaded_file is not None:
    if 1==1:  # Load and split text
        loader = UnstructuredPDFLoader(r"C:\Users\juantello\Downloads\CFRD - New Bulletin.pdf")
        @st.cache_data
        def load_data():
            return loader.load()
    st.write("Loading data...")
    data = load_data()
#     # Remove the references section
    data[0].page_content = data[0].page_content.split('References')[0]
    @st.cache_data
    def split_documents(_data):
        return text_splitter.split_documents(data)

    texts = split_documents(data)  # texts is a list 
    st.write(f"Number of documents: {len(texts)}")
        
    embeddings = OpenAIEmbeddings()

    run_summarization = st.button("Run summarization")
    
    if run_summarization:
        llm = ChatOpenAI(temperature=0.5, max_tokens=400)
        chain = load_summarize_chain(llm, chain_type="map_reduce")
        st.write("Running summarization...")
        summary = chain.run(texts[:5])
        st.success(summary)

    # st.write(f"type(texts): {type(texts)}")
    # st.write(f"type(texts[0]): {type(texts[0])}")
    db = Chroma.from_documents(texts, embeddings)

from langchain.chains import RetrievalQA
if 1==1:  # Sample questions
   
    llm = ChatOpenAI(temperature=0.5, max_tokens=400)
    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(),
        chain_type="stuff",
        retriever=db.as_retriever(),
        return_source_documents=True,
        )
    [c1, c2] = st.columns([2, 1])
    
    lang = c2.selectbox("Answer in: ", ["en", "es", "fr", "de", "it", "pt", "ru", "sv", "tr", "zh"])
    
    query = c1.text_input("Ask a question of this PDF: (Puede ser en español!)", "")
    
    if query != "":  # Ask questions of the paper
        @st.cache_data
        def run_retrieval(query):
            return qa({"query": query + f". Answer in {lang} language", "n": 1})
        st.write("Thinking...")
        result = run_retrieval(query)
        st.success(result["result"])
        st.markdown(f"[Source document]({result['source_documents'][0].metadata['source']})")
# if 1==1:  # Display images
#     # display the image in streamlit
#     folder = pdf_file.split("\\")[-2]
#     file_name = pdf_file.split("\\")[-1].replace(".pdf", "")

#     image_files = glob.glob(f"images/{folder}/{file_name}/*.png")

#     # Filter image files to those with size > 100 KB

#     # Sort these files from largest to smallest
#     image_files.sort(key=os.path.getsize, reverse=True)

#     # Display 9 images in a 3 x 3 grid
#     with st.expander("Images from the paper"):
#         for i in range(3):
#             cols = st.columns(3)
#             for j in range(3):
#                 image_i= i*3 + j
#                 try:
#                     cols[j].image(image_files[image_i])
#                 except: 
#                     pass
            


# query = st.text_input("Ask a question of this PDF: (Puede ser en español!)", "")
# if query != "":  # Ask questions of the paper
#     docs = db.similarity_search(query)
#     # docs = docsearch.similarity_search(query, include_metadata=True)
#     response = chain.run(input_documents=docs, question=query)
#     st.success(pretty_print(response))
#     e1 = st.expander("Relevant fragments from the PDF:")    
#     for i, doc in enumerate(docs):
#         e1.write(f"Relevant document # {i}:")
#         e1.write(doc.page_content.replace("\n\n", "\n "))

    
# e2 = st.expander("Full text head")
# e2.write(data[0].page_content.replace("\n\n", "\n ")[0:10000])
    