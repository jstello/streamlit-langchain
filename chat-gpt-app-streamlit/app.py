import os
from openai import OpenAI, AzureOpenAI
import streamlit as st
st.title("🤖 Funes the Memorious 📚")
with st.sidebar:
    model = st.selectbox(
        "Choose a model", ("gpt-3.5-turbo", "gpt-4", "gpt-4-0613", "gpt-4-32k", "gpt-4-32k-0613", "gpt-3.5-turbo-16k", "gpt-3.5-turbo-0613", "gpt-3.5-turbo-16k-0613")
    )

with st.expander("About this app"):
    st.info("""
            🤖 Welcome to Funes the memorious, your AI-powered chatbot inspired by the incredible mind of Jorge Luis Borges! 📚

Just like the unforgettable character from Borges' story, I am here to assist you in effortlessly analyzing and understanding your PDF documents. Funes, known for his remarkable memory, serves as the inspiration behind my name and purpose.

In Borges' story "Funes, the Memorious," the protagonist possessed an extraordinary memory that enabled him to remember every detail of his life. Similarly, I aim to provide you with a comprehensive understanding of your uploaded PDFs. By leveraging the power of artificial intelligence, I summarize the key points and answer your questions, ensuring that you grasp the essence of the documents swiftly.

Think of me as your digital assistant, working tirelessly to simplify the complexities of your PDF materials. Whether you are a student, researcher, or professional, I am here to aid you in accessing the information you need promptly and efficiently.

So, feel free to upload your PDFs and let me accompany you on your journey of exploration and comprehension. Ask me anything about the content, request summaries, or seek clarification on specific details - all inspired by the concept of Funes' extraordinary memory.

Together, let's unlock the knowledge within your documents and embark on a seamless and intellectually stimulating experience. Explore. Understand. Remember. Welcome to Funes he memorious! 🌟
            """)
    
# client = OpenAI()
client = AzureOpenAI(
    api_version="2023-07-01-preview",
    azure_endpoint="https://ai-proxy.lab.epam.com",
    api_key=os.environ["DIAL_API_KEY"],
)

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model="gpt-35-turbo-16k",
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})