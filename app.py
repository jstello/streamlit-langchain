import pandas as pd
import streamlit as st
st.title("PDF Summarization Web App")

with st.expander("About"):
    st.write("This is a web app that summarizes a pdf file. It is built using streamlit and pdfplumber. It is a part of the Langchain project.")
    
df = pd.read_csv("summary.csv")

# Dropdown menu
category = st.selectbox("Select a category", df["Category"].unique())

df_selected = df[df['Category'] == category]

file = st.selectbox("Select a file from a total of " + str(len(df_selected)) + " files", df_selected["title"].unique())

st.success(df_selected[df_selected['title'] == file]["summary"].values[0])

st.write("The original file can be found here: " + df_selected[df_selected['title'] == file]["file_path"].values[0])

st.write()