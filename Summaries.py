import os
import pandas as pd
import streamlit as st
import glob
st.title("PDF Summarization Web App")

with st.expander("About"):
    st.write(
        """
        This is a web app that summarizes a pdf paper on the behavior of Concrete Face Rockfill Dams.
        They are organized by category and can be selected from the dropdown menu. 
        The summary was generated using the Chat GPT API from OpenAI. The app is not meant to replace 
        reading the original paper, but rather to help guide the user in selecting 
        a particular paper to read.
        """
        )

df = pd.read_csv("summary2.csv")

if "cat_index" not in st.session_state:
    st.session_state["cat_index"] = 0

cat_index = st.session_state["cat_index"]

# st.write(f"cat_index: {cat_index}")
# st.write(f"Number of categories: {len(df['Category'].unique())}")
# Dropdown menu keeping track of the category so that 
# it is persistent across pages
categories = df["Category"].unique()
category = st.selectbox("Select a category", categories, index=int(cat_index))
cat_index = categories.tolist().index(category)
st.session_state["cat_index"] = cat_index
# st.write(f"cat_index: {cat_index}")

df_selected = df[df['Category'] == category]

if "file_index" not in st.session_state:
    st.session_state["file_index"] = 0
    file_index = 0
else:
    file_index = st.session_state["file_index"]

files = df_selected["title"].unique()    
file = st.selectbox(
    "Select a file from a total of " + str(len(df_selected)) + " files",
    files,
    index=int(file_index)
    )

file_index = files.tolist().index(file)
st.session_state["file_index"] = file_index

summary = df_selected[df_selected['title'] == file]["summary"].values[0]
st.success(summary)

pdf_file = df_selected[df_selected['title'] == file]["file_path"].values[0]

st.session_state["pdf_file"] = pdf_file
st.session_state["title"] = df_selected[df_selected['file_path'] == pdf_file]["title"].values[0]
st.session_state["summary"] = summary

st.write("The original file can be found here: " + pdf_file)

st.write()