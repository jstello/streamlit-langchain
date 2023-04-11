import pandas as pd
import streamlit as st
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

# Dropdown menu
category = st.selectbox("Select a category", df["Category"].unique())

df_selected = df[df['Category'] == category]

file = st.selectbox("Select a file from a total of " + str(len(df_selected)) + " files", df_selected["title"].unique())

summary = df_selected[df_selected['title'] == file]["summary"].values[0]
st.success(summary)

pdf_file = df_selected[df_selected['title'] == file]["file_path"].values[0]

st.session_state["pdf_file"] = pdf_file
st.session_state["title"] = df_selected[df_selected['file_path'] == pdf_file]["title"].values[0]
st.session_state["summary"] = summary

st.write("The original file can be found here: " + pdf_file)

st.write()