import PyPDF2
import os
import pandas as pd
import streamlit as st
import glob
st.title("PDF Summarization Web App")

with st.expander("About"):
    st.info(
        """
        This is a web app that summarizes a pdf paper on the behavior of Concrete Face Rockfill Dams.
        On the Questions page you can ask questions about the paper and get an answer based only
        on the paper provided, avoiding the dreaded hallucinations of Large Language Models.
        The papers are organized by category and can be selected from the dropdown menu. 
        The summaries were generated using the Chat GPT API from OpenAI. 
        
        The app is not meant to substitute reading the original paper, but rather to help guide the user in selecting 
        a particular paper to read. 
        
        In the Upload page the user can upload a pdf file and the app will extract the images from it
        and generate a summary.
        """
        )
    
import os
import PyPDF2
def extract_images(pdf_file):
    folder = pdf_file.split("\\")[-2]
    file_name = pdf_file.split("\\")[-1].replace(".pdf", "")
    pdf_file = open(pdf_file.replace('\\', os.sep), 'rb')
    pdf_reader = PyPDF2.PdfReader(pdf_file)

    # Create the images directory if it doesn't exist
    if not os.path.exists('images'):
        os.makedirs('images')

    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        if '/XObject' in page['/Resources']:
            x_objects = page['/Resources']['/XObject'].get_object()
            if x_objects:
                for obj in x_objects:
                    if x_objects[obj]['/Subtype'] == '/Image':
                        # Extract the image data
                        image_data = x_objects[obj]._data
                        # Write the image data to a file in the images directory
                        # Check if folder exists
                        if not os.path.exists(f"images/{folder}/{file_name}"):
                            os.makedirs(f"images/{folder}/{file_name}")
                        filename = f"images/{folder}/{file_name}/image_{page_num}_{obj.replace('/', '_')}.png"
                        with open(filename, 'wb') as f:
                            f.write(image_data)


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

with st.expander("Summary"):
    st.success(summary)

pdf_file = df_selected[df_selected['title'] == file]["file_path"].values[0]

# Extract images from pdf
extract_images(pdf_file)

# display the image in streamlit
folder = pdf_file.split("\\")[-2]
file_name = pdf_file.split("\\")[-1].replace(".pdf", "")

image_files = glob.glob(f"images/{folder}/{file_name}/*.png")

# Filter image files to those with size > 100 KB
st.write("There are " + str(len(image_files)) + " images in this paper")
# Sort these files from largest to smallest
image_files.sort(key=os.path.getsize, reverse=True)

st.write("There are " + str(len(image_files)) + " images in this paper")
st.write(f"len(image_files)//3 {len(image_files)//3}")
# Display 9 images in a 3 x 3 grid
with st.expander("All images from the paper"):
    cols = {}
    for i in range(len(image_files)//3):
        cols[i] = st.columns(3)
        for j in range(3):
            image_i= i*3 + j
            try:
                cols[i][j].image(image_files[image_i])
            except: 
                print(f"Error displaying image {image_files[image_i]}")
                pass

st.session_state["pdf_file"] = pdf_file
st.session_state["title"] = df_selected[df_selected['file_path'] == pdf_file]["title"].values[0]
st.session_state["summary"] = summary

file_path = r"https://github.com/jstello/streamlit-langchain/blob/master/" + pdf_file.replace('\\', '/')
file_path = file_path.replace(' ', '%20')

st.markdown(f"The original file can be found [here]({file_path})")

