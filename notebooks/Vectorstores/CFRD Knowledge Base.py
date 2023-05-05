import streamlit as st


# set up pinecone
import pinecone
import os
from tqdm.auto import tqdm

pinecone.init(api_key=os.getenv("PINECONE_API_KEY"))

index_name = 'icold'

index = pinecone.Index(name=index_name)