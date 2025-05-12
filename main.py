import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFaceHub
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

def getLLamaresponse(input_text, no_words, blog_style):
    # Initialize the HuggingFace model
    llm = HuggingFaceHub(repo_id='tiiuae/falcon-7b-instruct', model_kwargs={"temperature":0.01, "max_new_tokens":256})
    llm.client.api_url = '//Your url'
    
    # Define prompt template
    template = """
        Write a blog for {blog_style} job profile for a topic {input_text}
        within {no_words} words.
    """
    prompt = PromptTemplate(input_variables=["blog_style", "input_text", 'no_words'], template=template)
    
    # Generate response
    response = llm(prompt.format(blog_style=blog_style, input_text=input_text, no_words=no_words))
    return response

# Streamlit app configuration
st.set_page_config(page_title="Blogs Generating", page_icon='🤖', layout='centered', initial_sidebar_state='collapsed')

st.header("Generate Blogs 🤖")

# Input fields
input_text = st.text_input("Enter the Blog Topic")

# Create columns for additional inputs
col1, col2 = st.columns([5, 5])
with col1:
    no_words = st.text_input('No of Words')
with col2:
    blog_style = st.selectbox('Writing the blog for', ('Researchers', 'Data Scientist', 'Common People'), index=0)

# Generate Button
submit = st.button("Generate")

# Display the result if the button is clicked
if submit:
    st.write(getLLamaresponse(input_text, no_words, blog_style))
