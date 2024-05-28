import os
import time
import numpy as np
import pandas as pd

import streamlit as st
import weaviate
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai

from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

WEAVIATE_API_KEY = st.secrets["WEAVIATE_API_KEY"]
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=GOOGLE_API_KEY)

static_pdf_file_1 = r"./data/AhmedSamy-DataScientist2024 (1).pdf"
static_pdf_file_2 = r"./data/Profile.pdf"

@st.cache_resource
def stream_data(text_input="Error With the system"):
    for word in text_input.split(" "):
        yield word + " "
        time.sleep(0.02)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PyPDFLoader(pdf)
        pages = pdf_reader.load_and_split()
        for page in pages:
            text += page.page_content
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=0)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    vector_store.save_local("faiss_index")

def get_conversation_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperatue=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversation_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write(stream_data(text_input=f"""{response["output_text"]}"""))

def main():
    st.set_page_config(page_title="Chat PDF", page_icon=":speech_balloon:")
    st.header("Hello, I'm Ahmed Samy -Chat with Me 👨‍🏫😀")
    image_path = 'data/BG.jpeg'
    st.image(image_path, width=700)

    user_question = st.text_input("Ask a Question:")
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("🙋🙋")
        static_files = [static_pdf_file_1, static_pdf_file_2]

        # Add an image in the sidebar
        sidebar_image_path = './data/samy.jpg'
        st.image(sidebar_image_path, width=300)
        
        #if st.button("Submit & Process"):
        #with st.spinner("Processing..."):
        raw_text = get_pdf_text(static_files)
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)
        st.success("Done")

if __name__ == "__main__":
    main()
