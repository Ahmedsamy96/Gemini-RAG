import os
import streamlit as st
import weaviate
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter #library to split pdf files

from langchain_google_genai import GoogleGenerativeAIEmbeddings #to embed the text
import google.generativeai as genai

from langchain_community.vectorstores import FAISS #for vector embeddings
from langchain_google_genai import ChatGoogleGenerativeAI #
from langchain.chains.question_answering import load_qa_chain #to chain the prompts
from langchain.prompts import PromptTemplate #to create prompt templates

WEAVIATE_API_KEY = st.secrets["WEAVIATE_API_KEY"]
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=GOOGLE_API_KEY)

@st.cache_data

def get_pdf_text(pdf_docs):
    text = ""
    # iterate over all pdf files uploaded
    for pdf in pdf_docs:
        pdf_reader = PyPDFLoader(pdf)
        pages = pdf_reader.load_and_split()

        # iterate over all pages in a pdf
        for page in pages:
            text += page.page_content
    return text



def get_text_chunks(text):
    # create an object of RecursiveCharacterTextSplitter with specific chunk size and overlap size
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 10000, chunk_overlap = 0)
    # now split the text we have using object created
    chunks = text_splitter.split_text(text)

    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001") # google embeddings
    vector_store = FAISS.from_texts(text_chunks,embeddings) # use the embedding object on the splitted text of pdf docs
    vector_store.save_local("faiss_index") # save the embeddings in local

def get_conversation_chain():

    # define the prompt
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model = "gemini-pro", temperatue = 0.3) # create object of gemini-pro
    prompt = PromptTemplate(template = prompt_template, input_variables= ["context","question"])
    chain = load_qa_chain(model,chain_type="stuff",prompt = prompt)
    return chain

def user_input(user_question):
    # user_question is the input question
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")

    # load the local faiss db
    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)

    # docs = text_splitter.split_documents(documents)
    # auth_config = weaviate.AuthApiKey(api_key=WEAVIATE_API_KEY)
    # weaviate_client = weaviate.Client(url="https://gemini-rag-y70wzfwq.weaviate.network",auth_client_secret=auth_config)
    # new_db = WeaviateVectorStore.from_documents(docs, embeddings, client=weaviate_client)

    # using similarity search, get the answer based on the input
    docs = new_db.similarity_search(user_question)

    chain = get_conversation_chain()


    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])

def main():
    st.set_page_config(
        page_title="Gemini PDF Chatbot",
        page_icon="🤖"
    )

    # Sidebar for uploading PDF files
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

    # Main content area for displaying chat messages
    st.title("Chat with PDF files using Gemini🤖")
    st.write("Welcome to the chat!")
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)


if __name__ == "__main__":
    main()

