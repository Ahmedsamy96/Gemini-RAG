import os
from datasets import Dataset
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

from dotenv import load_dotenv
load_dotenv()

# Load API keys
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Define paths and filenames
static_pdf_file = r"C:/Users/ahmed/OneDrive/Documents/Gemini_RAG/data/Actual Budget Report 2022.pdf"

# Load PDF text
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PyPDFLoader(pdf)
        pages = pdf_reader.load_and_split()
        for page in pages:
            text += page.page_content
    return text

# Split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=0)
    return text_splitter.split_text(text)

# Generate vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    vector_store.save_local("faiss_index")

# Load conversation chain
def get_conversation_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context.
    If the answer is not available in the context, just say, "Answer is not available in the context."
    
    Context:
    {context}
    
    Question: 
    {question}
    
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.6)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Process user question
def user_input(user_question, docs):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    # Use similarity search to retrieve relevant documents based on user question
    docs = new_db.similarity_search(user_question)
    chain = get_conversation_chain()
    # Generate response based on the retrieved documents and user question
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

questions = ["What did the president say about Justice Breyer?",
                 "What did the president say about Intel's CEO?",
                 "What did the president say about gun violence?"]

ground_truths = ["The president said that Justice Breyer has dedicated his life to serve the country and thanked him for his service.",
                "The president said that Pat Gelsinger is ready to increase Intel's investment to $100 billion.",
                "The president asked Congress to pass proven measures to reduce gun violence."]


# Main function
def main(questions=questions):
    # Load PDF text
    raw_text = get_pdf_text([static_pdf_file])
    # Split text into chunks
    text_chunks = get_text_chunks(raw_text)
    # Generate vector store
    get_vector_store(text_chunks)
    
    answers = []
    contexts = []
    for question in questions:
        # Process each question and retrieve the answer
        answer = user_input(question, None)  # Since we're not using Weaviate for retrieval, set docs=None
        # Questions to ask
        answers.append(answer)
        # Retrieve context (for demonstration purposes, using the first chunk)
        contexts.append(text_chunks[0])  # Adjust as needed based on the actual context you want to provide
    return answers, contexts


# Execute main function
answers, contexts = main()
# Convert contexts to a sequence of strings
#contexts = [{"context": context} for context in contexts]
#contexts = [context["context"] for context in contexts]  # Extract context strings

# To dict
data = {
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths
}

# Convert dict to dataset
dataset = Dataset.from_dict(data)
print(dataset)


result = evaluate(
    dataset = dataset, 
    metrics=[
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
    ],
)

df = dataset.to_pandas()
df.to_csv("./Evaluation_file.csv")
print("Done")