# Arabic Chat PDF - Interactive Document Query System

![alt text](https://github.com/Ahmedsamy96/Gemini-RAG/blob/main/data/BG.jpeg)

**Project Overview**
  
* **Description:** Chat PDF is a web application that allows users to ask Arabic questions in natural language about the contents of uploaded PDFs. The system leverages advancements in generative AI and vector embeddings to deliver precise answers directly extracted from PDF documents.

<hr>

**Features**

* **PDF Ingestion:** Upload and process PDF documents for information retrieval.
* **Text Extraction:** Accurately extract text content from uploaded PDFs.
* **Text Chunking:** Split extracted text into manageable segments for efficient processing.
* **Text Embedding:** Generate mathematical representations (embeddings) of text chunks using Google's generative AI for semantic similarity.
* **Vector Indexing:** Store and efficiently retrieve text embeddings using FAISS, a high-performance library for similarity search.
* **Natural Language Understanding (NLU):** Utilize generative AI models to comprehend user queries and generate relevant responses based on the document context.
* **Interactive User Interface (UI):** User-friendly interface built with Streamlit for seamless interaction with the system.

<hr>

**Technology Stack**

* **Streamlit:** A Python framework for rapidly developing web applications.
* **Weaviate (Optional):** A vector database for storing and querying embeddings (not used in the current implementation).
* **Google Generative AI:** Google's suite of AI services for generating text embeddings and answering questions in context.
* **FAISS:** A vector database for storing and querying embeddings for efficient similarity search of vector embeddings
* **PyPDFLoader:** A Python library for loading and extracting text from PDF files.
* **LangChain:** A toolkit designed for building language model chains products and services in a simpler way.

<hr>

### **Getting Started**

**Prerequisites**

* A Google Cloud account with access to Generative AI services ([https://cloud.google.com/ai-platform/docs/technical-overview](https://cloud.google.com/ai-platform/docs/technical-overview))
* Python 3.8 or later ([https://www.python.org/downloads/](https://www.python.org/downloads/))

**Installation**

1. Clone the repository:

   ```bash
   git clone https://github.com/Ahmedsamy96/Gemini-RAG
   cd chat-pdf
   ```

2. Install required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up your API keys in Streamlit's secret management:

   * Navigate to your Streamlit app settings.
   * Add `WEAVIATE_API_KEY` (if using Weaviate) and `GOOGLE_API_KEY` to the secrets section.

**Usage**

1. Start the Streamlit application:

   ```bash
   streamlit run Rag_system.py
   ```

2. Navigate to the provided local URL (usually http://localhost:8501).
3. 2 PDF documents that contain our business are already uploaded in the interface, just select one of them.
4. Enter your question in the text box and submit it to receive an answer based on the document's content.

<hr>

### **Code Structure**

**Main Components**

* **PDF Text Extraction:** The `get_pdf_text` function handles loading and reading text content from PDFs using the `PyPDFLoader` library.
* **Text Processing:** The `get_text_chunks` function utilizes `langchain`'s `RecursiveCharacterTextSplitter` to split the extracted text into manageable chunks for further processing.
* **Vector Embedding and Storing:** The `get_vector_store` function generates embeddings for the text chunks using Google's generative AI and stores them using FAISS for efficient retrieval.
* **Query Handling:** The `user_input` function takes a user query as input, retrieves relevant information using vector similarity search, and generates a response using a generative AI model chained with a prompt template.
<hr>

### **Workflow**

1. **Initialization:** Load configurations and API keys.
2. **Document Processing:** When a PDF is uploaded or selected, the system extracts and processes the text content.
3. **Index Creation:** Text embeddings are generated and stored using FAISS to create a searchable index.
4. **User Interaction:** Users can ask questions through the web interface. The system retrieves relevant information from the indexed embeddings and generates a response using the conversational AI model (Gemini Pro).

<hr>

### **The ways i held to improve the performance of the system:**
- **Using a better pdf parser:** One of the best ways to get better inference is reading the source of data correctly, so I tried many tools to find the best one for inference (PyPDFLoader).
- **Selection of a better LLM model that does well with Arabic text:** This will help you use the tokenizer of the model to transfer learn it not to start from a bad point -Bad means the Model is not trained on Arabic data well- as I may use a super powerful model in English but weak on Arabic text which is misleading and deceptive.
- **Select a good Vector DB:** One of the most familiar ones is FAISS it was good enough to proceed with it. (Replacements that I tried: Qdrant - Chroma - Weaviate)
- **Prompt Engineering:** Very important factor that is not hard but has a very high impact on the system response.

<hr>

**Model Inference**
The model in the trial is distinguished by 2 things:
- It does not elaborate on the answer. The easy, no-nonsense response
- It gives correct answers most of the time, even if they are from tables in most experiments.

![alt text](https://github.com/Ahmedsamy96/Gemini-RAG/blob/main/data/Inference.png)

<hr>

**FAQs**

* **How does the system handle multiple PDFs?**
   - The UI allows users to select which PDF they want to query against. Each PDF's text is processed and indexed separately.
* **What types of questions can I ask?**
   - Chat PDF is designed to answer factual questions based on the content of the loaded PDF documents. The accuracy of the answers depends on the quality of the phrasing and the clarity of the information within the PDFs.
* **Is there a limit to the size of the PDF?**
   - Large PDFs are handled by splitting the text into smaller chunks. However, extremely large documents are not available till now we limit them to 200 MB as large ones may require more processing time.

