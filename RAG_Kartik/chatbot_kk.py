
import os
import re
import logging
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document, HumanMessage
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.base import BaseCallbackHandler
from chromadb.config import Settings

# Ensure telemetry is disabled before importing Chroma or its dependencies
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Callback handler for logging LLM events
class LoggingCallbackHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs):
        logger.info(f"New token generated: {token}")
    
    def on_chain_start(self, serialized: dict, inputs: dict, **kwargs):
        logger.info(f"Chain started with inputs: {inputs}")

    def on_chain_end(self, outputs: dict, **kwargs):
        logger.info(f"Chain ended with outputs: {outputs}")

# Load PDF and extract text
def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        return ""

# Process extracted text into LangChain Documents
def process_pdf_to_documents(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    return [Document(page_content=text, metadata={"source": pdf_path})]

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# Chunk documents
def chunk_documents(documents):
    return [
        Document(page_content=chunk, metadata=doc.metadata)
        for doc in documents
        for chunk in text_splitter.split_text(doc.page_content)
    ]

# Initialize HuggingFace embedding model with CUDA support
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Configure Chroma vector store
settings = Settings(
    persist_directory="./chroma_store",  # Directory for local persistence
    anonymized_telemetry=False,         # Disable telemetry
)

# vectorstore = Chroma(
#     embedding_function=embedding_model,
#     persist_directory="./chroma_store",  # Ensure persist directory matches settings
# )

vectorstore = Chroma(
    embedding_function=embedding_model,
    persist_directory="./chroma_store",  # Ensure persist directory matches settings
    client_settings=settings  # Explicitly pass the Settings object
)


# Enhanced Prompt Template for ChatGPT-like Responses
CHATGPT_RAG_TEMPLATE = """
You are a friendly and highly knowledgeable assistant trained to provide detailed and informative answers.
Use the chat history and provided context to answer thoroughly and in a conversational tone.
Ensure your answers are clear, engaging, and helpful.

- Your answer should be approximately 40 lines long, providing detailed information and covering multiple aspects of the question.
- Include examples, explanations, or additional relevant information to meet the length requirement.
- Keep a warm, approachable, and empathetic tone while remaining professional.

Chat History:
{history}

Context:
{context}

Answer the following question:
{question}
"""

# Updated ChatPromptTemplate
chatgpt_rag_prompt = ChatPromptTemplate.from_template(CHATGPT_RAG_TEMPLATE)

# Updated enforce_line_count function
def enforce_line_count(response, target_lines=40):
    """Ensure the response is truncated to approximately 40 lines without adding filler text."""
    lines = response.split('\n')
    return '\n'.join(lines[:target_lines])

# Initialize Llama model with CUDA
llm = ChatOllama(
    model="llama3.2:1b",
    device="cuda"  # Use CUDA for LLaMA inference
)

# Initialize conversation memory
memory = ConversationBufferMemory(return_messages=True)

# Normalize query
def normalize_query(query):
    """Normalize the query by removing special characters, converting to lowercase, and stripping extra spaces."""
    query = query.lower()
    query = re.sub(r'\s+', ' ', query)
    query = re.sub(r'[^\w\s]', '', query)
    return query.strip()

# Generate detailed response
def generate_detailed_response(question):
    normalized_question = normalize_query(question)
    docs = vectorstore.as_retriever().invoke(normalized_question)
    context = "\n\n".join(doc.page_content for doc in docs)
    
    # Format the prompt for a detailed response
    formatted_prompt = chatgpt_rag_prompt.format(
        history=memory.chat_memory.messages,
        context=context,
        question=normalized_question
    )
    
    # Use CUDA with mixed precision for faster inference
    from torch.cuda.amp import autocast
    with autocast():
        response = llm.invoke([HumanMessage(content=formatted_prompt)])
    
    # Enforce line count
    detailed_response = enforce_line_count(response.content, target_lines=40)
    
    # Update chat memory for maintaining conversation history
    memory.chat_memory.add_user_message(question)
    memory.chat_memory.add_ai_message(detailed_response)
    
    return detailed_response

# Interactive Chat Loop
def chat():
    print("Kk_ChatGPT: Hello! Ask me anything, and I'll provide a detailed answer. Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Kk_ChatGPT: Goodbye! Feel free to return if you have more questions.")
            break
        try:
            response = generate_detailed_response(user_input)
            print(f"Kk_ChatGPT: {response}\n")
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            print("Kk_ChatGPT: Sorry, I encountered an issue. Please try again!")

# Run the PDF-based RAG pipeline
def run_pdf_rag_pipeline(pdf_path):
    documents = process_pdf_to_documents(pdf_path)
    if not documents:
        logger.info("No valid text could be extracted from the PDF.")
        return

    chunks = chunk_documents(documents)
    for batch in batch_data(chunks, 40000):
        vectorstore.add_documents(batch)
    
    chat()

# Batch data for vector store
def batch_data(data, batch_size):
    """Yield successive batches from the data."""
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

if __name__ == "__main__":
    pdf_path = ""  # Replace with your PDF file path
    run_pdf_rag_pipeline(pdf_path)

