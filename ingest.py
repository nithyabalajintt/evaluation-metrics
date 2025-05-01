from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.documents import Document
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
import shutil
from langchain.vectorstores.chroma import Chroma 
from langchain_openai import AzureOpenAIEmbeddings
from dotenv import load_dotenv


# Paths
DATA_PATH = "data\pdf_files"
CHROMA_PATH = "chroma_finance_docs"

# Load environment variables from a .env file
load_dotenv()

embedding_model_name = os.environ.get("ADA_AZURE_OPENAI_MODEL_NAME")
embedding_endpoint_url = os.environ.get("ADA_AZURE_OPENAI_ENDPOINT")
embedding_api_key = os.environ.get("ADA_AZURE_OPENAI_API_KEY")
embedding_deployment_name = os.environ.get("ADA_AZURE_OPENAI_DEPLOYMENT")
# version_number = os.environ.get("API_VERSION_GA")

embedder = AzureOpenAIEmbeddings(model=embedding_model_name, api_key = embedding_api_key, azure_endpoint=embedding_endpoint_url)


def load_documents():
  """
  Load PDF documents from the specified directory using PyPDFDirectoryLoader.
  Returns:
  List of Document objects: Loaded PDF documents represented as Langchain
                                                          Document objects.
  """

  document_loader = PyPDFDirectoryLoader(DATA_PATH, glob = "**/[!.]*.pdf") 

  return document_loader.load() 

documents = load_documents() 
print(documents)

def split_text(documents: list[Document]):
  """
  Split the text content of the given list of Document objects into smaller chunks.
  Args:
    documents (list[Document]): List of Document objects containing text content to split.
  Returns:
    list[Document]: List of Document objects representing the split text chunks.
  """
  # Initialize text splitter with specified parameters
  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200, 
    length_function=len, 
    add_start_index=True, # Flag to add start index to each chunk
  )

  chunks = text_splitter.split_documents(documents)
  print(f"Split {len(documents)} documents into {len(chunks)} chunks.")


  document = chunks[0]
  print(document.page_content)
  print(document.metadata)

  return chunks 


def save_to_chroma(chunks: list[Document]):
  """
  Save the given list of Document objects to a Chroma database.
  Args:
  chunks (list[Document]): List of Document objects representing text chunks to save.
  Returns:
  None
  """

  # Clear out the existing database directory if it exists
  if os.path.exists(CHROMA_PATH):
    shutil.rmtree(CHROMA_PATH)

  # Create a new Chroma database from the documents using OpenAI embeddings
  db = Chroma.from_documents(
    chunks,
    embedder,
    persist_directory=CHROMA_PATH
  )

  # Persist the database to disk
  db.persist()
  print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


def generate_data_store():
  """
  Function to generate vector database in chroma from documents.
  """
  documents = load_documents() # Load documents from a source
  chunks = split_text(documents) # Split documents into manageable chunks
  save_to_chroma(chunks) # Save the processed data to a data store


# Generate the data store
generate_data_store()

