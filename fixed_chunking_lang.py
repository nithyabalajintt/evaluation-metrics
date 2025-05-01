import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import AzureChatOpenAI
from ragas import EvaluationDataset, evaluate
from ragas.metrics import Faithfulness, FactualCorrectness, ContextRelevance, ContextUtilization, ContextRecall
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

### Environment Variables Validation
def get_env_var(var_name):
    """
    Retrieve an environment variable and throw an informative error if missing.
    """
    value = os.getenv(var_name)
    if not value:
        raise RuntimeError(f"Environment variable '{var_name}' is missing or not set.")
    return value

# Azure OpenAI Configuration
deployment_name = get_env_var("AZURE_OPENAI_DEPLOYMENT")
model_name = get_env_var("AZURE_OPENAI_MODEL_NAME")
api_version = get_env_var("API_VERSION_GA")
api_key = get_env_var("AZURE_OPENAI_API_KEY")
azure_endpoint = get_env_var("AZURE_OPENAI_ENDPOINT")

# Azure OpenAI Embedding Configuration
embedding_deployment_name = get_env_var("ADA_AZURE_OPENAI_DEPLOYMENT")
embedding_model_name = get_env_var("ADA_AZURE_OPENAI_MODEL_NAME")

### Paths
doc_path = "data/test_data.pdf"        # Path to the test PDF document
ground_truth_path = "data/ground_truth.json"  # Path to the ground truth JSON file
chunk_size = 1000                     # Chunk size for document splitting
chunk_overlap = 200                   # Overlap for chunks


# Step 1: Initialize Azure OpenAI Components
# LLM for answering questions
azure_llm = AzureChatOpenAI(
    deployment_name=deployment_name,
    openai_api_key=api_key,
    azure_openai_api_base=azure_endpoint,
    azure_openai_api_version=api_version
)

# Embedding model for vector database storage
embedder = AzureOpenAIEmbeddings(
    azure_deployment=embedding_deployment_name,
    model=embedding_model_name,
    openai_api_key=api_key,
    chunk_size=1
)

# Step 2: PDF Loading and Chunking
# Load the PDF document
loader = PyPDFLoader(doc_path)
raw_documents = loader.load()

# Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)
documents = text_splitter.split_documents(raw_documents)

# Step 3: Vector Database Initialization (Chroma)
# Initialize a Chroma vector database for document indexing
persist_directory = "./chroma_db"
vector_db = Chroma.from_documents(
    documents=documents,
    embedding=embedder,
    persist_directory=persist_directory
)

# Step 4: Retrieval-based Question Answering Chain
retriever = vector_db.as_retriever()  # Create a retriever from the vector database
retrieval_qa = RetrievalQA.from_chain_type(
    llm=azure_llm,
    retriever=retriever,
    chain_type="stuff"  # Combine relevant documents into a single prompt
)

# Step 5: Create Dataset for Evaluation
def create_eval_ds(retrieval_qa, ground_truth_path):
    """
    Create evaluation dataset with questions, answers, and predictions.
    """
    with open(ground_truth_path, "r") as file:
        ground_truth_data = json.load(file)

    eval_data = []
    for item in ground_truth_data:
        question = item["question"]
        answer = item["answer"]

        # Use the QA chain to generate the model's response
        generated_answer = retrieval_qa.run(question)

        # Add data to evaluation dataset
        eval_data.append({
            "question": question,
            "answer": answer,
            "generated": generated_answer,
            "context": None  # Optional, add context if needed for metrics
        })

    return eval_data

# Prepare evaluation dataset
eval_data = create_eval_ds(retrieval_qa, ground_truth_path)

# Step 6: Perform Evaluation Using Ragas
evaluation_dataset = EvaluationDataset.from_list(eval_data)
results = evaluate(
    dataset=evaluation_dataset,
    metrics=[
        Faithfulness(llm=azure_llm),
        FactualCorrectness(llm=azure_llm),
        ContextRelevance(llm=azure_llm),
        ContextUtilization(llm=azure_llm),
        ContextRecall(llm=azure_llm)
    ]
)

# Step 7: Print Evaluation Results
for score in results.scores:
    print(f"{score.metric}: {score.value:.2f}")

# Step 8: Clean Up Vector Database (Optional)
# If you want to delete the database after processing
if os.path.exists(persist_directory):
    import shutil
    shutil.rmtree(persist_directory)