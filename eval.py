import os
from dotenv import load_dotenv, find_dotenv

# LangChain Imports
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import AzureChatOpenAI

# Evaluation-specific imports
from ragas import EvaluationDataset, evaluate
from ragas.metrics import Faithfulness, FactualCorrectness, ContextRelevance, ContextUtilization, ContextRecall

import json
from datasets import Dataset

# Load environment variables
load_dotenv(find_dotenv())

# Paths
doc_path = "data/test_data.pdf"
ground_truth_path = "data/ground_truth.json"
chunk_size = 1000
chunk_overlap = 200

# Azure OpenAI Configurations
model_name = os.getenv("AZURE_OPENAI_MODEL_NAME")
endpoint_url = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT")
version_number = os.getenv("API_VERSION_GA")
embedding_model_name = os.getenv("ADA_AZURE_OPENAI_MODEL_NAME")
embedding_endpoint_url = os.getenv("ADA_AZURE_OPENAI_ENDPOINT")

# Initialize Azure OpenAI LLM Chat Model
azure_llm = AzureChatOpenAI(
    openai_api_key=api_key,
    deployment_name=deployment_name,
    azure_openai_api_version=version_number,
    azure_openai_api_base=endpoint_url
)

# Initialize Azure OpenAI Embeddings
embedding_model = OpenAIEmbeddings(
    model=model_name,
    chunk_size=1,
    openai_api_key=api_key
)

# Initialize ChromaDB Vector Database
chroma_db = Chroma(
    embedding_function=embedding_model,
    persist_directory="./chroma_db"  # Specify path for persistence
)

# Load documents into ChromaDB (if necessary)
if not os.path.exists('./chroma_db'):  # Skip if already present
    from langchain.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    
    # Load and chunk documents
    loader = PyPDFLoader(doc_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = splitter.split_documents(documents)
    
    # Persist to vector database
    chroma_db.add_documents(texts)

# Build RetrievalQA Chain
retriever = chroma_db.as_retriever()
qa_chain = RetrievalQA.from_chain_type(
    llm=azure_llm,
    chain_type="stuff",  # Configurable
    retriever=retriever
)

# Helper function for evaluation
def create_eval_ds(agent, chain, ground_truth_path):
    """
    Prepare the evaluation dataset based on the ground truth format.
    Returns a list of dictionaries with 'context', 'question', 'generated' (output), and 'answer' (ground truth).
    """
    with open(ground_truth_path, 'r') as f:
        ground_truth_data = json.load(f)
    
    evaluation_data = []
    for item in ground_truth_data:
        question = item["question"]
        ground_truth_answer = item["answer"]
        
        # Agent performs the question-answering
        generated_answer = chain.run(question)
        
        evaluation_data.append({
            "context": "",  # Optional, additional context for Faithfulness Relevance can go here
            "question": question,
            "generated": generated_answer,
            "answer": ground_truth_answer
        })
    return evaluation_data

# Create Evaluation Dataset
eval_data = create_eval_ds(agent=None, chain=qa_chain, ground_truth_path=ground_truth_path)
eval_dataset = Dataset.from_dict(eval_data)

# Run Evaluation with Ragas
evaluation_dataset = EvaluationDataset.from_list(eval_dataset)
result = evaluate(
    dataset=evaluation_dataset,
    metrics=[
        Faithfulness(llm=azure_llm),
        ContextRelevance(llm=azure_llm),
        ContextUtilization(llm=azure_llm),
        ContextRecall(llm=azure_llm),
        FactualCorrectness(llm=azure_llm)
    ]
)

# Print Results
for score in result.scores:
    print(f"{score.metric}: {score.value:.2f}")