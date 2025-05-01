import os
from agno.agent import Agent
from agno.document.chunking.fixed import FixedSizeChunking
from agno.knowledge.pdf import PDFKnowledgeBase
import Chroma
from dotenv import load_dotenv, find_dotenv
 
from ragas import EvaluationDataset, evaluate
from ragas.metrics import Faithfulness, FactualCorrectness, ContextRelevance, ContextUtilization, ContextRecall
 
# Azure OpenAI imports
from openai import AzureOpenAI
from llama_index.llms.azure_openai import AzureOpenAI as AzureOpenAILlamaIndex
from langchain.embeddings import AzureOpenAIEmbeddings
 
load_dotenv(find_dotenv())
 
# Configuration
doc_path = "data/test_data.pdf"
ground_truth_path = "data/ground_truth.json" 
chunk_size = 1000
chunk_overlap = 200
 
# Azure OpenAI Configuration
model_name = os.environ.get("AZURE_OPENAI_MODEL_NAME")
endpoint_url = os.environ.get("AZURE_OPENAI_ENDPOINT")
api_key = os.environ.get("AZURE_OPENAI_API_KEY")
deployment_name = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
version_number = os.environ.get("API_VERSION_GA")
 
# Azure OpenAI Embeddings Configuration
embedding_model_name = os.environ.get("ADA_AZURE_OPENAI_MODEL_NAME")
embedding_endpoint_url = os.environ.get("ADA_AZURE_OPENAI_ENDPOINT")
embedding_api_key = os.environ.get("ADA_AZURE_OPENAI_API_KEY")
embedding_deployment_name = os.environ.get("ADA_AZURE_OPENAI_MODEL_NAME")
 
# Initialize Azure OpenAI LLM
eval_llm = AzureOpenAILlamaIndex(
    engine=deployment_name,
    model=model_name,
    api_key=api_key,
    api_version=version_number,
    azure_endpoint=endpoint_url
)
 
# Initialize Azure OpenAI Embedder
embedder = AzureOpenAIEmbeddings(
    model=embedding_model_name,
    api_key=embedding_api_key,
    azure_deployment=embedding_deployment_name,
    azure_endpoint=embedding_endpoint_url,
    api_version=version_number
)
 
# Initialize ChromaDB with Azure OpenAI embeddings
vector_db = ChromaDB(
    collection_name=os.environ.get('collection_name'),
    persist_directory="./chroma_db",
    embedder=embedder  # Using Azure OpenAI embeddings instead of Ollama
)
 
# Configure the knowledge base
knowledge_base = PDFKnowledgeBase(
    vector_db=vector_db,
    path=doc_path,
    chunking_strategy=FixedSizeChunking(
        chunk_size=chunk_size,
        overlap=chunk_overlap)
)
 
# Check if collection exists and load if not
if not vector_db.collection_exists():
    knowledge_base.load(recreate=False)
 
# Initialize agent
agent = Agent(knowledge=knowledge_base, search_knowledge=True, model=eval_llm)
 

# create the dataset for evaluation
eval_dataset = create_eval_ds(agent=agent, ground_truth_path=ground_truth_path)

# trigger evals
evaluation_dataset = EvaluationDataset.from_list(eval_dataset)
result = evaluate(
    dataset=evaluation_dataset,
    metrics=[
        Faithfulness(llm=eval_llm),
        ContextRelevance(llm=eval_llm),
        ContextUtilization(llm=eval_llm),
        ContextRecall(llm=eval_llm),
        FactualCorrectness(llm=eval_llm)
    ]
)

# Print evaluation results
for score in result.scores:
    print(score)
 
# Clean up ChromaDB collection if needed
if vector_db.collection_exists():
    vector_db.delete_collection()
 
# destroy the collection (for chroma)
if q_client.collection_exists(collection_name=os.environ.get('collection_name')):
    q_client.delete_collection(collection_name=os.environ.get('collection_name'))
