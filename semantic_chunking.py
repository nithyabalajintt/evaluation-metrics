import os

from agno.agent import Agent
from agno.document.chunking.semantic import SemanticChunking
from agno.embedder.ollama import OllamaEmbedder
from agno.models.anthropic import Claude
from qdrant_client import qdrant_client
from agno.knowledge.pdf import PDFKnowledgeBase
from agno.vectordb.qdrant import Qdrant
from dotenv import load_dotenv, find_dotenv

from ragas.llms import LlamaIndexLLMWrapper
from ragas import EvaluationDataset, evaluate
from ragas.metrics import Faithfulness, FactualCorrectness, ContextRelevance, ContextUtilization, ContextRecall

from llama_index.llms.openai import OpenAI

from create_eval_dataset import create_eval_ds

eval_llm = OpenAI(model='gpt-4o')

load_dotenv(find_dotenv())

doc_path = "data/test_data.pdf"
ground_truth_path = "data/ground_truth.json"

# initialize the LLM (default to openai)
claude = Claude(id="claude-3-5-sonnet-20241022")

# initialize the qdrant client.
q_client = qdrant_client.QdrantClient(url=os.environ.get('qdrant_url'), api_key=os.environ.get('api_key'))

# create the qdrant vector store instance
vector_db = Qdrant(
    collection=os.environ.get('collection_name'),
    url=os.environ.get('qdrant_url'),
    api_key=os.environ.get('api_key'),
    embedder=OllamaEmbedder(id="nomic-embed-text:latest", dimensions=768)
)

# configure the knowledge base
knowledge_base = PDFKnowledgeBase(vector_db=vector_db,
                                  path=doc_path,
                                  chunking_strategy=SemanticChunking(
                                      chunk_size=chunk_size,
                                      similarity_threshold=0.6)
                                  )

if not q_client.collection_exists(collection_name=os.environ.get('collection_name')):
    knowledge_base.load(recreate=False)

# initialize agent
agent = Agent(knowledge=knowledge_base,search_knowledge=True, model=claude)

# create the dataset for evaluation
eval_dataset = create_eval_ds(agent=agent, ground_truth_path=ground_truth_path)

evaluation_dataset = EvaluationDataset.from_list(eval_dataset)
evaluator_llm = LlamaIndexLLMWrapper(llm=eval_llm)
result = evaluate(dataset=evaluation_dataset, metrics=[Faithfulness(), ContextRelevance(),
                                                       ContextUtilization(), ContextRecall(),
                                                       FactualCorrectness()])

for score in result.scores:
    print(score)

# destroy the collection
if q_client.collection_exists(collection_name=os.environ.get('collection_name')):
    q_client.delete_collection(collection_name=os.environ.get('collection_name'))
