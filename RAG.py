import os
from langchain.vectorstores.chroma import Chroma 
from langchain_openai import AzureOpenAIEmbeddings
from dotenv import load_dotenv
from openai import AzureOpenAI
import streamlit as st

CHROMA_PATH = "chroma_finance_docs"
load_dotenv()

#GPT-4o
model_name = os.environ.get("AZURE_OPENAI_MODEL_NAME")
endpoint_url = os.environ.get("AZURE_OPENAI_ENDPOINT")
api_key = os.environ.get("AZURE_OPENAI_API_KEY")
deployment_name = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
version_number = os.environ.get("API_VERSION_GA")

embedding_model_name = os.environ.get("ADA_AZURE_OPENAI_MODEL_NAME")
embedding_endpoint_url = os.environ.get("ADA_AZURE_OPENAI_ENDPOINT")
embedding_api_key = os.environ.get("ADA_AZURE_OPENAI_API_KEY")
embedding_deployment_name = os.environ.get("ADA_AZURE_OPENAI_DEPLOYMENT")
# version_number = os.environ.get("API_VERSION_GA")

embedder = AzureOpenAIEmbeddings(model=embedding_model_name, api_key = embedding_api_key, azure_deployment= embedding_deployment_name, azure_endpoint=embedding_endpoint_url, api_version="2024-10-21")

def query_rag(query_text):
    """
    Query a Retrieval-Augmented Generation (RAG) system using Chroma database and OpenAI.
    Args:
    - query_text (str): The text to query the RAG system with.
    Returns:
    - formatted_response (str): Formatted response including the generated text and sources.
    - response_text (str): The generated response text.
    """

    # Prepare the database
    print(CHROMA_PATH)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedder)

    # Retrieving the context from the DB using similarity search
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    # print(results)

    # Check if there are any matching results or if the relevance score is too low
    if len(results) == 0 or results[0][1] < 0.7:
        print(f"Unable to find matching results.")

    # Combine context from matching documents
    context_text = "\n\n - -\n\n".join([doc.page_content for doc, _score in results])

    # Azure OpenAI
    client = AzureOpenAI(
            azure_endpoint=endpoint_url,
            azure_deployment=deployment_name,
            api_key=api_key,
            api_version=version_number,
        )

    rag_chat_completion = client.chat.completions.create(

        messages=[
            {
                "role": "system",
                "content": f"""You are a Retrieval Augemnetation Chatbot that will only reply to queries asked by the user based on the context provided to you from the vector database. If that answer to the query does not lie in the context provided to you, you will simply reply with \"I am sorry, that question lies beyond the context provided to me, I cannot answer it.\""""
            },
            {
                "role": "user",
                "content": f"""VECTOR DATABASE CONTEXT:
                {context_text}\n\nUSER QUERY:{query_text}""",
            }
        ],

        model=model_name,
        # model_kwargs= {"stream_options": {"include_usage": True}}
        # temperature=0.5,
        )

    # Generate response text based on the prompt
    response_text = rag_chat_completion.choices[0].message.content
    response_usage = rag_chat_completion.usage

    # Get sources of the matching documents
    sources = [doc.metadata.get("source", None) for doc, _score in results]
    print(query_text)

    # Format and return response including generated text and sources
    formatted_response = f"Response: {response_text}\nSources: {sources}\nUsage: {response_usage}"
    return formatted_response, response_text


query_text = "What are the benefits of perksplus"
formatted_response, response_text = query_rag(query_text)
print(formatted_response)

