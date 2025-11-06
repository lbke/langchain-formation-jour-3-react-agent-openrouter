# TODO: adapted to Chroma but not yet tested
# The Chroma vector store config might be different
# Inspired from LangSmith academy utils
# https://github.com/langchain-ai/intro-to-langsmith/blob/main/notebooks/module_1/utils.py
# https://github.com/langchain-ai/intro-to-langsmith/blob/a61b50fc7035af9ac5958a18a9131cee84c1373b/notebooks/module_1/utils.py
import os
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.sitemap import SitemapLoader
# from langchain_community.vectorstores import SKLearnVectorStore
from langchain_chroma import Chroma
# from langchain_openai import OpenAIEmbeddings
from chromadb.utils import embedding_functions

# https://docs.trychroma.com/production/deployment
# import chromadb
# Example setup of the client to connect to your chroma server
# client = chromadb.HttpClient(host='localhost', port=8000)

# Chroma provides a default embedding function
# But it is not used in LangGraph => we write a quick dummy wrapper to reuse this function
# @see https://docs.trychroma.com/docs/embeddings/embedding-functions#custom-embedding-functions
# TODO: ideally, the Chroma vector store should use this function as a default
# TODO: we could add explicit errors for the async functions


class ChromaEmbeddings:
    def __init__(self):
        self.embd = embedding_functions.DefaultEmbeddingFunction()

    def embed_query(self, query):
        return self.embd([query])[0]

    def embed_documents(self, docs):
        return self.embd(docs)


def get_langsmith_doc_retriever():
    """
    Ingest LangSmith doc
    Can be adapted for other sites with a sitemap
    Will cache the Chroma database in a tmp folder

    BONUS: improve to split the data ingestion step and the vector store creation step
    """
    persist_directory = os.path.join(tempfile.gettempdir(), "langsmith_docs")
    # embd = OpenAIEmbeddings()
    embd = ChromaEmbeddings()

    # If vector store exists, then load it
    if os.path.exists(persist_directory):
        print(
            f"Found LangSmith documentation for Chroma in {persist_directory}, will load it")
        # vectorstore = SKLearnVectorStore(
        vectorstore = Chroma(
            embedding_function=embd,
            persist_directory=persist_directory,
            # To use an HTTP Chroma client rather than a local directory
            # client=client
        )
        return vectorstore.as_retriever(lambda_mult=0)

    print(
        f"Ingesting LangSmith documentation, persisting in {persist_directory}")

    # Otherwise, index LangSmith documents and create new vector store
    ls_docs_sitemap_loader = SitemapLoader(
        web_path="https://docs.smith.langchain.com/sitemap.xml", continue_on_failure=True)
    ls_docs = ls_docs_sitemap_loader.load()

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(ls_docs)

    # vectorstore = SKLearnVectorStore.from_documents(
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        # (not embedding_function but embedding)
        embedding=embd,
        persist_directory=persist_directory,
    )
    print(f"Done persisting LangSmith documentation in {persist_directory}")
    return vectorstore.as_retriever(lambda_mult=0)
