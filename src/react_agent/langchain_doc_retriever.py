# Initialize an ephemercal in-memory Chroma vector store
# It will disappear when we stop the agent
# We import it in graph.py to feed the agent with documents
import chromadb
chroma_client = chromadb.Client()
documents = [
    {"id": "1", "content": "LangChain invokes LLMs"},
    {"id": "2", "content": "LangGraph runs agents"},
]
collection = chroma_client.create_collection(name="langchain_documentation")
collection.upsert(
    # Génération d'identifiants uniques pour chaque document
    ids=[d["id"] for d in documents],
    documents=[d["content"] for d in documents],
    # Metadata can't be nested, they have to be simple key-value pairs
    metadatas=[{"foo": "bar"}, {"foo": "zed"}]
)

# Test with
# python ./src/langchain_doc_retriever.py
if __name__ == "__main__":
    collection = chroma_client.get_collection(name="langchain_documentation")
    queries = ["What is LangChain?", "What is LangGraph?"]
    for query in queries:
        results = collection.query(
            query_texts=[query],
            n_results=1
        )
        print(f"Query: {query}")
        for doc in results['documents'][0]:
            print(f" - Retrieved document: {doc}")
