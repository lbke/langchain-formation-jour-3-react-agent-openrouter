# Import this file in your agent and check the logs when running the app
# https://docs.trychroma.com/docs/overview/getting-started
import chromadb
# Will create an ephemeral in-memory Chroma instance
# => it disappears when we stop the agent
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="test_collection")
collection.add(
    documents=[
        "This is a document about pineapple",
        "This is a document about oranges"
    ],
    ids=["id1", "id2"]
)
results = collection.query(
    # Chroma will embed this for you
    query_texts=["This is a query document about hawaii"],
    n_results=2  # how many results to return
)
print(results)
