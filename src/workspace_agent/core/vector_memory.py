import uuid


def save_memory(text: str, thread_id: str, collection, metadata: dict = None):
    """Saves a string memory to the injected ChromaDB vector store."""
    if not collection:
        return

    try:
        meta = metadata or {}
        meta["thread_id"] = thread_id

        collection.add(documents=[text], metadatas=[meta], ids=[str(uuid.uuid4())])
    except Exception as e:
        print(f"Failed to save memory to ChromaDB: {e}")


def semantic_search(query: str, collection, thread_id: str = None, limit: int = 3) -> list[str]:
    """
    Searches the ChromaDB vector store for relevant operational insights.
    If thread_id is provided, filters by thread AND type. Otherwise, searches globally by type.
    """
    if not query or not query.strip() or not collection:
        return []

    if thread_id:
        where_clause = {"$and": [{"thread_id": thread_id}, {"type": "operational_insight"}]}
    else:
        where_clause = {"type": "operational_insight"}

    try:
        results = collection.query(query_texts=[query], n_results=limit, where=where_clause)

        if results and results.get("documents") and results["documents"][0]:
            return results["documents"][0]
        return []
    except Exception as e:
        print(f"Vector search failed: {e}")
        return []
