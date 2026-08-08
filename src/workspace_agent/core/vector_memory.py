import uuid
from typing import Any

__all__ = ["save_memory", "semantic_search"]

# ==========================================
# Vector Store Memory Operations
# ==========================================


def save_memory(
    text: str,
    thread_id: str,
    collection: Any,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Saves a string memory to the injected ChromaDB vector store.

    State Transitions:
    - Mutates the injected vector collection by appending a new document with a generated UUID,
      the provided text, and metadata enriched with the thread_id.
    - Performs a no-op if the collection object is None or evaluates to False.

    Exceptions:
    - Catches and logs any underlying database or network Exception raised during the ChromaDB
      add operation; does not re-raise exceptions to the caller.
    """
    if not collection:
        return

    try:
        meta = metadata or {}
        meta["thread_id"] = thread_id

        collection.add(documents=[text], metadatas=[meta], ids=[str(uuid.uuid4())])
    except Exception as e:
        print(f"Failed to save memory to ChromaDB: {e}")


def semantic_search(
    query: str,
    collection: Any,
    thread_id: str | None = None,
    limit: int = 3,
) -> list[str]:
    """Searches the ChromaDB vector store for relevant operational insights.

    If thread_id is provided, filters by thread AND type. Otherwise, searches globally by type.

    State Transitions:
    - Read-only query execution against the injected ChromaDB collection.
    - Returns a list of matching document text strings ordered by semantic relevance
      up to `limit`.
    - Returns an empty list if the query is empty/whitespace, collection is missing,
      or no matches exist.

    Exceptions:
    - Catches and logs any Exception raised during the ChromaDB query operation;
      returns an empty list rather than propagating the exception to the caller.
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
