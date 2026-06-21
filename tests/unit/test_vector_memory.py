from src.workspace_agent.core import vector_memory

# ==========================================
# Component: save_memory
# ==========================================


def test_save_memory_success_standard(mocker):
    """Green Path: Verifies that memories are saved with the correct metadata."""
    mock_collection = mocker.Mock()

    vector_memory.save_memory(
        "test discovery", "thread_123", mock_collection, {"type": "operational_insight"}
    )

    mock_collection.add.assert_called_once()
    _, kwargs = mock_collection.add.call_args

    # Verify the text and metadata were correctly passed to ChromaDB
    assert kwargs["documents"] == ["test discovery"]
    assert kwargs["metadatas"][0]["thread_id"] == "thread_123"
    assert kwargs["metadatas"][0]["type"] == "operational_insight"


def test_save_memory_fallback_exception(mocker, capsys):
    """Edge Path: Verifies that database errors are caught and printed without crashing."""
    mock_collection = mocker.Mock()
    mock_collection.add.side_effect = Exception("DB Connection Lost")

    vector_memory.save_memory("test discovery", "thread_123", mock_collection)

    # Verify the error was printed to stdout
    captured = capsys.readouterr()
    assert "Failed to save memory to ChromaDB: DB Connection Lost" in captured.out


# ==========================================
# Component: semantic_search
# ==========================================


def test_semantic_search_success_global(mocker):
    """
    Green Path: Verifies that omitting the thread_id searches globally across all threads for
    insights.
    """
    mock_collection = mocker.Mock()
    mock_collection.query.return_value = {"documents": [["Global Result"]]}

    results = vector_memory.semantic_search("python rules", mock_collection)

    _, kwargs = mock_collection.query.call_args
    # Verify the fallback where clause correctly targets the operational_insight type
    assert kwargs["where"] == {"type": "operational_insight"}
    assert results == ["Global Result"]


def test_semantic_search_success_standard(mocker):
    """Green Path: Verifies that search queries are correctly formulated and returned."""
    mock_collection = mocker.Mock()
    mock_collection.query.return_value = {
        "documents": [["Result 1: Use type hints", "Result 2: Use black"]]
    }

    results = vector_memory.semantic_search("python rules", mock_collection, "thread_123", limit=2)

    mock_collection.query.assert_called_once()
    _, kwargs = mock_collection.query.call_args

    # Verify the query parameters and the new $and composite filter
    assert kwargs["query_texts"] == ["python rules"]
    assert kwargs["n_results"] == 2
    assert kwargs["where"] == {
        "$and": [{"thread_id": "thread_123"}, {"type": "operational_insight"}]
    }

    # Verify it unpacked the nested list correctly
    assert results == ["Result 1: Use type hints", "Result 2: Use black"]


def test_semantic_search_fallback_empty_query(mocker):
    """
    Edge Path: Empty or whitespace-only queries should short-circuit and return [] without
    calling the DB.
    """
    mock_collection = mocker.Mock()

    assert vector_memory.semantic_search("", mock_collection) == []
    assert vector_memory.semantic_search("   ", mock_collection) == []
    assert vector_memory.semantic_search(None, mock_collection) == []

    # Verify the ChromaDB client was never actually invoked
    mock_collection.query.assert_not_called()


def test_semantic_search_fallback_exception(mocker, capsys):
    """Edge Path: Verifies that search errors are caught and return empty lists."""
    mock_collection = mocker.Mock()
    mock_collection.query.side_effect = Exception("Search Timeout")

    results = vector_memory.semantic_search("python rules", mock_collection)

    captured = capsys.readouterr()
    assert "Vector search failed: Search Timeout" in captured.out
    assert results == []


def test_semantic_search_fallback_no_results(mocker):
    """Edge Path: Verifies that an empty database response returns an empty list safely."""
    mock_collection = mocker.Mock()
    mock_collection.query.return_value = {}

    results = vector_memory.semantic_search("python rules", mock_collection)

    assert results == []
