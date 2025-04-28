# import unittest
# from unittest.mock import MagicMock, patch
# from typing import TYPE_CHECKING, Callable, Any

# from langchain_core.embeddings import Embeddings

# from langchain_opensearch.vectorstores import OpenSearchVectorStore

# try:
#     from opensearchpy.helpers import bulk as opensearch_bulk_helper
# except ImportError:
#     opensearch_bulk_helper = None


# class TestOpenSearchVectorStore(unittest.TestCase):
#     def test_import(self):None
#         """Test that the OpenSearchVectorStore class can be imported."""
#         self.assertTrue(hasattr(OpenSearchVectorStore, "add_texts"))

#     @patch("opensearchpy.helpers.bulk")
#     def test_add_texts_mock(self, mock_bulk_helper):None
#         """Test add_texts, mocking the opensearchpy bulk helper."""
#         mock_client = MagicMock()
#         mock_embeddings = MagicMock(spec=Embeddings)
#         mock_embeddings.embed_documents.return_value = [[0.1, 0.2], [0.3, 0.4]]

#         vector_store = OpenSearchVectorStore(
#             client=mock_client,
#             index_name="test_index",
#             embedding_function=mock_embeddings,
#         )

#         texts = ["Hello", "World"]
#         vector_store.add_texts(texts)

#         mock_embeddings.embed_documents.assert_called_once_with(texts)
#         mock_bulk_helper.assert_called_once()

#         args, kwargs = mock_bulk_helper.call_args
#         self.assertIs(args[0], mock_client)
#         self.assertEqual(len(args[1]), len(texts))
#         self.assertEqual(args[1][0]["_op_type"], "index")
#         self.assertIn("_source", args[1][0])
#         self.assertEqual(args[1][0]["_source"]["text"], texts[0])

#     def test_similarity_search_mock(self):None
#         """Test similarity_search with mocked client."""
#         mock_client = MagicMock()
#         mock_client.search.return_value = {
#             "hits": {
#                 "hits": [
#                     {"_id": "test_id", "_source": {"text": "Hello"}, "_score": 0.9}
#                 ]
#             }
#         }
#         mock_embeddings = MagicMock(spec=Embeddings)
#         mock_embeddings.embed_query.return_value = [0.1, 0.2]

#         vector_store = OpenSearchVectorStore(
#             client=mock_client,
#             index_name="test_index",
#             embedding_function=mock_embeddings,
#         )

#         results = vector_store.similarity_search("test query", k=1)
#         mock_embeddings.embed_query.assert_called_once_with("test query")
#         mock_client.search.assert_called_once()
#         self.assertEqual(len(results), 1)
#         self.assertEqual(results[0].page_content, "Hello")


# if __name__ == "__main__":
#     unittest.main()


from __future__ import annotations

import unittest
from collections.abc import Callable
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from langchain_opensearch.vectorstores import OpenSearchVectorStore

# Define a type alias for the bulk helper function
BulkHelperType = Callable[..., Any]

if TYPE_CHECKING:
    # This block is only evaluated by type checkers (e.g., MyPy)
    try:
        from opensearchpy.helpers import bulk as opensearch_bulk_helper_real

        opensearch_bulk_helper: BulkHelperType | None = opensearch_bulk_helper_real
    except ImportError:
        # If import fails for type checker, define as None
        opensearch_bulk_helper = None
else:
    # This block is executed at runtime
    try:
        from opensearchpy.helpers import bulk as opensearch_bulk_helper_real

        opensearch_bulk_helper: BulkHelperType | None = opensearch_bulk_helper_real
    except ImportError:
        # Assign None at runtime if import fails
        opensearch_bulk_helper = None


class TestOpenSearchVectorStore(unittest.TestCase):
    def test_import(self) -> None:
        """Test that the OpenSearchVectorStore class can be imported."""
        self.assertTrue(hasattr(OpenSearchVectorStore, "add_texts"))

    # @patch("langchain_opensearch.vectorstores._bulk_ingest_embeddings")
    @patch("opensearchpy.helpers.bulk")
    def test_add_texts_mock(self, mock_bulk_helper: MagicMock) -> None:
        """Test add_texts method with a mocked OpenSearch bulk helper."""
        if opensearch_bulk_helper is None:
            self.skipTest("opensearchpy.helpers.bulk not available")

        # Create mock objects
        mock_client = MagicMock()
        mock_embeddings = MagicMock(spec=Embeddings)
        mock_embeddings.embed_documents.return_value = [[0.1, 0.2], [0.3, 0.4]]

        # Initialize the vector store
        vector_store = OpenSearchVectorStore(
            client=mock_client,
            index_name="test_index",
            embedding_function=mock_embeddings,
        )

        # Test adding texts
        texts = ["Hello", "World"]
        vector_store.add_texts(texts)

        # Assertions
        mock_embeddings.embed_documents.assert_called_once_with(texts)
        mock_bulk_helper.assert_called_once()

        # Check arguments passed to the mock bulk helper
        args, kwargs = mock_bulk_helper.call_args
        self.assertIs(args[0], mock_client)  # First arg should be the client
        actions = args[1]  # Second arg should be the list of actions
        self.assertEqual(len(actions), len(texts))
        self.assertEqual(actions[0]["_op_type"], "index")
        self.assertIn("_source", actions[0])
        self.assertEqual(actions[0]["_source"]["text"], texts[0])
        self.assertIn("vector_field", actions[0]["_source"])
        self.assertEqual(len(actions[0]["_source"]["vector_field"]), 2)

    def test_similarity_search_mock(self) -> None:
        """Test similarity_search method with a mocked client."""
        # Create mock objects
        mock_client = MagicMock()
        mock_client.search.return_value = {
            "hits": {
                "total": {"value": 1, "relation": "eq"},
                "max_score": 0.9,
                "hits": [
                    {
                        "_index": "test_index",
                        "_id": "test_id",
                        "_score": 0.9,
                        "_source": {
                            "text": "Hello",
                            "vector_field": [0.5, 0.6],
                            "metadata": {"source": "doc1"},
                        },
                    }
                ],
            }
        }
        mock_embeddings = MagicMock(spec=Embeddings)
        mock_embeddings.embed_query.return_value = [0.1, 0.2]

        # Initialize the vector store
        vector_store = OpenSearchVectorStore(
            client=mock_client,
            index_name="test_index",
            embedding_function=mock_embeddings,
        )

        # Test similarity search
        query = "test query"
        results = vector_store.similarity_search(query, k=1)

        # Assertions
        mock_embeddings.embed_query.assert_called_once_with(query)
        mock_client.search.assert_called_once()

        # Check the search query passed to the mock client
        args, kwargs = mock_client.search.call_args
        self.assertEqual(kwargs.get("index"), "test_index")

        # Check the results
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], Document)
        self.assertEqual(results[0].page_content, "Hello")
        self.assertEqual(results[0].metadata, {"source": "doc1"})


if __name__ == "__main__":
    unittest.main()
