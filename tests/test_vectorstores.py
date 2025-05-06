from __future__ import annotations
import unittest
from typing import TYPE_CHECKING, Any, Optional
from unittest.mock import MagicMock, patch, call
import numpy as np
import pytest # Import pytest

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

try:
    from opensearchpy import OpenSearch
    from opensearchpy import AsyncOpenSearch
    from opensearchpy.helpers import bulk as opensearch_bulk_helper_real
    # Async bulk helper is not used in this minimal set, but keep for robustness
    from opensearchpy.helpers import async_bulk as opensearch_async_bulk_helper_real
except ImportError:
    # Provide dummy classes/values if opensearch-py is not installed
    OpenSearch = MagicMock()
    AsyncOpenSearch = MagicMock()
    opensearch_bulk_helper_real = None
    opensearch_async_bulk_helper_real = None


# Import only the necessary functions from your package for this minimal test set
from langchain_opensearch.vectorstores import (
    OpenSearchVectorStore,
    _get_opensearch_client, # For dependency error testing
    _get_async_opensearch_client, # For dependency error testing
    # Only import helpers if they are directly tested, not if they are patched by public methods
    # _validate_embeddings_and_bulk_size, # Excluded from minimal set
    # _is_aoss_enabled, # Excluded from minimal set
    # _validate_aoss_with_engines, # Excluded from minimal set
)

# Define these based on real imports or None for conditional skipping
opensearch_bulk_helper: Callable[..., Any] | None = opensearch_bulk_helper_real


# Use pytest.mark.asyncio just in case, even if no async tests included
pytestmark = pytest.mark.asyncio


# --- Minimal Test Class focusing on Issue/PR Core Requirements ---
class TestOpenSearchVectorStoreMinimal(unittest.TestCase):

    def test_import(self) -> None:
        """Test that the OpenSearchVectorStore class can be imported."""
        self.assertTrue(hasattr(OpenSearchVectorStore, "__init__"))
        # Check the explicitly mentioned core methods
        self.assertTrue(hasattr(OpenSearchVectorStore, "add_texts"))
        self.assertTrue(hasattr(OpenSearchVectorStore, "similarity_search"))


    # Test basic Initialization paths (client, url, and the error case)
    def test_init_with_client(self) -> None:
        """Test __init__ successfully sets client when provided."""
        mock_client = MagicMock(spec=OpenSearch)
        mock_embeddings = MagicMock(spec=Embeddings)
        index_name = "test_index"

        store = OpenSearchVectorStore(
            client=mock_client,
            index_name=index_name,
            embedding_function=mock_embeddings,
        )
        self.assertIs(store.client, mock_client)
        self.assertIsNone(store.async_client) # Should not initialize async if client is provided
        self.assertEqual(store.index_name, index_name)
        self.assertIs(store.embedding_function, mock_embeddings)

    # Patch the helper functions that __init__ calls when url is provided
    @patch("langchain_opensearch.vectorstores._get_async_opensearch_client")
    @patch("langchain_opensearch.vectorstores._get_opensearch_client")
    # Patch is_aoss_enabled as it's called by __init__
    @patch("langchain_opensearch.vectorstores._is_aoss_enabled", return_value=False) # Assume non-AOSS for minimal case
    def test_init_with_url(
        self, mock_is_aoss_enabled: MagicMock, mock_get_sync_client: MagicMock, mock_get_async_client: MagicMock
    ) -> None:
        """Test __init__ successfully creates clients when opensearch_url is provided."""
        mock_sync_client_instance = MagicMock(spec=OpenSearch)
        # Return dummy clients from the patched getters
        mock_get_sync_client.return_value = mock_sync_client_instance
        mock_get_async_client.return_value = MagicMock(spec=AsyncOpenSearch) # Still needs to be called


        mock_embeddings = MagicMock(spec=Embeddings)
        index_name = "test_index"
        opensearch_url = "http://my-os:9200"
        # Pass minimal kwargs needed by __init__
        extra_kwargs = {"timeout": 60} # Example kwarg that might be passed to getters

        store = OpenSearchVectorStore(
            index_name=index_name,
            embedding_function=mock_embeddings,
            opensearch_url=opensearch_url,
            **extra_kwargs # Pass kwargs
        )

        # Assert getters were called with URL and kwargs
        mock_get_sync_client.assert_called_once_with(opensearch_url, **extra_kwargs)
        mock_get_async_client.assert_called_once_with(opensearch_url, **extra_kwargs)

        # Assert clients are set
        self.assertIs(store.client, mock_sync_client_instance)
        self.assertIsInstance(store.async_client, MagicMock) # Check it's a mock async client

        # Assert is_aoss check was made
        mock_is_aoss_enabled.assert_called_once()


    def test_init_raises_error_if_no_client_or_url(self) -> None:
        """Test __init__ raises ValueError if neither client nor opensearch_url is provided."""
        expected_message = "Either 'client' or 'opensearch_url' must be provided."
        mock_embeddings = MagicMock(spec=Embeddings)
        with pytest.raises(ValueError, match=expected_message):
            OpenSearchVectorStore(index_name="test_index", embedding_function=mock_embeddings)


    # Test Dependency Import Errors for client getters
    @patch("opensearchpy.OpenSearch", side_effect=ImportError("test ImportError"))
    def test_get_opensearch_client_import_error(self, mock_opensearch_class: MagicMock) -> None:
        """Test _get_opensearch_client raises ImportError if opensearch-py sync client is not installable."""
        url = "http://dummy-url:9200"
        with pytest.raises(ImportError, match="Could not import OpenSearch"):
            _get_opensearch_client(url)
        mock_opensearch_class.assert_called_once_with(url) # Check URL is passed to the attempt


    @patch("opensearchpy.AsyncOpenSearch", side_effect=ImportError("test ImportError"))
    def test_get_async_opensearch_client_import_error(self, mock_async_opensearch_class: MagicMock) -> None:
        """Test _get_async_opensearch_client raises ImportError if opensearch-py async client is not installable."""
        url = "http://dummy-url:9200"
        with pytest.raises(ImportError, match="Could not import AsyncOpenSearch"):
            _get_async_opensearch_client(url)
        mock_async_opensearch_class.assert_called_once_with(url)


    # Test add_texts method (calls internal __add)
    # Patch the internal __add method (name mangled)
    @patch.object(OpenSearchVectorStore, "_OpenSearchVectorStore__add", return_value=["mock_id1", "mock_id2"])
    def test_add_texts_mock(self, mock_add: MagicMock) -> None:
        """Test add_texts calls internal __add helper correctly after embedding."""
        mock_client = MagicMock(spec=OpenSearch)
        mock_embeddings = MagicMock(spec=Embeddings)
        texts = ["Hello", "World"]
        embeddings_list = [[0.1, 0.2], [0.3, 0.4]]
        mock_embeddings.embed_documents.return_value = embeddings_list # Mock embedding call

        vector_store = OpenSearchVectorStore(client=mock_client, index_name="test_index", embedding_function=mock_embeddings)

        metadatas = [{"a": 1}, {"b": 2}]
        ids = ["id1", "id2"]

        result_ids = vector_store.add_texts(texts, metadatas=metadatas, ids=ids)

        # Assert embedding function was called
        mock_embeddings.embed_documents.assert_called_once_with(texts)

        # Assert internal __add helper was called with correct args
        # __add expects texts as Iterable, embeddings as list, metadatas, ids, bulk_size, and other kwargs
        mock_add.assert_called_once()
        # add_texts converts iterable texts to list and passes embeddings, metadatas, ids
        expected_add_args = (list(texts), embeddings_list)
        # add_texts passes metadatas, ids, and default bulk_size as kwargs
        expected_add_kwargs = {"metadatas": metadatas, "ids": ids, "bulk_size": 500} # Check default bulk_size

        # Use call() to compare arguments explicitly
        mock_add.assert_called_once_with(*expected_add_args, **expected_add_kwargs)

        # Check the return value
        self.assertEqual(result_ids, ["mock_id1", "mock_id2"])


    # Test similarity_search method (calls embed_query and internal _raw_similarity_search_with_score_by_vector)
    # Patch the internal _raw_similarity_search_with_score_by_vector method (name mangled if private)
    # Assuming _raw_similarity_search_with_score_by_vector is public or needs patching by name
    @patch.object(OpenSearchVectorStore, "_raw_similarity_search_with_score_by_vector")
    def test_similarity_search_mock(self, mock_raw_search: MagicMock) -> None:
        """Test similarity_search calls embed_query and _raw_similarity_search_with_score_by_vector correctly."""
        mock_embeddings = MagicMock(spec=Embeddings)
        query_vector = [0.1, 0.2]
        mock_embeddings.embed_query.return_value = query_vector # Mock embedding call

        # Mock return value from _raw_similarity_search_with_score_by_vector
        # Should be a list of dicts representing search hits
        mock_raw_search.return_value = [{
             "_id": "test_id", "_score": 0.9,
             "_source": {"text": "Hello", "metadata": {"source": "doc1"}} # Include minimal source fields
        }]

        vector_store = OpenSearchVectorStore(client=MagicMock(spec=OpenSearch), index_name="test_index", embedding_function=mock_embeddings)

        query = "test query"
        k_value = 1

        results = vector_store.similarity_search(query, k=k_value)

        # Assert embedding function was called
        mock_embeddings.embed_query.assert_called_once_with(query)

        # Assert internal raw search helper was called with correct args
        # similarity_search calls similarity_search_with_score, which calls _raw_...
        # Check args passed to _raw_...
        mock_raw_search.assert_called_once_with(
            embedding=query_vector, # The embedding is passed
            k=k_value,
            score_threshold=0.0, # Check default threshold
            query_text=query, # query_text is added by similarity_search_with_score
            # Check for other kwargs passed from similarity_search if any
        )

        # Check the final results format (List of Documents)
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], Document)
        self.assertEqual(results[0].page_content, "Hello")
        self.assertEqual(results[0].metadata, {"source": "doc1"}) # Check metadata is parsed correctly