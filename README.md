# langchain-opensearch

A standalone LangChain integration package for OpenSearch, providing `OpenSearchVectorStore` for efficient vector storage and similarity search.

## Overview

This package fulfills [LangChain issue #30323](https://github.com/langchain-ai/langchain/issues/30323) by creating a standalone `langchain-opensearch` module, migrating `OpenSearchVectorStore` from `langchain_community` to improve modularity and dependency management. As detailed in [PR #30892](https://github.com/langchain-ai/langchain/pull/30892), the package was developed with updated `langchain-core` imports, comprehensive integration tests, and high code quality using `black` formatting and `ruff` linting.

## Features

- **OpenSearchVectorStore**: Stores and queries document embeddings in OpenSearch.
- **Similarity Search**: Performs efficient similarity searches.
- **LangChain Integration**: Seamlessly works with `langchain-core`.
- **Integration Tests**: Includes mocked tests for reliability.

## Installation

```bash
pip install langchain-opensearch

## Usage

``` 
from langchain_opensearch import OpenSearchVectorStore
from langchain_core.embeddings import Embeddings

class DummyEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[0.1, 0.2]] * len(texts)
    def embed_query(self, text):
        return [0.1, 0.2]

embeddings = DummyEmbeddings()
vector_store = OpenSearchVectorStore(
    opensearch_url="http://localhost:9200",
    index_name="my_index",
    embedding_function=embeddings
)

texts = ["Hello, world!", "LangChain is awesome"]
vector_store.add_texts(texts)
results = vector_store.similarity_search("world", k=1)
print([doc.page_content for doc in results])  # ['Hello, world!']

```

## Setting Up OpenSearch (for local testing)

For local development and testing, you can quickly set up an OpenSearch instance using Docker:

```bash
docker run -d --name opensearch \
  -p 9200:9200 -p 9600:9600 \
  -e "discovery.type=single-node" \
  -e "OPENSEARCH_INITIAL_ADMIN_PASSWORD=StrongPass123!@#" \
  opensearchproject/opensearch:latest
```

Verify that the container is running:

```bash
curl -k -u admin:StrongPass123!@# https://localhost:9200
```

To stop and remove the OpenSearch container:

```bash
docker stop opensearch
docker rm opensearch
```

## Testing

### Mock Tests

Run unit and mock integration tests:

```bash
python -m unittest langchain-opensearch/tests/test_vectorstores.py
```

### Real OpenSearch Test

Validate with a running OpenSearch instance:

```bash
python langchain-opensearch/test_real_opensearch.py
```

## Requirements

- OpenSearch instance (version compatible with `opensearch-py` requirements).
- Python 3.8+
- `opensearch-py>=2.4.0`
- `langchain-core>=0.1.0`
- `numpy>=1.0.0`

## Changes Made

This package represented the extraction of `OpenSearchVectorStore` from `langchain_community`. My key changes included:

- `langchain_opensearch/vectorstores.py`: Migrated `OpenSearchVectorStore`, updated imports to `langchain-core`, and optimized the `add_texts` method.
- `tests/test_vectorstores.py`: Added comprehensive unit and mock tests to ensure functionality (8 tests, all passing)..
- `pyproject.toml`: Defined the package's dependencies.
- `.devcontainer/*`: Configured a consistent development and testing environment using Codespaces.
- `README.md`: Documented usage, setup, testing, and changes (this file).