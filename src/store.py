from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb  # noqa: F401

            # TODO: initialize chromadb client + collection
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        """Build a normalized stored record for one document."""
        embedding = self._embedding_fn(doc.content)
        record = {
            'id': doc.id,
            'content': doc.content,
            'embedding': embedding,
            'metadata': {**doc.metadata, 'doc_id': doc.id}
        }
        return record

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        """Run in-memory similarity search over provided records."""
        if not records:
            return []
        
        query_embedding = self._embedding_fn(query)
        
        # Compute similarity scores for all records
        scored_records = []
        for record in records:
            score = _dot(query_embedding, record['embedding'])
            scored_records.append({
                'content': record['content'],
                'score': score,
                'metadata': record['metadata']
            })
        
        # Sort by score descending and return top_k
        scored_records.sort(key=lambda x: x['score'], reverse=True)
        return scored_records[:top_k]

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        if not docs:
            return
        
        if self._use_chroma and self._collection is not None:
            # ChromaDB implementation
            ids = []
            documents = []
            embeddings = []
            metadatas = []
            
            for doc in docs:
                record = self._make_record(doc)
                ids.append(f"{self._collection_name}_{self._next_index}")
                documents.append(record['content'])
                embeddings.append(record['embedding'])
                metadatas.append(record['metadata'])
                self._next_index += 1
            
            self._collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
        else:
            # In-memory implementation
            for doc in docs:
                record = self._make_record(doc)
                self._store.append(record)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        if self._use_chroma and self._collection is not None:
            # ChromaDB implementation
            query_embedding = self._embedding_fn(query)
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, self._collection.count())
            )
            
            search_results = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    search_results.append({
                        'content': doc,
                        'score': 1.0 - results['distances'][0][i] if 'distances' in results else 1.0,
                        'metadata': results['metadatas'][0][i] if 'metadatas' in results else {}
                    })
            return search_results
        else:
            # In-memory implementation
            return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        if self._use_chroma and self._collection is not None:
            return self._collection.count()
        else:
            return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        if metadata_filter is None:
            # No filter, just do regular search
            return self.search(query, top_k)
        
        if self._use_chroma and self._collection is not None:
            # ChromaDB implementation with where clause
            query_embedding = self._embedding_fn(query)
            
            # Build where clause for ChromaDB
            where_clause = {k: v for k, v in metadata_filter.items()}
            
            try:
                results = self._collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    where=where_clause
                )
                
                search_results = []
                if results['documents'] and results['documents'][0]:
                    for i, doc in enumerate(results['documents'][0]):
                        search_results.append({
                            'content': doc,
                            'score': 1.0 - results['distances'][0][i] if 'distances' in results else 1.0,
                            'metadata': results['metadatas'][0][i] if 'metadatas' in results else {}
                        })
                return search_results
            except Exception:
                # Fallback to manual filtering
                pass
        
        # In-memory implementation: filter then search
        filtered_records = []
        for record in self._store:
            # Check if all filter criteria match
            matches = all(
                record['metadata'].get(key) == value
                for key, value in metadata_filter.items()
            )
            if matches:
                filtered_records.append(record)
        
        return self._search_records(query, filtered_records, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        if self._use_chroma and self._collection is not None:
            # ChromaDB implementation
            try:
                # Query for documents with this doc_id
                results = self._collection.get(where={"doc_id": doc_id})
                if results['ids']:
                    self._collection.delete(ids=results['ids'])
                    return True
                return False
            except Exception:
                return False
        else:
            # In-memory implementation
            initial_size = len(self._store)
            self._store = [
                record for record in self._store
                if record['metadata'].get('doc_id') != doc_id
            ]
            return len(self._store) < initial_size
