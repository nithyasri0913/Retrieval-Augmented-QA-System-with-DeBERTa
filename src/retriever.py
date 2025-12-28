"""
Dense Retrieval System using Sentence Transformers and FAISS
"""

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm
import pickle
import os


class DocumentRetriever:
    """
    Dense retrieval system using Sentence Transformers for encoding
    and FAISS for efficient similarity search.
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', device: Optional[str] = None):
        """
        Initialize the document retriever.

        Args:
            model_name: Name of the sentence transformer model
            device: Device to run on ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name
        self.encoder = SentenceTransformer(model_name, device=device)
        self.index = None
        self.documents = []
        self.document_metadata = []

    def chunk_documents(
        self,
        texts: List[str],
        chunk_size: int = 512,
        overlap: int = 50,
        metadata: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """
        Split documents into overlapping chunks.

        Args:
            texts: List of document texts
            chunk_size: Maximum tokens per chunk
            overlap: Number of overlapping tokens between chunks
            metadata: Optional metadata for each document

        Returns:
            List of dictionaries containing chunk text and metadata
        """
        chunks = []

        for idx, text in enumerate(tqdm(texts, desc="Chunking documents")):
            # Simple word-based chunking (could be improved with tokenizer)
            words = text.split()

            doc_metadata = metadata[idx] if metadata else {'doc_id': idx}

            for i in range(0, len(words), chunk_size - overlap):
                chunk_words = words[i:i + chunk_size]
                chunk_text = ' '.join(chunk_words)

                if chunk_text.strip():  # Only add non-empty chunks
                    chunk_info = {
                        'text': chunk_text,
                        'chunk_id': len(chunks),
                        'doc_id': idx,
                        'start_word': i,
                        'end_word': i + len(chunk_words),
                        **doc_metadata
                    }
                    chunks.append(chunk_info)

        return chunks

    def build_index(self, documents: List[str], metadata: Optional[List[Dict]] = None):
        """
        Build FAISS index from documents.

        Args:
            documents: List of document texts or chunk dictionaries
            metadata: Optional metadata for each document
        """
        # Handle both raw texts and pre-chunked documents
        if isinstance(documents[0], dict):
            self.documents = [doc['text'] for doc in documents]
            self.document_metadata = documents
        else:
            self.documents = documents
            self.document_metadata = metadata or [{'doc_id': i} for i in range(len(documents))]

        print(f"Encoding {len(self.documents)} documents...")
        embeddings = self.encoder.encode(
            self.documents,
            show_progress_bar=True,
            convert_to_numpy=True,
            batch_size=32
        )

        # Build FAISS index
        dimension = embeddings.shape[1]
        print(f"Building FAISS index (dimension: {dimension})...")

        # Use IndexFlatL2 for exact search (can switch to IndexIVFFlat for speed)
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))

        print(f"Index built with {self.index.ntotal} vectors")

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """
        Retrieve top-k most relevant documents for a query.

        Args:
            query: Search query
            k: Number of documents to retrieve

        Returns:
            List of dictionaries with document text, metadata, and distance scores
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")

        # Encode query
        query_embedding = self.encoder.encode(
            [query],
            convert_to_numpy=True
        ).astype('float32')

        # Search
        distances, indices = self.index.search(query_embedding, k)

        # Prepare results
        results = []
        for idx, (distance, doc_idx) in enumerate(zip(distances[0], indices[0])):
            result = {
                'text': self.documents[doc_idx],
                'distance': float(distance),
                'rank': idx + 1,
                'relevance_score': 1 / (1 + distance),  # Convert distance to similarity
            }

            # Add metadata if available
            if self.document_metadata:
                result.update(self.document_metadata[doc_idx])

            results.append(result)

        return results

    def save(self, directory: str):
        """
        Save the retriever to disk.

        Args:
            directory: Directory to save the retriever
        """
        os.makedirs(directory, exist_ok=True)

        # Save FAISS index
        if self.index is not None:
            faiss.write_index(self.index, os.path.join(directory, 'faiss.index'))

        # Save documents and metadata
        with open(os.path.join(directory, 'documents.pkl'), 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'metadata': self.document_metadata,
                'model_name': self.model_name
            }, f)

        print(f"Retriever saved to {directory}")

    @classmethod
    def load(cls, directory: str, device: Optional[str] = None):
        """
        Load a saved retriever from disk.

        Args:
            directory: Directory containing saved retriever
            device: Device to load model on

        Returns:
            Loaded DocumentRetriever instance
        """
        # Load documents and metadata
        with open(os.path.join(directory, 'documents.pkl'), 'rb') as f:
            data = pickle.load(f)

        # Create retriever instance
        retriever = cls(model_name=data['model_name'], device=device)
        retriever.documents = data['documents']
        retriever.document_metadata = data['metadata']

        # Load FAISS index
        index_path = os.path.join(directory, 'faiss.index')
        if os.path.exists(index_path):
            retriever.index = faiss.read_index(index_path)

        print(f"Retriever loaded from {directory}")
        return retriever
