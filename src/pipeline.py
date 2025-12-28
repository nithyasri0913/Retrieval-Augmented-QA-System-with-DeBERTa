"""
Production QA System Pipeline
"""

from typing import List, Dict, Optional, Any
import time
from .retriever import DocumentRetriever
from .qa_model import QAModel
from .langchain_integration import RAGChain


class ProductionQASystem:
    """
    Production-ready Question Answering system combining
    dense retrieval and extractive QA.
    """

    def __init__(
        self,
        retriever_model: str = 'all-MiniLM-L6-v2',
        qa_model: str = 'deepset/deberta-v3-base-squad2',
        device: Optional[str] = None,
        top_k: int = 3,
        qa_threshold: float = 0.3,
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ):
        """
        Initialize the production QA system.

        Args:
            retriever_model: Sentence transformer model name
            qa_model: DeBERTa model name
            device: Device to run on ('cuda', 'cpu', or None for auto)
            top_k: Number of passages to retrieve
            qa_threshold: Minimum confidence threshold for answers
            chunk_size: Document chunk size in words
            chunk_overlap: Overlap between chunks in words
        """
        self.top_k = top_k
        self.qa_threshold = qa_threshold
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        print("Initializing Production QA System...")
        print("=" * 50)

        # Initialize retriever
        print(f"\n1. Loading Retriever: {retriever_model}")
        self.retriever = DocumentRetriever(
            model_name=retriever_model,
            device=device
        )

        # Initialize QA model
        print(f"\n2. Loading QA Model: {qa_model}")
        device_id = None if device is None else (0 if device == 'cuda' else -1)
        self.qa_model = QAModel(
            model_name=qa_model,
            device=device_id
        )

        # Initialize RAG chain
        print("\n3. Initializing RAG Chain")
        self.rag_chain = RAGChain(
            retriever=self.retriever,
            qa_model=self.qa_model,
            top_k=top_k,
            qa_threshold=qa_threshold,
            rerank=True
        )

        self.is_indexed = False
        print("\n" + "=" * 50)
        print("System initialized successfully!")

    def index_corpus(
        self,
        documents: List[str],
        metadata: Optional[List[Dict]] = None,
        use_chunking: bool = True
    ):
        """
        Index a corpus of documents.

        Args:
            documents: List of document texts
            metadata: Optional metadata for each document
            use_chunking: Whether to chunk documents
        """
        print("\nIndexing corpus...")
        print("=" * 50)

        start_time = time.time()

        if use_chunking:
            print(f"Chunking {len(documents)} documents...")
            chunks = self.retriever.chunk_documents(
                documents,
                chunk_size=self.chunk_size,
                overlap=self.chunk_overlap,
                metadata=metadata
            )
            print(f"Created {len(chunks)} chunks")

            # Build index from chunks
            self.retriever.build_index(chunks)
        else:
            # Index documents directly
            self.retriever.build_index(documents, metadata)

        elapsed = time.time() - start_time
        self.is_indexed = True

        print(f"\nIndexing completed in {elapsed:.2f} seconds")
        print("=" * 50)

    def answer_question(
        self,
        question: str,
        return_all_candidates: bool = False,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Answer a question using the RAG pipeline.

        Args:
            question: The question to answer
            return_all_candidates: Whether to return all candidate answers
            verbose: Whether to print detailed information

        Returns:
            Dictionary containing answer and metadata
        """
        if not self.is_indexed:
            raise ValueError(
                "Corpus not indexed. Call index_corpus() first."
            )

        if verbose:
            print(f"\nQuestion: {question}")
            print("-" * 50)

        start_time = time.time()

        # Get answer from RAG chain
        result = self.rag_chain.answer_question(question)

        elapsed = time.time() - start_time

        if verbose:
            if result['answer']:
                print(f"Answer: {result['answer']}")
                print(f"Confidence: {result['confidence']:.3f}")
                print(f"Combined Score: {result['best_score']:.3f}")
                print(f"Candidates found: {result['num_candidates']}")
            else:
                print("No answer found with sufficient confidence")
            print(f"Time: {elapsed:.3f}s")
            print("-" * 50)

        # Add timing information
        result['query_time'] = elapsed

        # Filter candidates if not requested
        if not return_all_candidates and 'all_candidates' in result:
            result['num_candidates'] = len(result['all_candidates'])
            del result['all_candidates']

        return result

    def batch_answer(
        self,
        questions: List[str],
        verbose: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Answer multiple questions.

        Args:
            questions: List of questions to answer
            verbose: Whether to print progress

        Returns:
            List of answer dictionaries
        """
        if verbose:
            print(f"\nAnswering {len(questions)} questions...")
            print("=" * 50)

        start_time = time.time()
        results = []

        for i, question in enumerate(questions, 1):
            if verbose:
                print(f"\n[{i}/{len(questions)}] {question}")

            result = self.answer_question(question, verbose=False)
            results.append(result)

            if verbose and result['answer']:
                print(f"  → {result['answer']} (confidence: {result['confidence']:.3f})")

        elapsed = time.time() - start_time

        if verbose:
            print("\n" + "=" * 50)
            print(f"Completed in {elapsed:.2f}s")
            print(f"Average time per question: {elapsed/len(questions):.3f}s")

        return results

    def search(self, query: str, k: Optional[int] = None) -> List[Dict]:
        """
        Search for relevant documents without QA.

        Args:
            query: Search query
            k: Number of results (defaults to self.top_k)

        Returns:
            List of retrieved documents
        """
        if not self.is_indexed:
            raise ValueError("Corpus not indexed. Call index_corpus() first.")

        k = k or self.top_k
        return self.retriever.retrieve(query, k=k)

    def save(self, directory: str):
        """
        Save the system to disk.

        Args:
            directory: Directory to save the system
        """
        import os
        os.makedirs(directory, exist_ok=True)

        # Save retriever (includes index and documents)
        self.retriever.save(directory)

        # Save configuration
        import json
        config = {
            'top_k': self.top_k,
            'qa_threshold': self.qa_threshold,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'qa_model': self.qa_model.model_name
        }
        with open(os.path.join(directory, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)

        print(f"\nSystem saved to {directory}")

    @classmethod
    def load(
        cls,
        directory: str,
        device: Optional[str] = None
    ):
        """
        Load a saved system from disk.

        Args:
            directory: Directory containing saved system
            device: Device to load on

        Returns:
            Loaded ProductionQASystem instance
        """
        import json
        import os

        # Load configuration
        with open(os.path.join(directory, 'config.json'), 'r') as f:
            config = json.load(f)

        # Load retriever
        retriever = DocumentRetriever.load(directory, device=device)

        # Create system instance
        system = cls(
            retriever_model=retriever.model_name,
            qa_model=config['qa_model'],
            device=device,
            top_k=config['top_k'],
            qa_threshold=config['qa_threshold'],
            chunk_size=config['chunk_size'],
            chunk_overlap=config['chunk_overlap']
        )

        # Replace retriever with loaded one
        system.retriever = retriever
        system.is_indexed = True

        # Update RAG chain
        system.rag_chain.retriever = retriever

        print(f"\nSystem loaded from {directory}")
        return system

    def get_stats(self) -> Dict[str, Any]:
        """
        Get system statistics.

        Returns:
            Dictionary with system statistics
        """
        stats = {
            'indexed': self.is_indexed,
            'num_documents': len(self.retriever.documents) if self.is_indexed else 0,
            'retriever_model': self.retriever.model_name,
            'qa_model': self.qa_model.model_name,
            'top_k': self.top_k,
            'qa_threshold': self.qa_threshold,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap
        }
        return stats

    def __repr__(self):
        stats = self.get_stats()
        return f"""ProductionQASystem(
    Indexed: {stats['indexed']}
    Documents: {stats['num_documents']}
    Retriever: {stats['retriever_model']}
    QA Model: {stats['qa_model']}
    Top-K: {stats['top_k']}
)"""
