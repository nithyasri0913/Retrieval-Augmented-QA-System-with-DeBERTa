"""
LangChain Integration for RAG Pipeline
"""

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from typing import List, Optional, Dict, Any
from .retriever import DocumentRetriever
from .qa_model import QAModel


class CustomRAGRetriever(BaseRetriever):
    """
    Custom LangChain retriever that combines dense retrieval with extractive QA.
    """

    retriever: DocumentRetriever
    qa_model: Optional[QAModel] = None
    top_k: int = 5
    qa_threshold: float = 0.3
    combine_scores: bool = True

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        retriever: DocumentRetriever,
        qa_model: Optional[QAModel] = None,
        top_k: int = 5,
        qa_threshold: float = 0.3,
        combine_scores: bool = True
    ):
        """
        Initialize the RAG retriever.

        Args:
            retriever: DocumentRetriever instance
            qa_model: Optional QAModel instance for answer extraction
            top_k: Number of documents to retrieve
            qa_threshold: Minimum QA confidence score
            combine_scores: Whether to combine retrieval and QA scores
        """
        super().__init__()
        self.retriever = retriever
        self.qa_model = qa_model
        self.top_k = top_k
        self.qa_threshold = qa_threshold
        self.combine_scores = combine_scores

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: The search query
            run_manager: Callback manager (not used)

        Returns:
            List of LangChain Document objects
        """
        # Retrieve passages using dense retrieval
        passages = self.retriever.retrieve(query, k=self.top_k)

        documents = []

        # If QA model is available, extract answers
        if self.qa_model:
            for passage in passages:
                # Extract answer from passage
                answer_result = self.qa_model.extract_answer(
                    query,
                    passage['text'],
                    threshold=self.qa_threshold
                )

                if answer_result:
                    # Combine retrieval and QA scores
                    if self.combine_scores:
                        combined_score = (
                            passage['relevance_score'] * 0.4 +
                            answer_result['score'] * 0.6
                        )
                    else:
                        combined_score = answer_result['score']

                    # Create LangChain document with metadata
                    doc = Document(
                        page_content=passage['text'],
                        metadata={
                            'answer': answer_result['answer'],
                            'answer_score': answer_result['score'],
                            'retrieval_score': passage['relevance_score'],
                            'combined_score': combined_score,
                            'distance': passage['distance'],
                            'rank': passage['rank'],
                            'answer_start': answer_result['start'],
                            'answer_end': answer_result['end'],
                            **{k: v for k, v in passage.items() if k not in ['text', 'distance', 'rank', 'relevance_score']}
                        }
                    )
                    documents.append(doc)
        else:
            # Return retrieved passages without QA
            for passage in passages:
                doc = Document(
                    page_content=passage['text'],
                    metadata={
                        'retrieval_score': passage['relevance_score'],
                        'distance': passage['distance'],
                        'rank': passage['rank'],
                        **{k: v for k, v in passage.items() if k not in ['text', 'distance', 'rank', 'relevance_score']}
                    }
                )
                documents.append(doc)

        # Sort by combined score if using QA
        if self.qa_model and self.combine_scores and documents:
            documents.sort(
                key=lambda x: x.metadata.get('combined_score', 0),
                reverse=True
            )

        return documents

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """
        Async version of get_relevant_documents.
        """
        # For now, just call the sync version
        # Can be implemented with async models in the future
        return self._get_relevant_documents(query, run_manager=run_manager)


class RAGChain:
    """
    Complete RAG chain for question answering.
    """

    def __init__(
        self,
        retriever: DocumentRetriever,
        qa_model: QAModel,
        top_k: int = 5,
        qa_threshold: float = 0.3,
        rerank: bool = False
    ):
        """
        Initialize the RAG chain.

        Args:
            retriever: DocumentRetriever instance
            qa_model: QAModel instance
            top_k: Number of documents to retrieve
            qa_threshold: Minimum QA confidence score
            rerank: Whether to rerank results by QA score
        """
        self.retriever = retriever
        self.qa_model = qa_model
        self.top_k = top_k
        self.qa_threshold = qa_threshold
        self.rerank = rerank

    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Answer a question using the RAG pipeline.

        Args:
            question: The question to answer

        Returns:
            Dictionary with answer and supporting information
        """
        # Retrieve relevant passages
        passages = self.retriever.retrieve(question, k=self.top_k)

        # Extract answers from each passage
        candidates = []
        for passage in passages:
            answer = self.qa_model.extract_answer(
                question,
                passage['text'],
                threshold=self.qa_threshold
            )

            if answer:
                candidate = {
                    'answer': answer['answer'],
                    'confidence': answer['score'],
                    'context': passage['text'],
                    'retrieval_score': passage['relevance_score'],
                    'distance': passage['distance'],
                    'answer_start': answer['start'],
                    'answer_end': answer['end'],
                }

                # Add metadata if available
                if 'doc_id' in passage:
                    candidate['doc_id'] = passage['doc_id']
                if 'chunk_id' in passage:
                    candidate['chunk_id'] = passage['chunk_id']

                # Calculate combined score
                candidate['combined_score'] = (
                    passage['relevance_score'] * 0.4 +
                    answer['score'] * 0.6
                )

                candidates.append(candidate)

        # Sort candidates
        if self.rerank:
            candidates.sort(key=lambda x: x['confidence'], reverse=True)
        else:
            candidates.sort(key=lambda x: x['combined_score'], reverse=True)

        # Prepare response
        if candidates:
            best_answer = candidates[0]
            return {
                'answer': best_answer['answer'],
                'confidence': best_answer['confidence'],
                'context': best_answer['context'],
                'all_candidates': candidates,
                'num_candidates': len(candidates),
                'best_score': best_answer['combined_score']
            }
        else:
            return {
                'answer': None,
                'confidence': 0.0,
                'context': passages[0]['text'] if passages else None,
                'all_candidates': [],
                'num_candidates': 0,
                'best_score': 0.0,
                'message': 'No answer found with sufficient confidence'
            }

    def batch_answer(self, questions: List[str]) -> List[Dict[str, Any]]:
        """
        Answer multiple questions.

        Args:
            questions: List of questions

        Returns:
            List of answer dictionaries
        """
        return [self.answer_question(q) for q in questions]
