"""
Extractive QA Model using DeBERTa
"""

from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
import torch
from typing import Dict, Optional, List
import warnings


class QAModel:
    """
    Extractive Question Answering model using DeBERTa.
    """

    def __init__(
        self,
        model_name: str = 'deepset/deberta-v3-base-squad2',
        device: Optional[int] = None,
        max_length: int = 384,
        doc_stride: int = 128
    ):
        """
        Initialize the QA model.

        Args:
            model_name: Hugging Face model name
            device: Device to run on (None for auto, 0 for GPU, -1 for CPU)
            max_length: Maximum sequence length
            doc_stride: Stride for handling long contexts
        """
        self.model_name = model_name
        self.max_length = max_length
        self.doc_stride = doc_stride

        # Auto-detect device if not specified
        if device is None:
            device = 0 if torch.cuda.is_available() else -1

        print(f"Loading QA model: {model_name}")
        print(f"Device: {'GPU' if device >= 0 else 'CPU'}")

        # Initialize pipeline
        self.qa_pipeline = pipeline(
            "question-answering",
            model=model_name,
            tokenizer=model_name,
            device=device,
            max_seq_len=max_length,
            doc_stride=doc_stride
        )

        # Load tokenizer for additional functionality
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def extract_answer(
        self,
        question: str,
        context: str,
        threshold: float = 0.3,
        top_k: int = 1,
        handle_impossible_answer: bool = True
    ) -> Optional[Dict]:
        """
        Extract answer from context for a given question.

        Args:
            question: The question to answer
            context: The context containing the answer
            threshold: Minimum confidence score (0-1)
            top_k: Number of top answers to return
            handle_impossible_answer: Whether to handle "no answer" cases

        Returns:
            Dictionary with answer, score, and position, or None if no answer found
        """
        try:
            # Run QA pipeline
            result = self.qa_pipeline(
                question=question,
                context=context,
                top_k=top_k,
                handle_impossible_answer=handle_impossible_answer
            )

            # Handle single vs multiple answers
            if top_k == 1:
                result = [result]

            # Filter by threshold and format results
            valid_answers = []
            for ans in result:
                # Filter out empty or very short answers (likely false positives)
                answer_text = ans['answer'].strip()
                if ans['score'] >= threshold and len(answer_text) > 0:
                    valid_answers.append({
                        'answer': ans['answer'],
                        'score': float(ans['score']),
                        'start': ans['start'],
                        'end': ans['end'],
                        'context': context
                    })

            # Return best answer or None
            if valid_answers:
                return valid_answers[0] if top_k == 1 else valid_answers
            return None

        except Exception as e:
            warnings.warn(f"Error extracting answer: {str(e)}")
            return None

    def batch_extract_answers(
        self,
        question: str,
        contexts: List[str],
        threshold: float = 0.3
    ) -> List[Optional[Dict]]:
        """
        Extract answers from multiple contexts for the same question.

        Args:
            question: The question to answer
            contexts: List of contexts to search
            threshold: Minimum confidence score

        Returns:
            List of answer dictionaries (or None for each context)
        """
        results = []
        for context in contexts:
            answer = self.extract_answer(question, context, threshold)
            results.append(answer)
        return results

    def get_answer_with_context(
        self,
        question: str,
        context: str,
        threshold: float = 0.3,
        context_window: int = 100
    ) -> Optional[Dict]:
        """
        Extract answer and include surrounding context.

        Args:
            question: The question to answer
            context: The context containing the answer
            threshold: Minimum confidence score
            context_window: Number of characters to include before/after answer

        Returns:
            Dictionary with answer, score, and extended context
        """
        result = self.extract_answer(question, context, threshold)

        if result:
            # Add surrounding context
            start = max(0, result['start'] - context_window)
            end = min(len(context), result['end'] + context_window)
            result['extended_context'] = context[start:end]
            result['context_start'] = start
            result['context_end'] = end

        return result

    def validate_answer(
        self,
        question: str,
        answer: str,
        context: str
    ) -> Dict:
        """
        Validate if an answer is present in the context.

        Args:
            question: The question
            answer: The proposed answer
            context: The context

        Returns:
            Dictionary with validation results
        """
        # Check if answer is in context
        answer_in_context = answer.lower() in context.lower()

        # Get model's answer for comparison
        model_answer = self.extract_answer(question, context, threshold=0.0)

        validation = {
            'answer_in_context': answer_in_context,
            'model_answer': model_answer['answer'] if model_answer else None,
            'model_score': model_answer['score'] if model_answer else 0.0,
            'answers_match': False
        }

        if model_answer:
            validation['answers_match'] = (
                answer.lower().strip() == model_answer['answer'].lower().strip()
            )

        return validation

    def __call__(self, question: str, context: str, **kwargs) -> Optional[Dict]:
        """
        Convenience method to extract answer.
        """
        return self.extract_answer(question, context, **kwargs)
