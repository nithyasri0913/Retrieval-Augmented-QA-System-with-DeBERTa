"""
Evaluate the QA system on SQuAD dataset
"""

import sys
sys.path.append('..')

from src.pipeline import ProductionQASystem
from datasets import load_dataset
from tqdm import tqdm
import json


def evaluate_on_squad(
    num_samples: int = 100,
    split: str = 'validation',
    save_results: bool = True
):
    """
    Evaluate the QA system on SQuAD dataset.

    Args:
        num_samples: Number of samples to evaluate
        split: Dataset split ('train' or 'validation')
        save_results: Whether to save results to file
    """
    print("Loading SQuAD dataset...")
    dataset = load_dataset('squad', split=split)

    # Take subset for evaluation
    dataset = dataset.select(range(min(num_samples, len(dataset))))

    print(f"\nEvaluating on {len(dataset)} samples from SQuAD {split} set")
    print("=" * 70)

    # Initialize QA system
    qa_system = ProductionQASystem(
        retriever_model='all-MiniLM-L6-v2',
        qa_model='deepset/deberta-v3-base-squad2',
        top_k=1,  # Only need top passage since context is provided
        qa_threshold=0.0,  # No threshold for evaluation
        chunk_size=512,
        chunk_overlap=50
    )

    results = []
    correct_exact = 0
    total_answered = 0

    for example in tqdm(dataset, desc="Evaluating"):
        question = example['question']
        context = example['context']
        gold_answers = example['answers']['text']

        # Index the context (treating it as a single document)
        qa_system.index_corpus([context], use_chunking=False)

        # Get answer
        result = qa_system.answer_question(question, verbose=False)

        predicted_answer = result.get('answer', '')
        confidence = result.get('confidence', 0.0)

        # Check exact match (case insensitive)
        is_correct = False
        if predicted_answer:
            total_answered += 1
            predicted_lower = predicted_answer.lower().strip()
            for gold in gold_answers:
                if predicted_lower == gold.lower().strip():
                    is_correct = True
                    correct_exact += 1
                    break

        results.append({
            'question': question,
            'context': context,
            'gold_answers': gold_answers,
            'predicted_answer': predicted_answer,
            'confidence': confidence,
            'correct': is_correct
        })

    # Calculate metrics
    exact_match = (correct_exact / len(dataset)) * 100 if dataset else 0
    answer_rate = (total_answered / len(dataset)) * 100 if dataset else 0

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"Total samples: {len(dataset)}")
    print(f"Answered: {total_answered} ({answer_rate:.2f}%)")
    print(f"Exact Match: {correct_exact}/{len(dataset)} ({exact_match:.2f}%)")
    print("=" * 70)

    # Show some examples
    print("\nSample Predictions:")
    print("-" * 70)
    for i, result in enumerate(results[:5], 1):
        print(f"\n{i}. Question: {result['question']}")
        print(f"   Gold: {result['gold_answers'][0]}")
        print(f"   Predicted: {result['predicted_answer']}")
        print(f"   Correct: {result['correct']} (Confidence: {result['confidence']:.3f})")

    # Save results
    if save_results:
        output = {
            'metrics': {
                'exact_match': exact_match,
                'answer_rate': answer_rate,
                'total_samples': len(dataset),
                'correct': correct_exact,
                'answered': total_answered
            },
            'predictions': results
        }

        with open('squad_evaluation_results.json', 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\nResults saved to squad_evaluation_results.json")

    return results


if __name__ == "__main__":
    evaluate_on_squad(num_samples=100)
