#!/usr/bin/env python3
"""
Simple interactive QA system - just run this!
"""

import sys
from src.pipeline import ProductionQASystem

# Sample documents - replace with your own!
documents = [
    "Python is a high-level, interpreted programming language created by Guido van Rossum and first released in 1991. Python's design philosophy emphasizes code readability with significant use of whitespace. It supports multiple programming paradigms including procedural, object-oriented, and functional programming.",
    "Machine learning is a subset of artificial intelligence that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. It focuses on the development of computer programs that can access data and use it to learn for themselves. The primary aim is to allow computers to learn automatically without human intervention.",
    "Natural Language Processing (NLP) is a branch of artificial intelligence that helps computers understand, interpret and manipulate human language. NLP draws from many disciplines including computer science and computational linguistics. Modern NLP algorithms are based on machine learning, especially statistical machine learning and deep learning models.",
    "The Eiffel Tower is located in Paris, France. It was built in 1889 and stands 330 meters tall.",
    "Albert Einstein developed the theory of relativity in 1915. He won the Nobel Prize in Physics in 1921."
]

print("="*70)
print("INTERACTIVE QA SYSTEM")
print("="*70)

print("\nInitializing...")
qa_system = ProductionQASystem(
    retriever_model='all-MiniLM-L6-v2',
    qa_model='deepset/deberta-v3-base-squad2',
    top_k=5,
    qa_threshold=0.0
)

print("\nIndexing documents...")
qa_system.index_corpus(documents, use_chunking=False)

print("\n" + "="*70)
print("Ready! Type your questions (or 'quit' to exit)")
print("="*70)

while True:
    try:
        question = input("\nYour question: ").strip()

        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        if not question:
            continue

        result = qa_system.answer_question(question, verbose=False)

        print("-"*70)
        if result['answer']:
            print(f"✓ Answer: {result['answer']}")
            print(f"  Confidence: {result['confidence']:.3f}")
        else:
            print("✗ No answer found")
        print("-"*70)

    except KeyboardInterrupt:
        print("\n\nGoodbye!")
        break
    except EOFError:
        break
