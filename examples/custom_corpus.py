"""
Example: Building a QA system with a custom corpus
This demonstrates how to use the system with your own documents
"""

import sys
sys.path.append('..')

from src.pipeline import ProductionQASystem
import os


def load_documents_from_directory(directory: str) -> list:
    """
    Load all text files from a directory.

    Args:
        directory: Path to directory containing text files

    Returns:
        List of document texts
    """
    documents = []
    metadata = []

    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
                documents.append(text)
                metadata.append({
                    'filename': filename,
                    'filepath': filepath
                })

    return documents, metadata


def main():
    """
    Example workflow for building a QA system with custom documents.
    """

    # Option 1: Use sample documents
    print("Creating sample corpus...")
    documents = [
        """
        Python is a high-level, interpreted programming language created by Guido van Rossum
        and first released in 1991. Python's design philosophy emphasizes code readability
        with significant use of whitespace. It supports multiple programming paradigms
        including procedural, object-oriented, and functional programming.
        """,
        """
        Machine learning is a subset of artificial intelligence that provides systems the
        ability to automatically learn and improve from experience without being explicitly
        programmed. It focuses on the development of computer programs that can access data
        and use it to learn for themselves. The primary aim is to allow computers to learn
        automatically without human intervention.
        """,
        """
        Natural Language Processing (NLP) is a branch of artificial intelligence that helps
        computers understand, interpret and manipulate human language. NLP draws from many
        disciplines including computer science and computational linguistics. Modern NLP
        algorithms are based on machine learning, especially statistical machine learning
        and deep learning models.
        """
    ]

    metadata = [
        {'topic': 'Programming Languages', 'title': 'Python Overview'},
        {'topic': 'AI', 'title': 'Machine Learning Basics'},
        {'topic': 'AI', 'title': 'NLP Introduction'}
    ]

    # Option 2: Load from directory (uncomment to use)
    # documents, metadata = load_documents_from_directory('./your_documents/')

    # Initialize the QA system
    print("\nInitializing QA System...")
    qa_system = ProductionQASystem(
        retriever_model='all-MiniLM-L6-v2',
        qa_model='deepset/deberta-v3-base-squad2',
        top_k=5,
        qa_threshold=0.2,
        chunk_size=512,
        chunk_overlap=50
    )

    # Index the corpus
    print("\nIndexing corpus...")
    qa_system.index_corpus(
        documents=documents,
        metadata=metadata,
        use_chunking=True
    )

    # Save the indexed system for later use
    print("\nSaving the system...")
    qa_system.save('./saved_qa_system')

    # Interactive Q&A
    print("\n" + "=" * 70)
    print("Interactive Question Answering")
    print("Type 'quit' to exit")
    print("=" * 70)

    while True:
        question = input("\nYour question: ").strip()

        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        if not question:
            continue

        result = qa_system.answer_question(
            question,
            return_all_candidates=True,
            verbose=False
        )

        print("\n" + "-" * 70)
        if result['answer']:
            print(f"Answer: {result['answer']}")
            print(f"Confidence: {result['confidence']:.3f}")

            # Show context
            if result.get('context'):
                print(f"\nContext snippet:")
                print(f"{result['context'][:200]}...")

            # Show alternative answers
            if result.get('all_candidates') and len(result['all_candidates']) > 1:
                print(f"\nAlternative answers:")
                for i, candidate in enumerate(result['all_candidates'][1:3], 1):
                    print(f"{i}. {candidate['answer']} (confidence: {candidate['confidence']:.3f})")
        else:
            print("No answer found with sufficient confidence.")
            print(result.get('message', ''))

        print("-" * 70)


def demo_loading_saved_system():
    """
    Demonstrate loading a previously saved system.
    """
    print("Loading saved QA system...")
    qa_system = ProductionQASystem.load('./saved_qa_system')

    print("\nSystem loaded successfully!")
    print(qa_system)

    # Ask a question
    result = qa_system.answer_question(
        "What is Python?",
        verbose=True
    )


if __name__ == "__main__":
    # Run main example
    main()

    # Uncomment to test loading
    # demo_loading_saved_system()
