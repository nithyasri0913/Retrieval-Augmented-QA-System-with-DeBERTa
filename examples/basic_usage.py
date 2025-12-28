"""
Basic usage example for the Production QA System
"""

import sys
sys.path.append('..')

from src.pipeline import ProductionQASystem


def main():
    # Sample documents (you can replace with your own corpus)
    documents = [
        """
        The transformer architecture was introduced in the paper "Attention is All You Need"
        by Vaswani et al. in 2017. It relies entirely on self-attention mechanisms to compute
        representations of input and output sequences without using recurrent neural networks.
        The architecture has become the foundation for many modern NLP models including BERT,
        GPT, and T5.
        """,
        """
        BERT (Bidirectional Encoder Representations from Transformers) is a pre-training
        approach for NLP introduced by Google in 2018. Unlike traditional language models
        that read text sequentially, BERT reads the entire sequence of words at once,
        allowing it to learn the context of a word based on all of its surroundings.
        """,
        """
        DeBERTa (Decoding-enhanced BERT with disentangled attention) improves upon BERT
        using two novel techniques. First, it uses disentangled attention where each word
        is represented using two vectors encoding content and position. Second, it uses
        an enhanced mask decoder that incorporates absolute positions in the decoding layer.
        DeBERTa was developed by Microsoft in 2020.
        """,
        """
        GPT (Generative Pre-trained Transformer) is a series of language models developed
        by OpenAI. Unlike BERT which is bidirectional, GPT uses a unidirectional approach,
        predicting the next token based only on previous tokens. GPT-3, released in 2020,
        has 175 billion parameters and demonstrated impressive few-shot learning capabilities.
        """,
        """
        FAISS (Facebook AI Similarity Search) is a library developed by Facebook AI Research
        for efficient similarity search and clustering of dense vectors. It contains algorithms
        that search in sets of vectors of any size, up to ones that possibly do not fit in RAM.
        FAISS is widely used in retrieval-augmented generation (RAG) systems.
        """
    ]

    # Document metadata (optional)
    metadata = [
        {'title': 'Transformers', 'year': 2017, 'topic': 'Architecture'},
        {'title': 'BERT', 'year': 2018, 'topic': 'Pre-training'},
        {'title': 'DeBERTa', 'year': 2020, 'topic': 'Improved BERT'},
        {'title': 'GPT', 'year': 2020, 'topic': 'Generative Models'},
        {'title': 'FAISS', 'year': 2019, 'topic': 'Vector Search'}
    ]

    # Initialize the QA system
    print("Initializing QA System...")
    print("=" * 70)

    qa_system = ProductionQASystem(
        retriever_model='all-MiniLM-L6-v2',  # Fast and efficient
        qa_model='deepset/deberta-v3-base-squad2',  # Accurate QA model
        top_k=3,  # Retrieve top 3 passages
        qa_threshold=0.0,  # No threshold - accept all answers
        chunk_size=512,  # Chunk size in words
        chunk_overlap=50  # Overlap between chunks
    )

    # Index the corpus
    print("\nIndexing documents...")
    qa_system.index_corpus(
        documents=documents,
        metadata=metadata,
        use_chunking=True
    )

    # Ask questions
    print("\n" + "=" * 70)
    print("QUESTION ANSWERING")
    print("=" * 70)

    questions = [
        "Who developed DeBERTa?",
        "When was the transformer architecture introduced?",
        "What is FAISS used for?",
        "How many parameters does GPT-3 have?",
        "What makes BERT different from traditional language models?"
    ]

    for question in questions:
        result = qa_system.answer_question(
            question,
            return_all_candidates=False,
            verbose=True
        )

    # Batch answering
    print("\n" + "=" * 70)
    print("BATCH ANSWERING")
    print("=" * 70)

    batch_questions = [
        "What year was BERT released?",
        "What are the two techniques used in DeBERTa?"
    ]

    batch_results = qa_system.batch_answer(batch_questions, verbose=True)

    # Search without QA
    print("\n" + "=" * 70)
    print("DOCUMENT SEARCH (without QA)")
    print("=" * 70)

    search_query = "language models"
    print(f"\nSearch query: {search_query}\n")

    search_results = qa_system.search(search_query, k=3)
    for i, result in enumerate(search_results, 1):
        print(f"{i}. {result['title']} (Relevance: {result['relevance_score']:.3f})")
        print(f"   {result['text'][:100]}...\n")

    # Get system statistics
    print("\n" + "=" * 70)
    print("SYSTEM STATISTICS")
    print("=" * 70)
    stats = qa_system.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")

    # Save the system (optional)
    # qa_system.save('./saved_model')

    # Load the system (optional)
    # loaded_system = ProductionQASystem.load('./saved_model')


if __name__ == "__main__":
    main()
