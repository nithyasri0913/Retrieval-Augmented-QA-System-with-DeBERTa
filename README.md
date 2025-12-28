# RAG QA System with Dense Retrieval and DeBERTa

A production-ready Retrieval-Augmented Generation (RAG) system combining dense retrieval (Sentence Transformers + FAISS) with extractive question answering (DeBERTa) for accurate, context-aware responses.

## Features

- **Dense Retrieval**: Fast and accurate document retrieval using Sentence Transformers and FAISS
- **Extractive QA**: Precise answer extraction using DeBERTa (state-of-the-art QA model)
- **LangChain Integration**: Seamless integration with LangChain for advanced workflows
- **Production Ready**: Optimized for real-world deployment with batching, caching, and persistence
- **Flexible**: Easy to customize models, chunk sizes, and confidence thresholds
- **Scalable**: FAISS-powered indexing handles millions of documents efficiently

## System Architecture

```
Query → Sentence Transformer → FAISS Index → Top-K Passages → DeBERTa QA → Answer
```

### Components

1. **Dense Retrieval** ([src/retriever.py](src/retriever.py))
   - Sentence Transformers for encoding documents/queries
   - FAISS for efficient similarity search
   - Smart document chunking with overlap

2. **Extractive QA** ([src/qa_model.py](src/qa_model.py))
   - DeBERTa for precise answer extraction
   - Confidence thresholding and answer validation
   - Support for "no answer" detection

3. **LangChain Integration** ([src/langchain_integration.py](src/langchain_integration.py))
   - Custom retriever for LangChain workflows
   - Score combination (retrieval + QA)
   - Answer reranking

4. **Production Pipeline** ([src/pipeline.py](src/pipeline.py))
   - Complete end-to-end system
   - Save/load functionality
   - Batch processing and statistics

## Installation

### Prerequisites
- Python 3.8+
- CUDA (optional, for GPU acceleration)

### Setup

```bash
# Clone the repository
cd DeBERTa

# Install dependencies
pip install -r requirements.txt

# Optional: Install with GPU support
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install faiss-gpu
```

## Quick Start

```python
from src.pipeline import ProductionQASystem

# Initialize the system
qa_system = ProductionQASystem(
    retriever_model='all-MiniLM-L6-v2',
    qa_model='deepset/deberta-v3-base-squad2',
    top_k=3,
    qa_threshold=0.3
)

# Index your documents
documents = [
    "Python is a high-level programming language...",
    "Machine learning is a subset of AI...",
    # ... more documents
]

qa_system.index_corpus(documents)

# Ask questions
result = qa_system.answer_question("What is Python?")
print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']:.3f}")
```

## Usage Examples

### Basic Usage
See [examples/basic_usage.py](examples/basic_usage.py) for a complete example with:
- Document indexing
- Single and batch question answering
- Document search
- System statistics

```bash
cd examples
python basic_usage.py
```

### Custom Corpus
See [examples/custom_corpus.py](examples/custom_corpus.py) for:
- Loading documents from files
- Interactive Q&A session
- Saving and loading systems

```bash
cd examples
python custom_corpus.py
```

### SQuAD Evaluation
See [examples/squad_evaluation.py](examples/squad_evaluation.py) for:
- Evaluating on SQuAD dataset
- Calculating exact match accuracy
- Comparing with gold answers

```bash
cd examples
python squad_evaluation.py
```

## Important files:

- run_interactive.py - Main interactive script
- examples/custom_corpus.py - Alternative interactive script
- examples/basic_usage.py - Full demo
- examples/squad_evaluation.py - Benchmark
- test_system.py - System verification


## Advanced Configuration

### Model Selection

**Retriever Models** (trade-off: speed vs accuracy)
- `all-MiniLM-L6-v2`: Fast, good quality (default)
- `all-mpnet-base-v2`: Better accuracy, slower
- `multi-qa-mpnet-base-dot-v1`: Optimized for Q&A

**QA Models**
- `deepset/deberta-v3-base-squad2`: Good balance (default)
- `deepset/deberta-v3-large-squad2`: Higher accuracy, slower
- `microsoft/deberta-v3-base`: Original Microsoft version

### Chunking Strategy

```python
qa_system = ProductionQASystem(
    chunk_size=512,      # Tokens per chunk
    chunk_overlap=50,    # Overlap between chunks
)
```

### Confidence Thresholds

```python
qa_system = ProductionQASystem(
    qa_threshold=0.3,    # Minimum QA confidence (0-1)
    top_k=5,             # Number of passages to retrieve
)
```

### Save and Load

```python
# Save the system
qa_system.save('./my_qa_system')

# Load later
loaded_system = ProductionQASystem.load('./my_qa_system')
```

## Performance Optimization

### GPU Acceleration
```python
# Use GPU for faster inference
qa_system = ProductionQASystem(device='cuda')
```

### Batch Processing
```python
questions = ["Question 1?", "Question 2?", "Question 3?"]
results = qa_system.batch_answer(questions)
```

### FAISS Index Types

For large-scale deployments, consider switching to approximate search:

```python
# In src/retriever.py, replace IndexFlatL2 with:
import faiss

# For faster search on large datasets
nlist = 100  # number of clusters
quantizer = faiss.IndexFlatL2(dimension)
index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
```

## Evaluation Metrics

### Retrieval Metrics
- Recall@k: Percentage of relevant docs in top-k
- MRR (Mean Reciprocal Rank): Position of first relevant result

### QA Metrics
- Exact Match: Exact string match with gold answer
- F1 Score: Token-level overlap with gold answer
- Confidence Distribution: QA model confidence scores

## Production Considerations

### Scaling Strategies
1. **Document Sharding**: Split large corpora across multiple indices
2. **FAISS GPU Indices**: Use GPU for faster similarity search
3. **Model Quantization**: Reduce model size for faster inference
4. **Microservices**: Deploy retriever and QA as separate services

### Error Handling
- No answer found → Returns context with confidence score
- Context too long → Automatic truncation with doc_stride
- Low confidence → Option to return multiple candidates

### Monitoring
```python
# Get system statistics
stats = qa_system.get_stats()
print(stats)

# Track query times
result = qa_system.answer_question(question)
print(f"Query time: {result['query_time']:.3f}s")
```

## Project Structure

```
DeBERTa/
├── src/
│   ├── __init__.py              # Package initialization
│   ├── retriever.py             # Dense retrieval with FAISS
│   ├── qa_model.py              # DeBERTa QA model wrapper
│   ├── langchain_integration.py # LangChain components
│   └── pipeline.py              # Production pipeline
├── examples/
│   ├── basic_usage.py           # Basic example
│   ├── custom_corpus.py         # Custom documents
│   ├── squad_evaluation.py      # SQuAD evaluation
│   └── README.md                # Examples documentation
├── requirements.txt             # Dependencies
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

## API Reference

### ProductionQASystem

Main class for the QA system.

**Methods:**
- `index_corpus(documents, metadata, use_chunking)`: Index documents
- `answer_question(question, return_all_candidates, verbose)`: Answer a question
- `batch_answer(questions, verbose)`: Answer multiple questions
- `search(query, k)`: Search without QA
- `save(directory)`: Save system to disk
- `load(directory)`: Load system from disk (class method)
- `get_stats()`: Get system statistics

### DocumentRetriever

Dense retrieval using Sentence Transformers and FAISS.

**Methods:**
- `chunk_documents(texts, chunk_size, overlap, metadata)`: Chunk documents
- `build_index(documents, metadata)`: Build FAISS index
- `retrieve(query, k)`: Retrieve top-k documents
- `save(directory)`: Save retriever
- `load(directory)`: Load retriever (class method)

### QAModel

Extractive QA using DeBERTa.

**Methods:**
- `extract_answer(question, context, threshold, top_k)`: Extract answer
- `batch_extract_answers(question, contexts, threshold)`: Batch extraction
- `get_answer_with_context(question, context, threshold, context_window)`: Answer with extended context
- `validate_answer(question, answer, context)`: Validate answer

## Troubleshooting

### Common Issues

**1. Out of Memory**
- Reduce `chunk_size` or `top_k`
- Use smaller models
- Enable GPU memory optimization

**2. Slow Indexing**
- Reduce batch size in encoding
- Use simpler retriever model
- Enable GPU acceleration

**3. Low Accuracy**
- Increase `top_k` to retrieve more passages
- Lower `qa_threshold`
- Use larger/better models
- Improve document chunking

## Contributing

Contributions are welcome! Areas for improvement:
- Additional retrieval models
- Better chunking strategies
- Reranking mechanisms
- Evaluation scripts
- Documentation

## License

MIT License

## Citation

If you use this system in your research, please cite:

```bibtex
@software{rag_qa_system,
  title = {RAG QA System with Dense Retrieval and DeBERTa},
  year = {2024},
  author = {Your Name},
  url = {https://github.com/yourusername/DeBERTa}
}
```

## Acknowledgments

- [Sentence Transformers](https://www.sbert.net/) for dense retrieval
- [FAISS](https://github.com/facebookresearch/faiss) for similarity search
- [Hugging Face](https://huggingface.co/) for DeBERTa models
- [LangChain](https://github.com/langchain-ai/langchain) for RAG orchestration

## Next Steps

1. **Start Small**: Test on SQuAD dataset first
2. **Build Iteratively**: Retriever → QA → Integration
3. **Evaluate Continuously**: Track metrics at each stage
4. **Optimize**: Profile bottlenecks and improve

For detailed examples, see the [examples/](examples/) directory.
