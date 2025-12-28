# Examples

This directory contains examples demonstrating how to use the RAG QA System.

## Files

### 1. basic_usage.py
Basic example showing the core functionality:
- Initializing the QA system
- Indexing documents
- Asking questions
- Batch question answering
- Document search

**Run:**
```bash
cd examples
python basic_usage.py
```

### 2. squad_evaluation.py
Evaluate the system on the SQuAD dataset:
- Load SQuAD dataset
- Evaluate exact match accuracy
- Compare with gold answers

**Run:**
```bash
cd examples
python squad_evaluation.py
```

### 3. custom_corpus.py
Build a QA system with your own documents:
- Load documents from files
- Interactive Q&A session
- Save and load trained systems

**Run:**
```bash
cd examples
python custom_corpus.py
```

## Usage Patterns

### Quick Start
```python
from src.pipeline import ProductionQASystem

# Initialize
qa_system = ProductionQASystem()

# Index documents
qa_system.index_corpus(documents)

# Ask questions
result = qa_system.answer_question("Your question?")
print(result['answer'])
```

### Advanced Configuration
```python
qa_system = ProductionQASystem(
    retriever_model='all-mpnet-base-v2',  # More accurate
    qa_model='deepset/deberta-v3-large-squad2',  # Larger model
    top_k=5,
    qa_threshold=0.3,
    chunk_size=512,
    chunk_overlap=50
)
```

### Save and Load
```python
# Save
qa_system.save('./my_qa_system')

# Load
loaded_system = ProductionQASystem.load('./my_qa_system')
```

## Next Steps

1. Try with your own documents
2. Adjust parameters for your use case
3. Experiment with different models
4. Evaluate on your specific domain
