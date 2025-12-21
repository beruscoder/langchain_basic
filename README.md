# LangChain Basic - RAG Architecture

A foundational implementation of a Retrieval-Augmented Generation (RAG) system using LangChain. This project demonstrates how to build a complete RAG pipeline that loads PDF documents, chunks them, creates embeddings, retrieves relevant context, and generates answers using an LLM.

## Features

- **PDF Document Loading**: Load and process PDF documents
- **Text Splitting**: Intelligent text chunking for optimal retrieval
- **Vector Store Management**: Build and manage FAISS-based vector databases
- **Semantic Retrieval**: Retrieve relevant documents based on semantic similarity
- **LLM Integration**: Generate answers using language models
- **Streaming Support**: Stream model responses for real-time output
- **Chat Engine**: Interactive chat interface for Q&A

## Project Structure

```
.
├── main.py                 # Main entry point for RAG pipeline
├── pdf_loader.py          # PDF document loading and processing
├── textsplitter.py        # Text splitting and vector store management
├── retriever.py           # Document retrieval engine
├── llm_loader.py          # LLM initialization and management
├── rag.py                 # Core RAG engine
├── chat_engine.py         # Chat interface
├── prompts.py             # Prompt templates and management
├── embedding/             # Embedding models directory
├── embeddings/            # Pre-computed embeddings
├── faiss_index_/          # FAISS vector store index
├── new.ipynb              # Jupyter notebook for experimentation
└── README.md              # This file
```

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup

1. Clone or download the project:
```bash
cd langchain-basic
```

2. Install required dependencies:
```bash
pip install langchain openai faiss-cpu pdf2image PyPDF2 pydantic
```

3. Set up environment variables:
```bash
# Add your OpenAI API key
export OPENAI_API_KEY='your-api-key-here'
```

## Usage

### Quick Start

1. **Place your PDF file** in the project directory (or update the filename in `main.py`)

2. **Run the RAG pipeline**:
```bash
python main.py
```

3. **Enter your query** when prompted:
```
Ask: What is the main topic discussed in the document?
```

The system will:
- Rewrite your query for better retrieval
- Retrieve relevant document chunks
- Generate an answer with streaming output
- Display the final answer

### Module Overview

#### `pdf_loader.py`
Handles PDF document loading and processing.

#### `textsplitter.py`
Splits documents into chunks and manages the FAISS vector store.

#### `retriever.py`
Retrieves relevant documents based on semantic similarity to queries.

#### `llm_loader.py`
Initializes and manages language model instances.

#### `rag.py`
Core RAG engine that orchestrates:
- Query rewriting
- Document retrieval
- Answer generation (with streaming support)

#### `prompts.py`
Contains prompt templates for:
- Query rewriting
- Answer generation

#### `chat_engine.py`
Interactive chat interface for multi-turn conversations.

## How It Works

### The RAG Pipeline

1. **Load**: Documents are loaded from PDF files
2. **Split**: Documents are split into manageable chunks
3. **Embed**: Chunks are converted to vector embeddings
4. **Index**: Embeddings are stored in a FAISS vector database
5. **Retrieve**: For a given query, relevant chunks are retrieved
6. **Generate**: An LLM generates an answer using the retrieved context

### Query Processing

```
User Query
    ↓
Query Rewriting (via LLM)
    ↓
Vector Similarity Search
    ↓
Document Retrieval
    ↓
Context + Query → LLM
    ↓
Generated Answer
```

## Configuration

### Vector Store Settings
- **Embedding Model**: Default OpenAI embeddings
- **Vector Database**: FAISS
- **Index Location**: `faiss_index_/`

### LLM Settings
- Update `llm_loader.py` to use different models
- Modify temperature, max_tokens, and other parameters as needed

### Prompt Templates
Customize prompts in `prompts.py` for different use cases.

## API Keys

You'll need API keys for:
- **OpenAI**: For embeddings and LLM inference
- Store in environment variables or `.env` file

## Advanced Usage

### Custom Document Processing
Modify `pdf_loader.py` to handle different file formats or add preprocessing steps.

### Fine-tuning Retrieval
Adjust chunk sizes and overlap in `textsplitter.py` for different document types.

### Chat Interface
Use `chat_engine.py` for interactive multi-turn conversations with memory.

## Performance Tips

1. **Chunk Size**: Balance between context preservation and retrieval precision
   - Smaller chunks: Better precision, less context
   - Larger chunks: More context, potential noise

2. **Embedding Model**: Consider using smaller models for faster inference

3. **Vector Store**: FAISS is fast for CPU; consider GPU acceleration for large indexes

4. **Retrieval**: Adjust the number of retrieved documents (k parameter)

## Troubleshooting

- **API Key Errors**: Ensure `OPENAI_API_KEY` is set correctly
- **PDF Loading Issues**: Check PDF format and file path
- **Memory Issues**: Reduce chunk size or use smaller models
- **Slow Retrieval**: Reduce the number of retrieved documents

## Future Enhancements

- [ ] Support for more document formats (DOCX, TXT, etc.)
- [ ] Multi-document cross-references
- [ ] Caching and memory optimization
- [ ] Batch query processing
- [ ] Web interface
- [ ] Custom embedding models
- [ ] Query expansion techniques
- [ ] Fallback generation strategies

## License

This project is open source and available for educational purposes.

## References

- [LangChain Documentation](https://python.langchain.com/)
- [FAISS Documentation](https://faiss.ai/)
- [OpenAI API](https://openai.com/api/)

---

Happy exploring with RAG! 🚀
