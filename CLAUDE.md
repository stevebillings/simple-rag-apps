# Simple RAG Apps - Development Guide

## Setup & Run Commands
```bash
# Setup environment
python -m venv .venv-srag
source .venv-srag/bin/activate
pip install -r requirements.txt
source sourceme
export OPENAI_API_KEY=your_openai_api_key
export PINECONE_API_KEY=your_pinecone_api_key

# Run applications
python src/tools/populate_pinecone.py                   # Populate boat manuals vector DB (default)
python src/tools/populate_pinecone.py --config faq      # Populate FAQ vector DB
python src/tools/chatbot.py                             # Run boat manual chatbot (default)
python src/tools/chatbot.py --config faq                # Run FAQ chatbot

# Run tests
python -m pytest                               # Run all tests
python -m pytest tests/corpus/                 # Run tests for the corpus module
python -m pytest -v                            # Run tests with verbose output
```

## Code Style Guidelines
- **Types**: Use Python type annotations throughout
- **Imports**: Group standard library, third-party, then local imports
- **Naming**: snake_case for variables/functions, PascalCase for classes
- **Classes**: Use abstract classes for interfaces/configuration
- **Error Handling**: Use specific exceptions with informative messages
- **Type Checking**: Run `mypy` for static type checking

## Testing Guidelines
- Mirror the `src` directory structure in the `tests` directory
- Use pytest for all tests
- Write tests for "easy to test" modules that require minimal mocking
- Follow the Arrange-Act-Assert pattern
- Use pytest fixtures for common test setup
- Use mock objects when appropriate to avoid external dependencies

This RAG application uses OpenAI for embeddings/LLM and Pinecone for vector storage.