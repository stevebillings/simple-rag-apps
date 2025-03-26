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
python tools/populate_pinecone_for_boat_manuals.py  # Populate vector DB
python tools/chatbot_for_boat_manuals.py            # Run boat manual chatbot
python tools/populate_pinecone_for_faq.py           # Populate FAQ vector DB
python tools/chatbot_for_faq.py                     # Run FAQ chatbot
```

## Code Style Guidelines
- **Types**: Use Python type annotations throughout
- **Imports**: Group standard library, third-party, then local imports
- **Naming**: snake_case for variables/functions, PascalCase for classes
- **Classes**: Use abstract classes for interfaces/configuration
- **Error Handling**: Use specific exceptions with informative messages
- **Type Checking**: Run `mypy` for static type checking

This RAG application uses OpenAI for embeddings/LLM and Pinecone for vector storage.