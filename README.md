# Simple RAG Applications

Python implementations of Retrieval-Augmented Generation (RAG) chatbots for two use cases:

1. **FAQ Chatbot**: Answers questions using predefined FAQ data
2. **Boat Manual Chatbot**: Answers questions using information extracted from boat owner manuals

## Project Structure

- **src/**: Source code directory
  - **config/**: Configuration classes for different applications
  - **corpus/**: Document handling (PDF parsing, text chunking, cleaning)
  - **llm/**: OpenAI API client for embeddings and completions
  - **vector_db/**: Pinecone vector database integration
  - **tools/**: Executable scripts for chatbots and data population
- **tests/**: Test directory mirroring src structure
  - Unit tests using pytest for core functionality
- **resources/**: Contains boat manual PDFs

## Technologies

- Python 3.10+
- OpenAI API (embeddings: text-embedding-3-small, LLM: GPT-4o)
- Pinecone vector database
- PyPDF2 for document parsing
- pytest for testing

## Setup

### Prerequisites
- Python 3.10+
- Pinecone account and API key
- OpenAI API key

### Installation

```bash
# Clone the repository
git clone <repository-url>

# Create and activate virtual environment
python -m venv .venv-srag
source .venv-srag/bin/activate

# Set PYTHONPATH
source sourceme

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY="your-openai-api-key"
export PINECONE_API_KEY="your-pinecone-api-key"
```

## Usage

### Populate Vector Databases

The application uses a unified tool for populating the Pinecone vector database, with configuration options:

```bash
# For boat manuals (default)
python src/tools/populate_pinecone.py

# For FAQ data
python src/tools/populate_pinecone.py --config faq
```

### Run Chatbots

The application uses a unified chatbot tool with configuration options:

```bash
# For boat manual chatbot (default)
python src/tools/chatbot.py

# For FAQ chatbot
python src/tools/chatbot.py --config faq
```

Enter questions when prompted, type 'exit' to quit.

### Configuration

The application uses JSON configuration files located in `resources/config/`:

- `boat_manuals_config.json` - Configuration for the boat manuals chatbot
- `faq_config.json` - Configuration for the FAQ chatbot

Each configuration specifies:
- Corpus type (pdfs or faq)
- Bot prompt
- Vector database settings
- Corpus directory path
- System prompt template

## Running Tests

Tests can be run using pytest:

```bash
# Run all tests
python -m pytest

# Run tests for a specific module
python -m pytest tests/corpus/test_text_chunker.py

# Run tests with verbose output
python -m pytest -v
```

## Example

```
Enter your question (or type 'exit' to quit):
> How do I maintain the boat hull?
>> [Response based on relevant content from the boat manual]

Enter your question (or type 'exit' to quit):
> exit
```