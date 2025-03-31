# Simple RAG Applications

This repo is a sandbox I'm using to learn about developing LLM-based applications.

It is a (simple) AI chatbot creation platform that enables you to quickly configure new Retrieval-Augmented Generation (RAG) LLM chatbots, without writing code. The resulting chatbot is a command line tool (a python script).

The following corpus formats are currently supported:

1. A set of one or more PDFs in a directory.
2. An FAQ (question:answer pairs) in .json format.

Included in this source repo are configrations for two use cases:

1. **FAQ Chatbot**: Answers questions using FAQ data read from a json file
2. **Boat Manual Chatbot**: Answers questions using information extracted from boat owner manuals

Configuring a new chatbot involves creating a new .json config file in the config directory (by default: resources/config), and provide the corpus file(s) that the config points to. Information on the config file format appears below.

## Project Structure

- **src/**: Source code directory
  - **config/**: Configuration class
  - **corpus/**: Code for handling (reading/parsing, cleaning, chunking) corpus content
  - **llm/**: OpenAI API client for embeddings and completions
  - **vector_db/**: Pinecone vector database integration
  - **tools/**: Executable scripts for chatbots and data population
- **tests/**: Test directory mirroring src structure
  - Unit tests using pytest for core functionality
- **resources/**:
  - **config**: Config files for different chatbots
  - **boat_manuals**: PDFs for the boat manual chatbot
  - **faq**: faq.json for the FAQ chatbot

## Technologies

- Python 3.10+
- OpenAI API (embeddings: text-embedding-3-small, LLM: GPT-4o)
- Pinecone vector database
- LangChain for document loading and text chunking
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

# Create virtual environment
python -m venv .venv-srag

# Set PYTHONPATH and activate virtual environment
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
python src/tools/populate_pinecone.py --config-name faq
```

You can override the default config file location (resources/config) using the --config-dir command line argument.

### Run Chatbots

The application uses a unified chatbot tool with configuration options:

```bash
# For boat manual chatbot (default)
python src/tools/chatbot.py

# For FAQ chatbot
python src/tools/chatbot.py --config-name faq
```

You can override the default config file location (resources/config) using the --config-dir command line argument.

Enter questions when prompted, type 'exit' to quit.

### Configuration

The application uses JSON configuration files located in `resources/config/`:

- `boat_manuals_config.json` - Configuration for the boat manuals chatbot
- `faq_config.json` - Configuration for the FAQ chatbot

Each configuration specifies:
- Corpus type (pdfs or faq)
- Bot prompt
- Vector database index name (Pinecone index name)
- Vector database namespace (Pinecone namespace)
- Corpus directory path
- System prompt template (be sure to include the "{}"; this is where the relevant content from your corpus will be inserted)

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
