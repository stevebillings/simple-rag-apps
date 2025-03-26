# Simple RAG Applications

Python implementations of Retrieval-Augmented Generation (RAG) chatbots for two use cases:

1. **FAQ Chatbot**: Answers questions using predefined FAQ data
2. **Boat Manual Chatbot**: Answers questions using information extracted from boat owner manuals

## Project Structure

- **config/**: Configuration classes for different applications
- **corpus/**: Document handling (PDF parsing)
- **llm/**: OpenAI API client for embeddings and completions
- **vector_db/**: Pinecone vector database integration
- **tools/**: Executable scripts for chatbots and data population
- **resources/**: Contains boat manual PDFs

## Technologies

- Python 3.10+
- OpenAI API (embeddings: text-embedding-3-small, LLM: GPT-4o)
- Pinecone vector database
- PyPDF2 for document parsing

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

```bash
# For boat manuals
python tools/populate_pinecone_for_boat_manuals.py

# For FAQ data
python tools/populate_pinecone_for_faq.py
```

### Run Chatbots

```bash
# Boat manual chatbot
python tools/chatbot_for_boat_manuals.py

# FAQ chatbot
python tools/chatbot_for_faq.py
```

Enter questions when prompted, type 'exit' to quit.

## Example

```
Enter your question (or type 'exit' to quit):
> How do I maintain the boat hull?
>> [Response based on relevant content from the boat manual]

Enter your question (or type 'exit' to quit):
> exit
```