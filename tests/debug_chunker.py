from src.corpus.text_chunker import TextChunker
from unittest.mock import Mock

# Create a mock word validator that considers all words valid by default
word_validator = Mock()
word_validator.is_valid.return_value = True

# Create a text chunker with standard settings  
chunker = TextChunker(
    word_validator=word_validator,
    chunk_size=5,
    overlap=2
)

# Test standard chunking
text = "This is a test of the chunking system"
chunks = chunker.chunk_text(text)
print("Standard chunking:")
for i, chunk in enumerate(chunks):
    print(f"Chunk {i}: '{chunk}'")

# Test with invalid words
word_validator.is_valid.side_effect = lambda word: word != "chunking"
chunks = chunker.chunk_text(text)
print("\nChunking with 'chunking' filtered out:")
for i, chunk in enumerate(chunks):
    print(f"Chunk {i}: '{chunk}'")