import os
import sys
from typing import Any, Dict, List, Optional

from frequently_asked_questions import FrequentlyAskedQuestions
from openai_client import OpenAiClient
from pinecone_client_faq import PineconeClientFaq


faq: FrequentlyAskedQuestions = FrequentlyAskedQuestions()
openai_client: OpenAiClient = OpenAiClient()
pinecone_client = PineconeClientFaq(faq=faq, openai_client=openai_client)
pinecone_client.populate_vector_database()
