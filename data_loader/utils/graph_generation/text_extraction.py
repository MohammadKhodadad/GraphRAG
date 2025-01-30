import fitz  # PyMuPDF
import re
import pandas as pd
import os
import openai
from openai import OpenAI
import json
import tqdm
import nltk

nltk.download("punkt_tab")
from nltk.tokenize import sent_tokenize

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file using PyMuPDF."""
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text("text") + "\n"
    return text

def clean_text(text):
    """Cleans the extracted text while preserving single and double newlines."""
    text = re.sub(r'[^\S\r\n]+', ' ', text)  # Replace multiple spaces and tabs with a single space, keep \n
    text = re.sub(r'[^a-zA-Z0-9.,;!?()\s\n]', '', text)  # Keep alphanumeric, punctuation, and whitespace (\n included)
    return text.strip()


def split_text(text, max_words=200):
    """Splits text into chunks by grouping sentences while keeping each chunk under `max_words`."""
    sentences = sent_tokenize(text)  # Tokenize into sentences
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence.split())

        if current_length + sentence_length > max_words:
            # Save the current chunk and start a new one
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            # Add the sentence to the current chunk
            current_chunk.append(sentence)
            current_length += sentence_length

    # Add any remaining text as the last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks