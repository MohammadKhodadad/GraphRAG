import fitz  # PyMuPDF
import re
import pandas as pd
import os
import openai
from openai import OpenAI
import json
import tqdm


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