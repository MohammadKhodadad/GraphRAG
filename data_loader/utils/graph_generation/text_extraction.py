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



def extract_introduction_with_limit(text, char_limit=3000):
    # Define section headers
    introduction_pattern = r"(?i)\bIntroduction\b"
    next_section_pattern = r"(?i)\b(?:Methods|Materials|Experiments|Related Work|Background|Results|Discussion|Conclusion)\b"

    # Locate "Introduction" section
    intro_match = re.search(introduction_pattern, text)
    if not intro_match:
        return "Introduction section not found."

    start_idx = intro_match.end()  # Start extracting after "Introduction"

    # Locate the next major section
    next_section_match = re.search(next_section_pattern, text[start_idx:])
    # end_idx = start_idx + next_section_match.start() if next_section_match else len(text)
    end_idx=min(start_idx+8000,len(text))

    # Extract introduction text
    introduction_text = text[start_idx:end_idx].strip()

    # Split text into paragraphs based on '\n\n' (double newline)
    paragraphs = re.split(r"\n\s*\n", introduction_text)

    # Accumulate paragraphs until reaching 3000 characters
    extracted_text = ""
    current_length = 0

    for idx, paragraph in enumerate(paragraphs):
        if current_length + len(paragraph) > char_limit and idx>0:
            break
        extracted_text += paragraph + "\n\n"
        current_length += len(paragraph)

    return extracted_text.strip()


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

if __name__=='__main__':
    address= '/media/torontoai/GraphRAG/GraphRAG/data_loader/data/chemrxiv_papers/AIMNet2:_A_Neural_Network_Potential_to_Meet_your_Neutral,_Charged,_Organic,_and_Elemental-Organic_Needs.pdf'
    cleaned_text= extract_introduction_with_limit(clean_text(extract_text_from_pdf(address)),2000)
    print(cleaned_text)