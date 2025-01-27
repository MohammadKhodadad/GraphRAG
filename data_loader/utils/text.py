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
    """Cleans the extracted text by removing unwanted characters and formatting."""
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces and newlines with a single space
    text = re.sub(r'[^a-zA-Z0-9.,;!?()\s]', '', text)  # Keep only alphanumeric and punctuation
    return text.strip()

def extract_introduction_from_pdf(pdf_path):
    """Extracts only the 'Introduction' section from a PDF file using PyMuPDF."""
    text = ""
    doc = fitz.open(pdf_path)
    
    for page in doc:
        text += page.get_text("text") + "\n"
    
    # Normalize spaces and newlines
    text = re.sub(r'\n+', '\n', text)

    # Identify introduction section using regex
    match = re.search(r"(?i)\bIntroduction\b(.*?)\n(?:[A-Z][a-z]+[ \t]*[A-Z]|\n{2,})", text, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    
    return "Introduction not found."

def extract_factual_statements(text, api_key, model="gpt-4o", max_facts=20): # Needs Work
    client = OpenAI(api_key=api_key)
    prompt = f"""
    Extract factual statements from the following text related to chemistry that are valuable by themselves in the tuple format: [entity1 , relation, entity2]. A factual statement should:
    - Be **verifiable** and **objective**.
    - Avoid opinions, assumptions, or subjective language.
    - Exclude speculative content, author perspectives, or discussions.
    - entity1, relation, and entity2 are strings.
    - entity1 and entity2 are single words or single compound words

    An example would be: [["Sodium chloride", "dissolves in", "water"]]
    If a fact contains multiple clauses, break it down into **separate, standalone statements**.
    Extract **at most {max_facts}** factual statements.

    TEXT:
    \"\"\"{text}\"\"\"

    Provide the output as a python list of tuples containing only the extracted factual statements  without any code formatting, backticks, or markdown.
    """
    response = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=0.2,
                model=model,
                max_tokens=1000
            )
    return json.dumps(eval(response.choices[0].message.content))
    

def process_pdfs_to_dataframe(directory, output_file="chemrxiv_data.csv",api_key=None):
    """Processes all PDFs in a given directory, extracts, cleans text, and stores in a DataFrame."""
    pdf_data = []
    
    for filename in tqdm.tqdm(os.listdir(directory)):
        if filename.endswith(".pdf"):
            try:
                pdf_path = os.path.join(directory, filename)
                # print(f"Processing: {filename}")
                raw_text = extract_text_from_pdf(pdf_path)
                cleaned_text = clean_text(raw_text)
                if api_key:
                    facts = extract_factual_statements(cleaned_text,api_key)
                    # cleaned_introduction = clean_text(introduction)
                else:
                    facts=''
                pdf_data.append({"filename": filename, "text": cleaned_text, "facts":facts})
            except Exception as e:
                print(f'Error with {filename}: {e}')

    
    df = pd.DataFrame(pdf_data)
    if output_file:
        df.to_csv(output_file)
    return df


if __name__=='__main__':
    import dotenv
    dotenv.load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")  # Ensure this is set in your environment
    pdf_path = "/media/torontoai/GraphRAG/GraphRAG/data_loader/data/chemrxiv_papers/A_General_Redox-Neutral_Platform_for_Radical_Cross-Coupling_.pdf"
    # intro_text = extract_introduction_from_pdf(pdf_path)
    # print(intro_text)
    raw_text = extract_text_from_pdf(pdf_path)
    cleaned_text = clean_text(raw_text)
    print(extract_factual_statements(cleaned_text[:10000],api_key))
    