import requests
import json
import tqdm
import time
import os

import fitz  # PyMuPDF
import re
import pandas as pd

def chemrxiv_fetch_all_papers_from_2024(output_file="chemrxiv_data.json",total=50):
    """
    Fetch all papers from ChemRxiv starting from 2024 and save them to a JSON file.

    Args:
        output_file (str): The file to save the data to.

    Returns:
        None
    """
    url = "https://chemrxiv.org/engage/chemrxiv/public-api/v1/items"

    params = {
        "limit": 50,                # Max results per page
        "sort": "VIEWS_COUNT_DESC", # Sort by most viewed papers
        "searchDateFrom": "2023-06-01", # Filter papers from 2024 onwards
        # "searchDateTo": "2023-12-01"
    }

    all_papers = []
    total_fetched = 0

    while True:
        try:
            print(f"Total fetched: {total_fetched}")
            response = requests.get(url, params=params)
            response.raise_for_status()

            data = response.json()  # Parse the JSON response
            papers = data.get("itemHits", [])

            if not papers:
                print("No more papers found.")
                break

            all_papers.extend(papers)
            total_fetched += len(papers)
            print(f"Fetched {len(papers)} papers. Total so far: {total_fetched}")

            # Check if there are more pages
            if total_fetched >= total or len(papers)==0:
                break

            # Update parameters for the next page (if pagination uses offset)
            params["skip"] = total_fetched

        except requests.exceptions.RequestException as e:
            print(f"Error fetching data from ChemRxiv API: {e}")
            break

    # Save all fetched data to a JSON file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_papers, f, indent=4)

    print(f"All data saved to {output_file}. Total papers: {total_fetched}")
    return all_papers

def chemrxiv_download_paper(paper,output_folder):
    time.sleep(0.2)
    # Get the paper title and URL
    paper_title = paper.get("item", {}).get("title", "unknown_title").replace("/", "-").replace(" ", "_")  # Sanitize filename
    paper_url = paper.get("item", {}).get("asset", {}).get("original", {}).get("url")
    paper_license = paper.get("item", {}).get("license", {}).get("name", "")
    
    if not paper_url:
        print(f"Skipping paper '{paper_title}' (No URL found)")
    elif ('ND' not in paper_license) and paper_license:
        # Determine the output file path
        file_extension = paper_url.split(".")[-1]  # Get file extension from URL
        file_name = f"{paper_title}.{file_extension}"
        file_path = os.path.join(output_folder, file_name)
        
        # Download the paper
        response = requests.get(paper_url, stream=True)
        response.raise_for_status()
        
        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)

def chemrxiv_download_papers(all_papers, output_folder="chemrxiv_papers"):
    """
    Download papers from the given list and save them to the specified output folder.

    Args:
        all_papers (list): List of papers with metadata including the download URL.
        output_folder (str): Folder to save the downloaded papers.

    Returns:
        None
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for paper in tqdm.tqdm(all_papers):
        for try_n in range(3):
            try:
                chemrxiv_download_paper(paper,output_folder)
            except Exception as e:
                print(f"Failed to download {try_n}. Let's go to sleep for 5 seconds")
                time.sleep(5)




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

def process_pdfs_to_dataframe(directory, output_file="chemrxiv_data.csv"):
    """Processes all PDFs in a given directory, extracts, cleans text, and stores in a DataFrame."""
    pdf_data = []
    
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(directory, filename)
            print(f"Processing: {filename}")
            raw_text = extract_text_from_pdf(pdf_path)
            cleaned_text = clean_text(raw_text)
            pdf_data.append({"filename": filename, "text": cleaned_text})
    
    df = pd.DataFrame(pdf_data)
    if output_file:
        df.to_csv(output_file)
    return df


# Example usage
if __name__=='__main__':

    all_papers=chemrxiv_fetch_all_papers_from_2024(output_file="../data/chemrxiv_data_2024.json")
    with open('../data/chemrxiv_data_2024.json','rb') as f:
        all_papers=json.load(f)
    chemrxiv_download_papers(all_papers,)
    process_pdfs_to_dataframe('./chemrxiv_papers')