import requests
import json
import tqdm
import time
import os

import fitz  # PyMuPDF
import re
import pandas as pd
if __name__=='__main__':
    from graph_generation.text_extraction import extract_text_from_pdf, clean_text, split_text, extract_introduction_with_limit
else:
    from .graph_generation.text_extraction import extract_text_from_pdf, clean_text, split_text, extract_introduction_with_limit

def chemrxiv_fetch_all_papers(output_file="chemrxiv_data.json",total=50, search_data_from="2023-06-01"):
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
        "searchDateFrom": search_data_from, # Filter papers from 2024 onwards
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

def chemrxiv_download_paper(paper,output_folder, name_id=True):
    time.sleep(0.2)
    # Get the paper title and URL
    paper_title = paper.get("item", {}).get("title", "unknown_title").replace("/", "-").replace(" ", "_")  # Sanitize filename
    paper_id = paper.get("item", {}).get("id", None)
    paper_url = paper.get("item", {}).get("asset", {}).get("original", {}).get("url")
    paper_license = paper.get("item", {}).get("license", {}).get("name", "")
    
    if not paper_url:
        print(f"Skipping paper '{paper_title}' (No URL found)")
    elif (not paper_id ) or (not paper_title):
        print('No ID or Title found.')
    elif ('ND' not in paper_license) and paper_license:
        # Determine the output file path
        file_extension = paper_url.split(".")[-1]  # Get file extension from URL
        if name_id:
            file_name = f"{paper_id}.{file_extension}"
        else:
            file_name = f"{paper_title}.{file_extension}"
        file_path = os.path.join(output_folder, file_name)
        if not os.path.exists(file_path):
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


def chemrxiv_embed_and_store(pipeline,data_folder):
    texts = []
    ids = []
    names=os.listdir(data_folder)
    for name in tqdm.tqdm(names):
        address=os.path.join(data_folder,name)
        try:
            if '.pdf' in address:
                text=extract_text_from_pdf(address)
            elif '.jsonl' in address:
                raise Exception('Not Implemented.')
            elif '.csv' in address:
                raise Exception('Not Implemented.')
            else:
                raise Exception('No supported input.')
            cleaned_text= extract_introduction_with_limit(clean_text(text),2000)
            texts.append(cleaned_text)
            ids.append(name)
        except Exception as e:
            print(f'Error: {e}')
            continue

    pipeline.cot.retriever.embed_and_store(texts, ids)

# Example usage
if __name__=='__main__':

    all_papers=chemrxiv_fetch_all_papers(output_file="../data/chemrxiv_data_2024.json")
    with open('../data/chemrxiv_data_2024.json','rb') as f:
        all_papers=json.load(f)
    chemrxiv_download_papers(all_papers,)
    process_pdfs_to_dataframe('./chemrxiv_papers')