import requests
import json
import os

def fetch_all_papers_from_2024(output_file="chemrxiv_data.json",total=500):
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
        "searchDateFrom": "2024-01-01" # Filter papers from 2024 onwards
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

# Example usage
if __name__=='__main__':

    all_papers=fetch_all_papers_from_2024(output_file="chemrxiv_data_2024.json")
