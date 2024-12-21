import tqdm
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from utils.pubchem import download_and_store_pubchem
from utils.wikipedia import wiki_fetch_combined_text, wiki_exists
import time
# Function to process a single row
def process_row(row):
    for i in range(5):
        try:
            name = row['name']
            if wiki_exists(name):
                combined_text = wiki_fetch_combined_text(name)
            else:
                combined_text = ''
            return combined_text
        except Exception as e:
            print(f'Error {e}. Lets sleep for {i}th time')
            time.sleep(5)
    combined_text = ''
    return combined_text

# Main script
if __name__ == "__main__":
    # # download_and_store_pubchem('data/pubchem_dump.csv')
    data = pd.read_csv('data/pubchem_dump.csv')
    # Use ThreadPoolExecutor for multithreading
    with ThreadPoolExecutor() as executor:
        # Submit tasks to the executor
        results = list(tqdm.tqdm(executor.map(process_row, data.to_dict('records')), total=len(data)))

    # Add the fetched text back to the DataFrame
    data['wiki_text'] = results
    data['combined_text'] = "wikipedia: " + data['wiki_text'] + '\n pubchem:' + data['text']

    # Save the DataFrame to a new CSV file
    data.to_csv('data/pubchem_dump_with_wiki_text.csv', index=False)
    # print(wiki_exists('Fenpropathrin'))
    # print(wiki_fetch_combined_text('Fenpropathrin'))