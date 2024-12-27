import tqdm
import json
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from utils.pubchem import download_and_store_pubchem
from utils.wikipedia import wiki_fetch_combined_text, wiki_exists
from utils.keyword_extractor import keyword_document_mapping
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
    # STAGE #1
    # download_and_store_pubchem('data/pubchem_dump.csv')

    #STAGE #2
    # data = pd.read_csv('data/pubchem_dump.csv')
    # with ThreadPoolExecutor() as executor:
    #     results = list(tqdm.tqdm(executor.map(process_row, data.to_dict('records')), total=len(data)))
    # data['wiki_text'] = results
    # data['combined_text'] = "wikipedia: " + data['wiki_text'].fillna('') + '\n pubchem:' + data['text'].fillna('')+ '\n'+data['properties']
    # data.to_csv('data/pubchem_dump_with_wiki_text.csv', index=False)

    # STAGE #3
    data=pd.read_csv('data/pubchem_dump_with_wiki_text.csv')
    data['name']=data['name'].fillna('nameless!!')
    keyword_to_documents = keyword_document_mapping(list(data.combined_text),list(data.name))
    with open('data/keywords.json', 'w', encoding='utf-8') as f:
        json.dump(keyword_to_documents, f, indent=4)

    
