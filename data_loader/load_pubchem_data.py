import tqdm
import os
import dotenv
import json
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from utils.pubchem import download_and_store_pubchem
from utils.wikipedia import add_wiki_data
from utils.keyword_extractor import  keyword_document_mapping_old
from utils.pubchem_questions import pubchem_generate_2_hop_questions
import time
# Function to process a single row


# Main script
if __name__ == "__main__":
    # STAGE #1
    # download_and_store_pubchem('data/pubchem_dump.csv')

    #STAGE #2
    # data = pd.read_csv('data/pubchem_dump.csv')
    # data = add_wiki_data(data)
    # data.to_csv('data/pubchem_dump_with_wiki_text.csv', index=False)

    # STAGE #3
    # data=pd.read_csv('data/pubchem_dump_with_wiki_text.csv').iloc[:10000] # For now
    # data['name']=data['name'].fillna('nameless!!')
    # keyword_to_documents =   keyword_document_mapping_old(list(data.combined_text),list(data.name)) # keyword_document_mapping(list(data.combined_text),list(data.name),8)
    # with open('data/keywords.json', 'w', encoding='utf-8') as f:
    #     json.dump(keyword_to_documents, f, indent=4)
    # STAGE #4
    dotenv.load_dotenv()
    data=pd.read_csv('data/pubchem_dump_with_wiki_text.csv').iloc[:10000] # For now
    with open('data/keywords.json', 'r', encoding='utf-8') as f:
        keyword_to_documents = json.load(f)
    qas=pubchem_generate_2_hop_questions(data,keyword_to_documents,api_key=os.environ.get("OPENAI_API_KEY"))
    with open('data/qas.json', 'w', encoding='utf-8') as f:
        json.dump(qas, f, indent=4)

    
