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


from utils.graph_generation.pipeline import graph_pipeline, graph_pipeline_from_csv, sample_graph_pipeline
from utils.question_generation.question_generation import generate_questions_from_paths, evaluate_questions

import time
# Function to process a single row


# Main script
if __name__ == "__main__":
    dotenv.load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    # STAGE #1
    # download_and_store_pubchem('data/pubchem_dump.csv')

    # STAGE #2
    # data = pd.read_csv('data/pubchem_dump.csv')
    # data = add_wiki_data(data)
    # data.to_csv('data/pubchem_dump_with_wiki_text.csv', index=False)

    # STAGE #3
    data=pd.read_csv('data/pubchem_dump_with_wiki_text.csv')
    graph_pipeline_from_csv('data/pubchem_dump_with_wiki_text.csv', './data/pubchem_graph_v1.json',api_key,'name','combined_text')
    # STAGE #4
    sampled_paths = sample_graph_pipeline('./data/pubchem_graph_v1.json',{2:10, 3:10, 4:10 })
    print(sampled_paths)
    generate_questions_from_paths(sampled_paths,api_key,'./data/pubchem_qas.json')
    # STAGE #5
    evaluate_questions('./data/pubchem_qas.json','./data/pubchem_qas_verified.json',api_key)
    