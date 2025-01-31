import os
import json
import dotenv
from utils.chemrxiv import chemrxiv_download_papers, chemrxiv_fetch_all_papers_from_2024
from utils.graph_generation.pipeline import graph_pipeline

dotenv.load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")  # Ensure this is set in your environment
all_papers=chemrxiv_fetch_all_papers_from_2024(output_file="./data/chemrxiv_data_2306.json",total=600)
with open('./data/chemrxiv_data_2306.json','rb') as f:
    all_papers=json.load(f)
chemrxiv_download_papers(all_papers,'./data/chemrxiv_papers')
# process_pdfs_to_dataframe('./data/chemrxiv_papers','./data/chemrxiv_data.csv',api_key)
graph_pipeline('./data/chemrxiv_papers','./data/chemrxiv_graph_v1.json',api_key)