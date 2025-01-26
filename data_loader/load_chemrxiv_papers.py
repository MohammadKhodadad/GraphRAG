import json
from utils.chemrxiv import chemrxiv_download_papers, chemrxiv_fetch_all_papers_from_2024, process_pdfs_to_dataframe


all_papers=chemrxiv_fetch_all_papers_from_2024(output_file="./data/chemrxiv_data_2306.json",total=15000)
with open('./data/chemrxiv_data_2306.json','rb') as f:
    all_papers=json.load(f)
chemrxiv_download_papers(all_papers,'./data/chemrxiv_papers')
process_pdfs_to_dataframe('./data/chemrxiv_papers','./data/chemrxiv_data.csv')