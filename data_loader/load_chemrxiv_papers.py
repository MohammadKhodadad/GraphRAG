import os
import json
import dotenv
from utils.chemrxiv import chemrxiv_download_papers, chemrxiv_fetch_all_papers
from utils.graph_generation.pipeline import graph_pipeline, sample_graph_pipeline
from utils.question_generation.question_generation import generate_questions_from_paths, evaluate_questions
from openai import OpenAI
dotenv.load_dotenv()

api_key = os.environ.get("OPENAI_API_KEY")  # Ensure this is set in your environment
# STAGE 1
# all_papers=chemrxiv_fetch_all_papers(output_file="./data/chemrxiv_data_1501.json",total=100000, search_data_from="2015-01-01")
# with open('./data/chemrxiv_data_1501.json','rb') as f:
#     all_papers=json.load(f)
# chemrxiv_download_papers(all_papers,'./data/chemrxiv_papers_v2')
# # # STAGE 2
# graph_pipeline('./data/chemrxiv_papers_v2','./data/chemrxiv_graph_v2.json',api_key)
# # STAGE 3
# sampled_paths = sample_graph_pipeline('./data/chemrxiv_graph_v2.json',{1:300, 2:300, 3:300, 4:300 })
# print(sampled_paths)
# # STAGE 4
# generate_questions_from_paths(sampled_paths,api_key,'./data/chemrxiv_qas_v2_3.json')
evaluate_questions('./data/chemrxiv_qas_v2_3.json','./data/chemrxiv_qas_v2_3_verified.json',api_key)
