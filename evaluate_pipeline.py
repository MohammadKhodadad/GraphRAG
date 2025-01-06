import os
import json
import tqdm
import dotenv
import pandas as pd
from utils.pipeline import Pipeline
from data_loader.utils.wikipedia import wiki_fetch_pages_in_category_recursive_combined
from data_loader.utils.answer_evaluation import bulk_evaluation
from utils.llm import gpt4o_query

dotenv.load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")
pipeline = Pipeline(api_key)
pipeline.retriever.load_model()

with open('data_loader/data/qas.json', 'r', encoding='utf-8') as f:
    qas = json.load(f)
references=[]
our_answers=[]
gpt_answers=[]
# for i in range(len(qas)):
for i in tqdm.tqdm(range(10)):
    question=qas[i]['question']
    answer=qas[i]['answer']
    our_response=pipeline.process_query( question, top_k=50,max_iterations=3,iterative_retrival_k=3,hybrid=True)
    gpt_response=gpt4o_query(question,api_key)
    references.append(answer)
    our_answers.append(our_response)
    gpt_answers.append(gpt_response)

print("Ours:")
print(bulk_evaluation(references,our_answers))
print("GPT's:")
print(bulk_evaluation(references,our_answers))