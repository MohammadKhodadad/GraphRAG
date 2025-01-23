from sentence_transformers import SentenceTransformer
import os
import json
import tqdm
import dotenv
import pandas as pd

from utils.pipeline import Pipeline
from data_loader.utils.answer_evaluation import bulk_evaluation
from utils.llm import gpt_query


if __name__ == '__main__':
    dotenv.load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    pipeline = Pipeline(api_key)
    # pipeline.retriever.load_model()
    # Load Q&A data
    with open('data_loader/data/qas.json', 'r', encoding='utf-8') as f:
        qas = json.load(f)

    references = []
    our_answers = []
    gpt4o_answers = []
    gpt4omini_answers = []

    for i in tqdm.tqdm(range(len(qas))):
        qa=qas[i]
        question = qa['question']
        answer = qa['answer']
    
        # Use the global pipeline and API key
        our_response = pipeline.process_query(
            question, top_k=50, max_iterations=0, hybrid=True,reranker=True,reranker_top_k=15
        )
        gpt4o_response = ''#gpt_query(question, api_key,)
        gpt4omini_response = ''# gpt_query(question, api_key, 'gpt-4o-mini')
        references.append(answer)
        our_answers.append(our_response)
        gpt4o_answers.append(gpt4o_response)
        gpt4omini_answers.append(gpt4omini_response)

    # Evaluate the answers
    answers={}
    answers['ours'] = bulk_evaluation(references, our_answers,os.environ.get("OPENAI_API_KEY"))
    answers['gpt4o'] = bulk_evaluation(references, gpt4o_answers,os.environ.get("OPENAI_API_KEY"))
    answers['gpt4omini'] = bulk_evaluation(references, gpt4omini_answers,os.environ.get("OPENAI_API_KEY"))
    print(answers)
    # with open('results.json', 'w') as f:
    #     json.dump(answers, f)
    pd.DataFrame(answers).to_csv('results.csv')
