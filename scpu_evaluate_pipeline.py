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
    with open('data_loader/data/pubchem_qas_verified.json', 'r', encoding='utf-8') as f:
        qas = json.load(f)

    references = []
    our_answers = []
    rag_answers = []
    gpt4o_answers = []
    gpto1_answers = []

    for i in tqdm.tqdm(range(len(qas))):
        qa=qas[i]
        question = qa['q'] 
        answer = qa['a']
    
        # Use the global pipeline and API key
        our_response = pipeline.process_query(
            question, top_k=10, max_iterations=3, hybrid=True,reranker=True,reranker_top_k=10
        )
        rag_response = pipeline.process_query(
            question, top_k=10, max_iterations=0, hybrid=True,reranker=True,reranker_top_k=10
        )
        gpt4o_response =  gpt_query(question, api_key,)
        gpto1_response =  gpt_query(question, api_key, 'o1')
        references.append(answer)
        our_answers.append(our_response)
        rag_answers.append(rag_response)
        gpt4o_answers.append(gpt4o_response)
        gpto1_answers.append(gpto1_response)
        # break
    # Evaluate the answers
    answers={}
    answers['ours'] = bulk_evaluation(references, our_answers,os.environ.get("OPENAI_API_KEY"))
    answers['rag'] = bulk_evaluation(references, rag_answers,os.environ.get("OPENAI_API_KEY"))
    answers['gpt4o'] = bulk_evaluation(references, gpt4o_answers,os.environ.get("OPENAI_API_KEY"))
    answers['o1'] = bulk_evaluation(references, gpto1_answers,os.environ.get("OPENAI_API_KEY"))
    print(answers)
    # with open('results.json', 'w') as f:
    #     json.dump(answers, f)
    pd.DataFrame(answers).to_csv('results.csv')
