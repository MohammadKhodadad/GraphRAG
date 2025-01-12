import os
import json
import dotenv
import pandas as pd
import multiprocessing
from tqdm import tqdm
from utils.pipeline import Pipeline
from data_loader.utils.answer_evaluation import bulk_evaluation
from utils.llm import gpt_query

# Global variables for the pipeline and API key
pipeline = None
api_key = None

def init_processes():
    """Initializer function for each process."""
    global pipeline, api_key
    # Load environment variables and initialize the pipeline
    
    api_key = os.environ.get("OPENAI_API_KEY")
    pipeline = Pipeline(api_key)
    pipeline.retriever.load_model()

def process_query(args):
    """Function to process a single query."""
    index, qa = args
    question = qa['question']
    answer = qa['answer']

    # Use the global pipeline and API key
    our_response = pipeline.process_query(
        question, top_k=50, max_iterations=3, iterative_retrival_k=3, hybrid=True
    )
    gpt4o_response = gpt_query(question, api_key)
    gpto1_response = gpt_query(question, api_key, 'o1')

    return index, answer, our_response, gpt4o_response, gpto1_response

def main():
    dotenv.load_dotenv()
    print(bulk_evaluation(['references'], ['test'],os.environ.get("OPENAI_API_KEY")))
    # Load Q&A data
    with open('data_loader/data/qas.json', 'r', encoding='utf-8') as f:
        qas = json.load(f)

    references = []
    our_answers = []
    gpt4o_answers = []
    gpto1_answers = []


    # Limit the number of processes to prevent resource exhaustion
    num_processes = min(8, multiprocessing.cpu_count())
    print(f"{num_processes} processors are available.")
    # Prepare arguments for each query
    args = [(i, qas[i]) for i in range(len(qas))]  # Adjust range as needed

    # Initialize the pool with the initializer function
    with multiprocessing.Pool(processes=num_processes, initializer=init_processes) as pool:
        results = list(tqdm(pool.imap_unordered(process_query, args), total=len(args)))

    # Sort results by index to maintain order
    results.sort()

    # Collect the results
    for _, answer, our_response, gpt4o_response, gpto1_response in results:
        references.append(answer)
        our_answers.append(our_response)
        gpt4o_answers.append(gpt4o_response)
        gpto1_answers.append(gpto1_response)

    # Evaluate the answers
    answers={}
    answers['ours'] = bulk_evaluation(references, our_answers,os.environ.get("OPENAI_API_KEY"))
    answers['gpt4o'] = bulk_evaluation(references, gpt4o_answers,os.environ.get("OPENAI_API_KEY"))
    answers['gpto1'] = bulk_evaluation(references, gpto1_answers,os.environ.get("OPENAI_API_KEY"))
    print(answers)
    with open('results.json', 'w') as f:
        json.dump(answers, f)
    pd.DataFrame(answers).to_csv('results.csv')
if __name__ == '__main__':
    main()
