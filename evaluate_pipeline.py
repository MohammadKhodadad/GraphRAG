import os
import json
import dotenv
import multiprocessing
from tqdm import tqdm
from utils.pipeline import Pipeline
from data_loader.utils.answer_evaluation import bulk_evaluation
from utils.llm import gpt4o_query

# Global variables for the pipeline and API key
pipeline = None
api_key = None

def init_processes():
    """Initializer function for each process."""
    global pipeline, api_key
    # Load environment variables and initialize the pipeline
    dotenv.load_dotenv()
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
    gpt_response = gpt4o_query(question, api_key)

    return index, answer, our_response, gpt_response

def main():
    # Load Q&A data
    with open('data_loader/data/qas.json', 'r', encoding='utf-8') as f:
        qas = json.load(f)

    references = []
    our_answers = []
    gpt_answers = []

    # Limit the number of processes to prevent resource exhaustion
    num_processes = min(4, multiprocessing.cpu_count())

    # Prepare arguments for each query
    args = [(i, qas[i]) for i in range(10)]  # Adjust range as needed

    # Initialize the pool with the initializer function
    with multiprocessing.Pool(processes=num_processes, initializer=init_processes) as pool:
        results = list(tqdm(pool.imap_unordered(process_query, args), total=len(args)))

    # Sort results by index to maintain order
    results.sort()

    # Collect the results
    for _, answer, our_response, gpt_response in results:
        references.append(answer)
        our_answers.append(our_response)
        gpt_answers.append(gpt_response)

    # Evaluate the answers
    print("Ours:")
    print(bulk_evaluation(references, our_answers))
    print("GPT's:")
    print(bulk_evaluation(references, gpt_answers))

if __name__ == '__main__':
    main()
