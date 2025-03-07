import os
import random
import dotenv
import json
from openai import OpenAI
from utils.qa_benchmark.evaluator import Evaluator
from utils.qa_benchmark.create_qas import load_qa_items
# Example usage:
if __name__ == "__main__":
    # Load environment variables.
    dotenv.load_dotenv()
    eval_api_key = os.environ.get("OPENAI_API_KEY")
    
    # Evaluation client for ask_openai (using o3-mini model as before)
    eval_client = OpenAI(api_key=eval_api_key)
    
    # Define model configurations to be evaluated.
    model_configs = [
        {"name": "openai", "api_key": os.environ.get("OPENAI_API_KEY"), "model_name": "o1"},
        {"name": "openai", "api_key": os.environ.get("OPENAI_API_KEY"), "model_name": "o3-mini"},
        {"name": "openai", "api_key": os.environ.get("OPENAI_API_KEY"), "model_name": "gpt-4o"}
        # You can add additional model configurations here.
        # {"name": "deepseek", "api_key": "YOUR_DEEPSEEK_API_KEY", "model_name": "d1"}
    ]
    
    # Instantiate the evaluator.
    evaluator = Evaluator(eval_client, model_configs)
    
    # Define a list of QA items for evaluation.
    qa_items = load_qa_items("/media/torontoai/GraphRAG/GraphRAG/data_loader/data/chemrxiv_qas.json")
    qa_items = random.sample(qa_items, 10)
    # Run the evaluation.
    df_results = evaluator.run_evaluation(qa_items)
    print("Evaluation Results:")
    print(df_results)


## ADD THE LENGTH OF PATH