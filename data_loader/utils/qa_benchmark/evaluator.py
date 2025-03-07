import os
import time
import tqdm
from openai import OpenAI
import dotenv
import pandas as pd

def ask_openai(client, prompt):
    """
    Given a client and a prompt, send the prompt to OpenAI and return the answer.
    """
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="o3-mini",
    )
    return response.choices[0].message.content.strip()

def evaluate_answer(expected, candidate, client):
    """
    Evaluate whether the candidate answer is correct.
    
    1. First, check for an exact match (ignoring case).
    2. If not an exact match, ask OpenAI using a yes/no prompt whether the candidate answer
       is semantically equivalent to the expected answer.
       
    Returns True if the candidate answer is considered correct, False otherwise.
    """
    if candidate.strip().lower() == expected.strip().lower():
        return True
    else:
        eval_prompt = (
            f"Expected answer: '{expected}'\n"
            f"Candidate answer: '{candidate}'\n"
            "Is the candidate answer semantically equivalent to the expected answer? "
            "Answer only with 'yes' or 'no'."
        )
        evaluation = ask_openai(client, eval_prompt)
        return evaluation.lower().strip() == "yes"

class QAModel:
    """
    A class representing a question-answering model.
    For now, it only supports an OpenAI-based model.
    """
    def __init__(self, api_key, name='openai', model_name='o1', **kwargs):
        self.api_key = api_key
        self.name = name.lower()
        self.model_name = model_name
        self.client = None
        if self.name == 'openai':
            self.client = OpenAI(api_key=api_key)

    def ask(self, prompt):
        """
        Ask the model a given prompt and return the response along with duration and token usage.
        """
        if self.name == 'openai':
            start_time = time.time()
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_name,
            )
            duration = time.time() - start_time
            tokens = response.usage.total_tokens if hasattr(response.usage, "total_tokens") else None
            return response.choices[0].message.content.strip(), duration, tokens
        else:
            raise Exception("Not implemented.")

    def prompt_with_question(self, question):
        """
        Build a prompt using only the question.
        """
        prompt = (
            f"Provide a single word (or compound word) answer to the following question:\n"
            f"Question: {question}"
        )
        return self.ask(prompt)

    def prompt_with_question_context(self, question, context):
        """
        Build a prompt using both the question and the provided context.
        """
        prompt = (
            "Provide a single word (or compound word) answer to the following question based on the Context.\n"
            f"Context: {context}\n"
            f"Question: {question}"
        )
        return self.ask(prompt)

    def call(self, qa):
        """
        Accepts a single QA dict or a list of QA dicts.
        Each dict should include:
            - 'question'
            - optionally, 'context'
            - 'expected' : the expected answer.
        
        Returns for each QA item:
            - answer_without, duration_without, tokens_without
            - answer_with, duration_with, tokens_with
        """
        def process_item(item):
            question = item.get("question")
            context = item.get("context", "")
            length = item.get("length",None)
            answer_without, duration_without, tokens_without = self.prompt_with_question(question)
            if context:
                answer_with, duration_with, tokens_with = self.prompt_with_question_context(question, context)
            else:
                answer_with, duration_with, tokens_with = answer_without, duration_without, tokens_without
            return {
                "question": question,
                "context": context,
                "length":length,
                "expected": item.get("expected", "").strip(),
                "answer_without": answer_without,
                "duration_without": duration_without,
                "tokens_without": tokens_without,
                "answer_with": answer_with,
                "duration_with": duration_with,
                "tokens_with": tokens_with
            }
        
        if isinstance(qa, list):
            return [process_item(item) for item in tqdm.tqdm(qa)]
        elif isinstance(qa, dict):
            return process_item(qa)
        else:
            raise ValueError("Input must be a dictionary or a list of dictionaries.")

class Evaluator:
    """
    Evaluator class that instantiates a set of models and evaluates each on a set of QA items.
    For each model, it produces two rows of results:
      - One for evaluation using answers without context.
      - One for evaluation using answers with context.
    The results include accuracy, average duration, and average token usage.
    """
    def __init__(self, eval_client, model_configs):
        """
        eval_client: an OpenAI client used for evaluation prompts.
        model_configs: list of dicts, each with keys 'name', 'api_key', and 'model_name'
        """
        self.eval_client = eval_client
        self.models = {}
        for config in model_configs:
            model_type = config.get("name", "openai")
            api_key = config["api_key"]
            variant = config.get("model_name", "o1")
            key = f"{model_type}-{variant}"
            self.models[key] = QAModel(api_key, name=model_type, model_name=variant)

    def run_evaluation(self, qa_items):
        """
        qa_items: list of dictionaries. Each should include:
            - 'question'
            - optionally 'context'
            - 'expected' (expected answer)
        
        For each model, the evaluator:
         - Runs the model on all QA items.
         - Evaluates the answers both without context and with context.
         - Aggregates accuracy, average duration, and average token usage for both runs.
         
        Returns a pandas DataFrame with two rows per model.
        """
        results = []

        for model_key, model in self.models.items():
            total = len(qa_items)
            # Aggregators for answers without context.
            correct_without = 0
            total_duration_without = 0.0
            total_tokens_without = 0

            # Aggregators for answers with context.
            correct_with = 0
            total_duration_with = 0.0
            total_tokens_with = 0
            print(f'Running queries for model :{model_key}')
            qa_results = model.call(qa_items)
            # Ensure we have a list of results.
            if not isinstance(qa_results, list):
                qa_results = [qa_results]
            print(f'Running evaluations for model :{model_key}')
            for res in qa_results:
                expected = res["expected"]
                # Evaluate without context.
                candidate_without = res["answer_without"].strip()
                is_correct_without = evaluate_answer(expected, candidate_without, self.eval_client)
                if is_correct_without:
                    correct_without += 1
                total_duration_without += res["duration_without"]
                if res["tokens_without"] is not None:
                    total_tokens_without += res["tokens_without"]

                # Evaluate with context.
                candidate_with = res["answer_with"].strip()
                is_correct_with = evaluate_answer(expected, candidate_with, self.eval_client)
                if is_correct_with:
                    correct_with += 1
                total_duration_with += res["duration_with"]
                if res["tokens_with"] is not None:
                    total_tokens_with += res["tokens_with"]

            accuracy_without = (correct_without / total) * 100 if total > 0 else 0
            avg_duration_without = total_duration_without / total if total > 0 else 0
            avg_tokens_without = total_tokens_without / total if total > 0 else 0

            accuracy_with = (correct_with / total) * 100 if total > 0 else 0
            avg_duration_with = total_duration_with / total if total > 0 else 0
            avg_tokens_with = total_tokens_with / total if total > 0 else 0

            results.append({
                "Model": model_key,
                "Run": "Without Context",
                "Accuracy (%)": round(accuracy_without, 2),
                "Avg Duration (s)": round(avg_duration_without, 2),
                "Avg Tokens": round(avg_tokens_without, 2)
            })

            results.append({
                "Model": model_key,
                "Run": "With Context",
                "Accuracy (%)": round(accuracy_with, 2),
                "Avg Duration (s)": round(avg_duration_with, 2),
                "Avg Tokens": round(avg_tokens_with, 2)
            })

        return pd.DataFrame(results)

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
        # You can add additional model configurations here.
        # {"name": "deepseek", "api_key": "YOUR_DEEPSEEK_API_KEY", "model_name": "d1"}
    ]
    
    # Instantiate the evaluator.
    evaluator = Evaluator(eval_client, model_configs)
    
    # Define a list of QA items for evaluation.
    qa_items = [
        {
            "question": "What is the capital of France?",
            "context": "France is a country in Europe with many famous landmarks.",
            "expected": "Paris"
        },
        {
            "question": "What is the largest planet in our Solar System?",
            "context": "The Solar System has eight planets and several dwarf planets.",
            "expected": "Jupiter"
        }
    ]
    
    # Run the evaluation.
    df_results = evaluator.run_evaluation(qa_items)
    print("Evaluation Results:")
    print(df_results)
