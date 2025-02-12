import pandas as pd
from openai import OpenAI
import os
import random
import json

def ask_openai(client,prompt):
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="o3-mini",
    )
    return response.choices[0].message.content.strip()

    
def generate_relation_question(client, entity1, relation, entity2, text):
    prompt = (
        f"You are given a text along with an entity and its relation to another entity.\n\n"
        f"Entity 1: {entity1}\n"
        f"Relation: {relation}\n"
        f"Entity 2 (Answer): {entity2}\n"
        f"Text: {text}\n\n"
        f"Your task is to generate a factual question about Entity 1 and its relation, where the answer is Entity1.\n"
        f"Ensure that the question is factual and can be answered solely based on the given text.\n"
        f"Do not mention a specific part of the text such as 'Abstract', 'Table #1', or etc.\n"
        f"Return a dictionary without any code formatting, backticks, or markdown, with keys 'q' and 'a'."
    )
    
    qa_dict = eval(ask_openai(client, prompt))
    return qa_dict

def generate_description_question(client, entity, description, text):
    prompt = (
        f"You are given a text along with an entity and a description.\n\n"
        f"Entity: {entity}\n"
        f"Description: {description}\n"
        f"Text: {text}\n\n"
        f"Your task is to generate a factual question about the entity that can be answered by a part of the description.\n"
        f"Ensure that the question is factual and can be answered solely based on the given text.\n"
        f"Do not mention a specific part of the text such as 'Abstract', 'Table #1', or etc.\n"
        f"Return a dictionary without any code formatting, backticks, or markdown, with keys 'q' and 'a'."
    )
    
    qa_dict = eval(ask_openai(client, prompt))
    return qa_dict

def generate_multihop_question(client, path):
    """
    Generates a multi-hop question based on a given path of entities and relations.
    
    Parameters:
        client: OpenAI client for querying the LLM.
        path: List of tuples (entity1, relation, entity2, text) forming a path.
        
    Returns:
        A dictionary with keys 'q' (multi-hop question) and 'a' (final answer).
    """
    try:
        if len(path) < 2:
            raise ValueError("Multi-hop questions require at least two edges.")

        # Generate individual questions for each relation in the path
        qa_pairs = []
        completed_path=[]
        for entity1, relation, entity2, text in path:
            qa = generate_relation_question(client, entity1, relation, entity2, text)
            qa_pairs.append(qa)
            completed_path.append([entity1, relation, entity2, text,qa['q'],qa['a']])

        # Extract only questions and answers from the generated QAs
        formatted_qas = "\n".join([f"Q{i+1}: {qa['q']}\nA{i+1}: {qa['a']}" for i, qa in enumerate(qa_pairs)])

        # Example to guide the LLM
        print(formatted_qas)
        example = """
    Example:
    Q1: What is oxidized to form Carbon Dioxide?
    A1: Methane
    Q2: What is used in Photosynthesis?
    A2: Carbon Dioxide
    Q3: What produces Oxygen?
    A3: Photosynthesis

    Multi-hop question:
    Q: What is oxidized to produce a substance that is used in a process that results in Oxygen?
    A: Methane
    """

        # Multi-hop question generation prompt
        prompt = (
            f"You are given multiple factual questions and their answers that are logically connected.\n"
            f"Your task is to chain them into a single, coherent multi-hop question that requires multiple reasoning steps.\n"
            f"Ensure that the (only) answer is the answer to the frist question, and the question naturally follows from the facts given.\n"
            f"You have to start from the last generated question and build up a single multi-hop question so it aggregates them all "
            f"and the answer is the answer to the first question.\n"
            f"None of the answers to any of the questions should be in the generated question.\n\n"
            f"Here is an example:\n{example}\n\n"
            f"Here are the generated questions and answers:\n{formatted_qas}\n\n"
            f"Return a python dictionary without any code formatting, backticks, or markdown, with keys 'q' (multi-hop question) and 'a' (final answer)."
        )

        # Get the final multi-hop question
        multi_hop_qa = eval(ask_openai(client, prompt))
        
        
        multi_hop_qa['path']=completed_path
    except Exception as e:
        print(e)
        multi_hop_qa={}
    return multi_hop_qa


def generate_questions_from_paths(paths, api_key, save_address=None):
    sampled_qas = []
    client = OpenAI(api_key=api_key)
    
    for length, paths in paths.items():
        for path in paths:
            sampled_qas.append(generate_multihop_question(client, path))
    
    if save_address:
        with open(save_address, "w", encoding="utf-8") as f:
            json.dump(sampled_qas, f, ensure_ascii=False, indent=4)
    
    return sampled_qas




if __name__ == "__main__":
    import dotenv
    dotenv.load_dotenv()
    api_key=os.environ.get("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    
    # # Test Relation Question
    # entity1 = "Water"
    # relation = "Boiling Point"
    # entity2 = "100°C"
    # text = "Water has a boiling point of 100°C under normal atmospheric pressure."
    # print(generate_relation_question(client, entity1, relation, entity2, text))
    
    # # Test Description Question
    # entity = "Oxygen"
    # description = "Oxygen is a colorless, odorless gas that is essential for respiration."
    # text = "Oxygen is an essential gas for life, aiding in respiration and combustion."
    # print(generate_description_question(client, entity, description, text))


    path = [
        ("Glucose", "is broken down into", "Pyruvate", "During glycolysis, glucose is broken down into pyruvate, producing ATP."),
        ("Pyruvate", "is converted into", "Acetyl-CoA", "Pyruvate undergoes decarboxylation to form Acetyl-CoA in the mitochondria."),
        ("Acetyl-CoA", "enters", "Krebs Cycle", "Acetyl-CoA enters the Krebs cycle, where it undergoes oxidation."),
        ("Krebs Cycle", "produces", "ATP", "The Krebs cycle produces ATP, NADH, and FADH2 as energy carriers."),
    ]



    qa_pair = generate_multihop_question(client, path)
    print(qa_pair)