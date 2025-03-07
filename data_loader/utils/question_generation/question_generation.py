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

    
def generate_relation_question(client, entity1, relation, entity2, text, entity1_meta=''):
    prompt = (
        f"You are given a text along with an entity and its relation to another entity.\n\n"
        f"Entity 1: {entity1}\n"
        f"Relation: {relation}\n"
        f"Entity 2 (Answer): {entity2}\n"
        f"Text: {text}\n\n"
        f"Information about Entity1: {entity1_meta if entity1_meta else 'None'}\n"
        f"Your task is to generate a factual question about Entity 1 and its relation, where the answer is Entity1.\n"
        f"Ensure that the question is factual and can be answered solely based on the given text and the information about Entity 1.\n"
        f"Do not point to the text such as 'Abstract', 'Table #1', 'in the text', 'in the article', or etc.\n"
        f"If the entity and relation is not specific enough, try to add descriptions FROM THE TEXT or FROM THE INFORMATION ABOUT ENTITY 1 to make it specifc.\n"
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
        f"Do not point to the text such as 'Abstract', 'Table #1', 'in the text', 'in the article', or etc.\n"
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
        

        # Generate individual questions for each relation in the path
        qa_pairs = []
        completed_path=[]
        for entity1, relation, entity2, text, source, meta1 in path:
            qa = generate_relation_question(client, entity1, relation, entity2, text, meta1)
            qa_pairs.append(qa)
            completed_path.append({'entity1':entity1, 'relation':relation, 'entity2': entity2, 'text': text ,'meta1':meta1 ,'q':qa['q'],'a':qa['a']})
        if len(path) < 2:
            return {'q':completed_path[0]['q'],'a':completed_path[0]['a'],'path':completed_path}
        # Extract only questions and answers from the generated QAs
        formatted_qas = "\n".join([f"Q{i+1}: {qa['q']}\nA{i+1}: {qa['a']}" for i, qa in enumerate(qa_pairs)])

        # Example to guide the LLM
        # print(formatted_qas)
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


def is_factual_chemistry_question(client, question, answer, path):
    """
    Determines whether a question is a factual chemistry question by checking its validity against the given path using an LLM.

    Parameters:
        client: OpenAI client for querying the LLM.
        question (str): The chemistry-related question to evaluate.
        answer (str): The expected answer to the question.
        path (list): List of dictionaries containing entities, relations, and text.

    Returns:
        str: 'yes' if the question is factual in chemistry, 'no' otherwise.
    """
    try:
        if not question or not answer or not path:
            return 'no'
        
        # Format the path for LLM input
        path_text = "\n".join([f"{entry['entity1']} {entry['relation']} {entry['entity2']}: {entry['text']} \n {entry['meta1']}" for entry in path])
        
        # Prompt for LLM
        prompt = f"""
        You are a chemistry expert. Your task is to determine if the given question is a factual chemistry question and answerable based on the provided path.
        
        ### Path Information:
        {path_text}
        
        ### Question:
        {question}
        
        ### Answer:
        {answer}
        
        Please analyze the path and verify if the question is a factual chemistry question and can be answered based on the given path. A factual question must be based on actual chemical properties, reactions, or experimentally verified principles. An answerable question should be solvable based on the give path. If the question is factual and answerable, return 'yes'. If it contains speculation, opinions, or lacks verifiable chemical grounding, or it is not solvable, return 'no'.
        
        ### Examples of Factual Chemistry Questions:
        ✅ "What dissolves in water?"
        ✅ "What catalyst is used in the reaction between A and B?"
        ✅ "Which compound undergoes oxidation in this reaction?"
        ✅ "What product is formed when sodium reacts with chlorine?"
        
        ### Examples of Non-Factual Chemistry Questions:
        🚫 "Why do some scientists think this reaction is inefficient?"
        🚫 "What is the best solvent for this reaction?"
        🚫 "Is this reaction useful in industry?"
        🚫 "Do you think this compound is a good catalyst?"

        Provide only 'yes' or 'no' as your response.
        """
        
        # Query the LLM
        response = ask_openai(client,prompt)
        
        return response
    
    except Exception as e:
        print(f"Error: {e}")
        return 'no'

def evaluate_questions(questions_address, verified_questions_address, api_key):
    selected_questions = []
    count = 0
    client = OpenAI(api_key=api_key)
    with open(questions_address,'rb') as f:
        questions=json.load(f)
    for question_item in questions:
        acceptable = is_factual_chemistry_question(client, question_item['q'],question_item['a'],question_item['path'])
        if acceptable.lower().strip()=='yes':
            selected_questions.append(question_item)
            count+=1
    with open(verified_questions_address, "w", encoding="utf-8") as f:
        json.dump(selected_questions, f, ensure_ascii=False, indent=4)
    print(f'{count} out of {len(questions)} were acceptable.')

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
    print(is_factual_chemistry_question(client, qa_pair['q'], qa_pair['a'], path))