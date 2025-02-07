import pandas as pd
from openai import OpenAI
import os
import random


def ask_openai(client,prompt):
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-4o",
    )
    return response.choices[0].message.content.strip()

    
def generate_relation_question(client, entity1, relation, entity2, text):
    prompt = (
        f"You are given a text along with an entity and its relation to another entity.\n\n"
        f"Entity 1: {entity1}\n"
        f"Relation: {relation}\n"
        f"Entity 2 (Answer): {entity2}\n"
        f"Text: {text}\n\n"
        f"Your task is to generate a factual question about Entity 1 and its relation, where the answer is Entity 2.\n"
        f"Ensure that the question is factual and can be answered solely based on the given text.\n"
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
        f"Return a dictionary without any code formatting, backticks, or markdown, with keys 'q' and 'a'."
    )
    
    qa_dict = eval(ask_openai(client, prompt))
    return qa_dict

if __name__ == "__main__":
    import dotenv
    dotenv.load_dotenv()
    api_key=os.environ.get("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    
    # Test Relation Question
    entity1 = "Water"
    relation = "Boiling Point"
    entity2 = "100°C"
    text = "Water has a boiling point of 100°C under normal atmospheric pressure."
    print(generate_relation_question(client, entity1, relation, entity2, text))
    
    # Test Description Question
    entity = "Oxygen"
    description = "Oxygen is a colorless, odorless gas that is essential for respiration."
    text = "Oxygen is an essential gas for life, aiding in respiration and combustion."
    print(generate_description_question(client, entity, description, text))
