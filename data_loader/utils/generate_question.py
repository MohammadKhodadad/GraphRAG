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

    


def generate_2_hop_questions(address="data/pubchem_dump_with_wiki_text.csv",count=3,api_key=None):
    client = OpenAI(api_key=api_key)
    data=pd.read_csv(address)
    data_with_edges=data[data.mentioned_entity!='[]']
    qas=[]
    # Generate a random list of 10 unique numbers between 0 and 100
    random_numbers = random.sample(range(0, len(data_with_edges)), count)

    for number in random_numbers:
        row=data_with_edges.iloc[number]
        mentioned_entity=[int(item)-1 for item in eval(row['mentioned_entity']) if int(item)<=50000]
        for edge in mentioned_entity:
            next_row=data.iloc[edge]
            if len(next_row['combined_text'])>10:
                qa={'source':row['name'],'source_cid':row['cid'],'bridge':next_row['name'],
                    'bridge_cid':next_row['cid'],'text1':row['combined_text'],'text2':next_row['combined_text'],}
                text1,bridge,text2=qa['text1'],qa['bridge'],qa['text2']
                prompt1= (
                    f"You are given a text and a bridge word.\n\n"
                    f"Text: {text1}\n"
                    f"Bridge Word: {bridge}\n"
                    f"Your job is to generate a question that requires the user to read the  document and find the bridge word as the answer. "
                    f"Return a dictionary without any code formatting, backticks, or markdown , with keys 'q' and 'a' "
                )
                q1= eval(ask_openai(client,prompt1))

                prompt2 = (
                    f"You are given a text and a bridge word.\n\n"
                    f"Text: {text2}\n"
                    f"Key Word: {bridge}\n"
                    f"Your job is to generate a question about the Key word "
                    f"Return a dictionary without any code formatting, backticks, or markdown , with keys 'q' and 'a' "
                )
                q2= eval(ask_openai(client,prompt2))


                prompt3 = (
                    f"You are given two question with their answers and a bridge word.\n\n"
                    f"question1: {q1['q']}\n"
                    f"answer1: {q1['a']}\n"
                    f"question2: {q2['q']}\n"
                    f"answer2: {q2['a']}\n"
                    f"Your job is to generate a multi-hop question, that combines these two questions.  "
                    f"The question should require the person to find the answer to the first question then look for the answer of the second question"
                    f"Return a dictionary without any code formatting, backticks, or markdown , with keys 'q' and 'a' "
                )
                q3= eval(ask_openai(client,prompt2))
                qa['question']=q3['q']
                qa['answer']=q3['a']
                qas.append(qa)
    return qas


if __name__=='__main__':
    import dotenv
    dotenv.load_dotenv()
    qas=generate_2_hop_questions(address="../data/pubchem_dump_with_wiki_text.csv",api_key=os.environ.get("OPENAI_API_KEY"),count=2)
    for i,qa in enumerate(qas):
        print(f'case: {i}')
        print(qa['question'])
        print(qa['answer'])

    
