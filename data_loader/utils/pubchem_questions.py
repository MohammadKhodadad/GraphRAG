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

    


def pubchem_generate_2_hop_questions(address="data/pubchem_dump_with_wiki_text.csv",count=3,api_key=None):
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
                prompt1 = (
                    f"You are given a text and a keyword.\n\n"
                    f"Text: {text1}\n"
                    f"Keyword: {bridge}\n"
                    f"Your job is to generate a factual question about the Keyword from the text, that explains 2 aspects of it (include safety if possible). The answer has to come from the given text.\n"
                    f"Return a dictionary without any code formatting, backticks, or markdown , with keys 'q' and 'a' "
                )
                q1= eval(ask_openai(client,prompt1))
                print('Q1:')
                print(q1['q'])
                print(q1['a'])
                print(text1,'\n')
                prompt2 = (
                    f"You are given a text and a keyword.\n\n"
                    f"Text: {text2}\n"
                    f"Keyword: {bridge}\n"
                    f"Your job is to generate a factual question about the Keyword from the text, that explains an aspect of it. The answer has to come from the given text.\n"
                    f"Return a dictionary without any code formatting, backticks, or markdown , with keys 'q' and 'a' "
                )
                q2= eval(ask_openai(client,prompt2))
                print('Q2:')
                print(q2['q'])
                print(q2['a'])
                print(text2,'\n')
                prompt3 = (
                    f"You are tasked with generating a factual multi-hop question that combines two questions:\n"
                    f"Details:\n"
                    f"question1: {q1['q']}\n"
                    f"answer1: {q1['a']}\n"
                    f"bridge word: {bridge}\n"
                    f"question2: {q2['q']}\n"
                    f"answer2: {q2['a']}\n\n"
                    f"Both questions describe the bridge word."
                    f"Generate a question that is the second question but does not have the bridge word, instead has the description from the first question.\n"
                    f"The answer, logically, has to be the answer to the second question."
                    f"Make sure the question has the following features:\n"
                    f"- The question should only have one part.\n"
                    f"- Do not mention the bridge word in the question.\n\n"
                    f"Return your output as a dictionary with keys 'q' (for the question) and 'a' (for the final answer, derived from combining both questions).\n"
                    f"Do not use any code formatting, backticks, or markdown in your response."
                    )

                q3= eval(ask_openai(client,prompt3))
                print('Q3:')
                print(q3['q'])
                print(q3['a'])
                print()
                qa['question']=q3['q']
                qa['answer']=q3['a']
                qas.append(qa)
    return qas


if __name__=='__main__':
    import dotenv
    dotenv.load_dotenv()
    qas=pubchem_generate_2_hop_questions(address="../data/pubchem_dump_with_wiki_text.csv",api_key=os.environ.get("OPENAI_API_KEY"),count=2)
    # for i,qa in enumerate(qas):
    #     print(f'case: {i}')
    #     print(qa['question'])
    #     print(qa['answer'])

    
