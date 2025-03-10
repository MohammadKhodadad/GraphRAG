import os
import json
import random 

with open('./data/chemrxiv_qas_v2_2.json','rb') as f:
    qas=json.load(f)
print(len(qas))
for qa in qas:
    qa['Question']=qa.pop('q')
    qa['Answer']=qa.pop('a')
    qa['sub_questions']='    '.join([f'Question #{i+1}: {item["q"]} Answer: {item["a"]}' for i,item in enumerate(qa.pop('path'))])
with open('./data/chemrxiv_qas_sample.json', "w", encoding="utf-8") as f:
    json.dump(qas, f, ensure_ascii=False, indent=4)
