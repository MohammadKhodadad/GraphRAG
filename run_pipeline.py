import wikipediaapi
from sentence_transformers import SentenceTransformer
import os
import json
import openai
import dotenv
import pandas as pd
import tqdm
from utils.pipeline import Pipeline
from data_loader.utils.wikipedia import wiki_fetch_pages_in_category_recursive_combined
from data_loader.utils.answer_evaluation import evaluate_similarity

if __name__ == "__main__":
    import dotenv
    dotenv.load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")

    # Step 1: Download Medical Wikipedia Pages
    # documents = wiki_fetch_pages_in_category_recursive_combined('Medicine', max_pages=1000, max_depth=2)
    # print(len(documents))
    # Step 2: Preprocess and Store Data
    pipeline = Pipeline(api_key)
    pipeline.retriever.load_model()
    # LOAD DATA
    # data=pd.read_csv('./data_loader/data/pubchem_dump_with_wiki_text.csv')
    # texts = []
    # ids = []
    # for title, page in documents.items():
    #     texts.append(page['text'])
    #     ids.append(title)
    # for index, row in tqdm.tqdm(data.head(10000).iterrows()):
    #     texts.append(row['combined_text'])
    #     ids.append(row['name'])
    #     if index%1000==0:
    #         pipeline.retriever.embed_and_store(texts, ids)
    #         texts=[]
    #         ids=[]

    # Step 3: Query the Pipeline
    with open('data_loader/data/qas.json', 'r', encoding='utf-8') as f:
        qas = json.load(f)
    question=qas[10]['question']
    answer=qas[10]['answer']
    print(question)
    # query =  "What role does the 2'-glucoside of phloretin, which is converted by hydrolytic enzymes in the small intestine and considered an irritant, play in the inhibition of glucose transport in the body?"
    response = pipeline.process_query( question, top_k=50,max_iterations=3,iterative_retrival_k=3,hybrid=True)
    print("Response:", response)
    print(evaluate_similarity(answer, response))
    print(evaluate_similarity(answer, "The 2'-glucoside of phloretin, also known as phloridzin, plays a critical role as an inhibitor of glucose transport in the body by targeting sodium-glucose cotransporters (SGLTs), particularly SGLT1 and SGLT2. When ingested, phloridzin is hydrolyzed by enzymes in the small intestine, releasing phloretin, which inhibits glucose uptake in the brush-border membrane of enterocytes. This inhibition reduces glucose absorption from the intestinal lumen into the bloodstream, impacting postprandial glucose levels. Furthermore, in the kidneys, phloridzin or its metabolites can inhibit renal glucose reabsorption by blocking SGLT2 in the proximal tubules, promoting glucosuria (excretion of glucose in urine) and lowering blood glucose levels. Despite its potential therapeutic uses, its irritant properties and poor systemic absorption limit its direct clinical application, although it has inspired the development of modern SGLT2 inhibitors used in managing diabetes.")) #Answer from chatgpt