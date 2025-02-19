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
from utils.llm import gpt_query
from data_loader.utils.chemrxiv import chemrxiv_embed_and_store
if __name__ == "__main__":
    import dotenv
    dotenv.load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")

    # Step 1: Download Medical Wikipedia Pages
    # documents = wiki_fetch_pages_in_category_recursive_combined('Medicine', max_pages=1000, max_depth=2)
    # print(len(documents))
    # Step 2: Preprocess and Store Data
    pipeline = Pipeline(api_key)
    # chemrxiv_embed_and_store(pipeline,'./data_loader/data/chemrxiv_papers/')
    # LOAD DATA
    # data=pd.read_csv('./data_loader/data/pubchem_dump_with_wiki_text.csv')
    
    # texts = []
    # ids = []
    # # for title, page in documents.items():
    # #     texts.append(page['text'])
    # #     ids.append(title)
    # for index, row in tqdm.tqdm(data.head(10000).iterrows()):
    #     texts.append(row['combined_text'])
    #     ids.append(row['name'])
    #     if index%1000==0 or index==9999:
    #         pipeline.cot.retriever.embed_and_store(texts, ids)
    #         texts=[]
    #         ids=[]
    question = 'What compound, known for its electron-withdrawing properties and role in site-selective CH functionalization, reacts with a nonalternant fused hydrocarbon (characterized by its near-gapless electronic structure) when exposed to trifluoromethanesulfonic anhydride and acid, ultimately yielding an unidentified side-product?' + " (The answer to this question is just a name of an entity, so just return that name with no text.)"
    response = pipeline.process_query(
        question, top_k=10, max_iterations=3, hybrid=True,reranker=True,reranker_top_k=10
    )
    print(f"Question: {question}")
    print(f"GraphRAG Response: {response}")