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
from data_loader.utils.pubchem import pubchem_embed_and_store
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
    
    # pubchem_embed_and_store(pipeline,'./data_loader/data/pubchem_dump_with_wiki_text.csv')
    question = 'Considering that the conformation of 3,6-bridged derivatives is influenced by their substituent groups, which simple sugar serves as the central core for most ellagitannins?'
    # question = "Palmitoylcarnitine participates in which metabolic process essential for energy production?"
    response = pipeline.process_query(
        question, top_k=5, max_iterations=5, hybrid=True,reranker=True,reranker_top_k=5
    )
    print(f"Question: {question}")
    print(f"GraphRAG Response: {response}")