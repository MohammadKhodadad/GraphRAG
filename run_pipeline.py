import wikipediaapi
from sentence_transformers import SentenceTransformer
import os
import openai
import dotenv

from utils.pipeline import Pipeline
from utils.wikipedia import wiki_fetch_pages_in_category_recursive_combined

if __name__ == "__main__":
    import dotenv
    dotenv.load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")

    # Step 1: Download Medical Wikipedia Pages
    # documents = wiki_fetch_pages_in_category_recursive_combined('Medicine', max_pages=1000, max_depth=2)
    # print(len(documents))
    # Step 2: Preprocess and Store Data
    pipeline = Pipeline(api_key)
    texts = []
    ids = []
    # for title, page in documents.items():
    #     texts.append(page['text'])
    #     ids.append(title)
    pipeline.retriever.load_model()
    # pipeline.retriever.embed_and_store(texts, ids)

    # Step 3: Query the Pipeline
    query = "What foods help with eye dryness?"
    response = pipeline.process_query( query, top_k=10,max_iterations=3,iterative_retrival_k=2,hybrid=True)
    print("Response:", response)