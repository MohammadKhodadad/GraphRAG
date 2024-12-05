from sentence_transformers import CrossEncoder
import os
import openai
from openai import OpenAI


from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import kneighbors_graph
import numpy as np

#HybridSearch
#Chemistry Benchmark
#Chemistry Retrieval
#GraphReasoning (Chain/Graph/Tree of Thoughts)
#GraphQuery
#FutureHouse
#ChemCrow
#KnowledgeGraph for NextQuestionGeneration & Retrieval
#GraphRAG Microsoft


class GraphReranker:
    def __init__(self, api_key, model_name: str = 'all-MiniLM-L6-v2',
                 cross_encoder_model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"):
        """
        Initialize the GraphReranker class.
        """
        self.embedding_model = SentenceTransformer(model_name)
        self.cross_encoder_model = CrossEncoder(cross_encoder_model_name)
        self.client = OpenAI(api_key=api_key)

    def extract_paragraphs(self, documents):
        """
        Extract paragraphs from documents.
        """
        if 'documents' not in documents:
            raise ValueError("The input dictionary must contain a 'documents' key.")
        
        paragraphized_documents = []
        for doc in documents['documents'][0]:
            try:
                paragraphs = [paragraph.strip() for paragraph in doc.split("\n") if paragraph.strip()]
                paragraphized_documents.append(paragraphs)
            except:
                print(doc)
                # pass
        return [paragraph for doc in paragraphized_documents for paragraph in doc]

    def cross_encoder_similarity_search(self, query, paragraphs, top_k=1):
        """
        Perform similarity search using the cross-encoder model.
        """
        text_pairs = [(query, paragraph) for paragraph in paragraphs]
        # print(text_pairs)
        scores=[self.cross_encoder_model.predict([text_pairs[i]])[0] for i in range(len(text_pairs))]
        # print(scores)
        # scores = self.cross_encoder_model.predict(text_pairs[:1],batch_size=1)
        scored_paragraphs = sorted(zip(paragraphs, scores), key=lambda x: x[1], reverse=True)
        return scored_paragraphs[:top_k]

    def next_question(self, query, pool, is_first=False):
        """
        Ask OpenAI what additional information is needed to answer the query.
        """
        pool_text = "\n\n".join(pool)
        prompt = (
            f"You are given a set of paragraphs and a query.\n\n"
            f"Query: {query}\n\n"
            f"Relevant Paragraphs:\n{pool_text}\n\n"
            f"Your job is to output the next best question based on the available relevant paragraphs that we need to answer to answer the query. "
            f"Return only the question as plain text."
        )
        if is_first:
            prompt = (
            f"You are given a query.\n\n"
            f"Query: {query}\n\n"
            f"Your job is to output the best starting term from the query that we need to search for in our databse to be able to answer the query. "
            f"Return only the term as plain text."
        )
        response = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="gpt-4o",
        )
        return response.choices[0].message.content.strip()

    def iterative_retrieval(self, query, documents, max_iterations=5, top_k=1):
        """
        Iteratively retrieve relevant paragraphs by interacting with OpenAI and using the cross-encoder.
        Args:
            query (str): The initial query.
            documents (dict): Input documents to retrieve from.
            max_iterations (int): Maximum number of iterations for the retrieval process.
            top_k (int): Number of top paragraphs to add at each step.
        Returns:
            list: The final pool of relevant paragraphs.
        """
        all_paragraphs = self.extract_paragraphs(documents)
        pool = []  # Start with an empty pool
        remaining_paragraphs = [paragraph for paragraph in all_paragraphs]

        # Initial similarity search
        # first_results = self.cross_encoder_similarity_search(query, remaining_paragraphs, top_k=top_k)
        first_question= self.next_question(query, pool, is_first=True)
        print(f"First Question:{ first_question}")
        first_results = self.cross_encoder_similarity_search(first_question, remaining_paragraphs, top_k=top_k)
        for paragraph, _ in first_results:
            if paragraph not in pool:
                pool.append(paragraph)
                remaining_paragraphs.remove(paragraph)

        for _ in range(max_iterations):
            # Ask OpenAI what is needed
            additional_question = self.next_question(query, pool)
            print(f"Next Question:{ additional_question}")
            # Perform similarity search with the additional question
            if len(remaining_paragraphs)==0:
                break
            top_paragraphs = self.cross_encoder_similarity_search(additional_question,remaining_paragraphs,top_k=top_k)

            # if not scored_paragraphs:
            #     break  # Exit if no more relevant paragraphs are found
            
            # Add top_k paragraphs to the pool
            for paragraph, _ in top_paragraphs:
                if paragraph not in pool:
                    pool.append(paragraph)
                    remaining_paragraphs.remove(paragraph)

        return pool

if __name__ == "__main__":
    import dotenv
    dotenv.load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")  # Ensure this is set in your environment
    graph_reranker = GraphReranker(api_key)

    documents = {
        "documents": [
            "Artificial intelligence is transforming various industries. It provides tools for personalized treatment.\n\n"
            "AI helps in diagnosing diseases earlier and improving healthcare outcomes.\n\n"
            "Machine learning is also being used in predictive analytics in finance.",
            
            "Natural language processing (NLP) enables better human-computer interactions.\n\n"
            "AI in education personalizes learning and automates administrative tasks.\n\n"
            "Robotics powered by AI is revolutionizing manufacturing processes."
        ]
    }

    query = "How is AI transforming healthcare?"
    relevant_paragraphs = graph_reranker.iterative_retrieval(query, documents, max_iterations=3, top_k=2)
    print("\n".join(relevant_paragraphs))