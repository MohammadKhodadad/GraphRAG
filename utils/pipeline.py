if __name__ == "__main__":
    
    from llm import OpenAIQA
    from retriever import Retriever
    from graph_reranker import GraphReranker
else:
    from .llm import OpenAIQA
    from .retriever import Retriever
    from .graph_reranker import GraphReranker



class Pipeline:
    def __init__(self,api_key):
        """
        Initialize the Pipeline with an OpenAIQA object and a Retriever object.

        Parameters:
        - openai_qa (OpenAIQA): The OpenAIQA object for question answering.
        - retriever (Retriever): The Retriever object for document retrieval.
        """
        self.openai_qa = OpenAIQA(api_key)
        self.retriever = Retriever()
        self.graph_reranker = GraphReranker(api_key)

    def process_query(self, query, top_k=5,max_iterations=5,iterative_retrival_k=2, hybrid=True):
        """
        Process the query by retrieving relevant documents and getting an answer.

        Parameters:
        - query (str): The user's question.
        - top_k (int): The number of top documents to retrieve.

        Returns:
        - str: The answer to the query.
        """
        # Retrieve relevant documents
        retrieved_docs = self.retriever.similarity_search(query, top_k=top_k,hybrid=hybrid)
        print(f"Number of Retrieved Docs: {len(retrieved_docs['documents'][0])}")
        # print(retrieved_docs)
        if iterative_retrival_k!=0 and max_iterations!=0:
            relevant_paragraphs = self.graph_reranker.iterative_retrieval(query, retrieved_docs, max_iterations=max_iterations, top_k=2)
            print(f"Number of Retrieved Paragraphs: {len(relevant_paragraphs)}")
            print('Paragraphs:\n\n',relevant_paragraphs,'\n\n\n\n')
        retrieved_docs['reranker']=[relevant_paragraphs]
        # Get the answer using OpenAIQA

        answer = self.openai_qa.query(query=query, documents=retrieved_docs)
        return answer


# Example Usage
if __name__ == "__main__":
    import os
    import dotenv
    dotenv.load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    pipeline = Pipeline(api_key)
    # Example texts to embed and store
    texts = [
        "This is the first document. I love playing Video Games",
        "This is the second document. I hate playing Video Games",
        "Here is another piece of text about documents. I feel neutral about video games",
    ]
    ids = ["doc1", "doc2", "doc3"]

    # Store embeddings in the Retriever
    pipeline.retriever.load_model()
    pipeline.retriever.embed_and_store(texts, ids)

    # Query the pipeline
    query = "Tell me about the first document."
    response = pipeline.process_query(query, top_k=2)
    print("Response:", response)