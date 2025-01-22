if __name__ == "__main__":
    
    from llm import OpenAIQA
    from cot import ChainOfThoughts
    from retriever import Retriever
    from graph_reranker import GraphReranker
else:
    from .llm import OpenAIQA
    from .cot import ChainOfThoughts
    from .retriever import Retriever
    from .graph_reranker import GraphReranker



class Pipeline:
    def __init__(self,api_key):
        """
        Initialize the Pipeline with an ChainOfThoughs object and a Retriever object.

        Parameters:
        - cot (OpenAIQA): The Chain of Thoughts object for question answering.
        - retriever (Retriever): The Retriever object for document retrieval.
        """
        retriever_model = Retriever()
        retriever_model.load_model()
        self.cot = ChainOfThoughts(api_key,retriever=retriever_model)

    def process_query(self,query: str ,max_iterations:int=5, top_k: int = 5, hybrid: bool=True, reranker:bool =True, reranker_top_k: int=3):
        """
        Process the query by retrieving relevant documents and getting an answer.

        Parameters:
        - query (str): The user's question.
        - top_k (int): The number of top documents to retrieve.

        Returns:
        - str: The answer to the query.
        """
        answer = self.cot.chain_of_thoughts(query=query,max_iterations=max_iterations,top_k=top_k,hybrid=hybrid,reranker=reranker,reranker_top_k=reranker_top_k)
        return answer['answer']


# Example Usage
if __name__ == "__main__":
    import os
    import dotenv
    dotenv.load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    pipeline = Pipeline(api_key)
    # Example texts to embed and store
    # texts = [
    #     "Aspirin (acetylsalicylic acid) is commonly used as an anti-inflammatory drug.",
    #     "Methanol (CH3OH) is a simple alcohol that is highly toxic to humans.",
    #     "The reaction between sodium chloride (NaCl) and silver nitrate (AgNO3) forms a white precipitate of silver chloride.",
    #     "Benzene is an aromatic hydrocarbon with a six-membered ring structure.",
    #     "Sulfuric acid (H2SO4) is a strong acid used in industrial chemical synthesis.",
    #     "DNA is composed of nucleotides that include adenine, thymine, cytosine, and guanine.",
    #     "Photosynthesis converts carbon dioxide and water into glucose and oxygen using sunlight.",
    #     "The Haber process synthesizes ammonia (NH3) from nitrogen and hydrogen under high pressure.",
    #     "Acetic acid (CH3COOH) is responsible for the sour taste of vinegar.",
    #     "Ethanol (C2H5OH) is a widely used solvent and the main component of alcoholic beverages.",
    #     "Polyethylene (PE) is the most commonly produced plastic, made from polymerization of ethylene.",
    #     "Catalysts like platinum and palladium are used in automotive catalytic converters to reduce emissions.",
    #     "Hydrochloric acid (HCl) is a strong acid used in laboratories and the production of PVC.",
    #     "Enzymes act as biological catalysts that speed up chemical reactions in living organisms.",
    #     "Ozone (O3) is a reactive gas that protects Earth from UV radiation in the stratosphere."
    # ]

    # ids = [
    #     "doc_aspirin",
    #     "doc_methanol",
    #     "doc_nacl_agno3",
    #     "doc_benzene",
    #     "doc_sulfuric_acid",
    #     "doc_dna",
    #     "doc_photosynthesis",
    #     "doc_haber_process",
    #     "doc_acetic_acid",
    #     "doc_ethanol",
    #     "doc_polyethylene",
    #     "doc_catalyst",
    #     "doc_hcl",
    #     "doc_enzymes",
    #     "doc_ozone"
    # ]

    # Store embeddings in the Retriever
    # pipeline.retriever.embed_and_store(texts, ids)

    # Query the pipeline
    query = 'What is the Haber process?'
    response = pipeline.process_query(query, top_k=2)
    print("Response:", response)