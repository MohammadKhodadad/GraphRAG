from sentence_transformers import SentenceTransformer
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings



class EmbeddingFunc(EmbeddingFunction):
    def __init__(self, model_name):
        super().__init__()
        self.model = SentenceTransformer(model_name)
    def __call__(self, input: Documents) -> Embeddings:
        embeddings = self.model.encode(input, convert_to_tensor=False).tolist() # ChromaDB does not work with np.array. Make it a list.
        return embeddings



class Retriever:
    def __init__(self, model_name: str = "all-mpnet-base-v2", chroma_dir: str = "chroma_db"):
        """
        Initialize the retriever class.
        :param model_name: Name of the sentence transformer model to use.
        :param chroma_dir: Directory where ChromaDB will store its data.
        """
        self.model_name = model_name
        self.chroma_dir = chroma_dir
        self.model= None
        self.client = None
        self.collection = None

    def load_model(self):
        """
        Load the embedding model and initialize ChromaDB.
        """
        self.client = chromadb.PersistentClient(path=self.chroma_dir)
        self.model= EmbeddingFunc(self.model_name)
        existing_collections = self.client.list_collections()
        if not 'docs' in [col.name for col in existing_collections]:
            print('"docs" does not exist. So, we will create it.')
            self.collection = self.client.create_collection( name="docs",embedding_function=self.model)
        else:
            print('"docs" already exists. So, we will load it.')
            self.collection = self.client.get_collection(name='docs',embedding_function=self.model)

    def embed_and_store(self, texts: list, ids: list):
        """
        Embed the input texts and store them in ChromaDB with the given IDs.
        :param texts: List of text documents to embed.
        :param ids: List of unique IDs corresponding to the texts.
        """
        if not self.model:
            raise ValueError("Model is not loaded. Call `load_model()` first.")
        if len(texts) != len(ids):
            raise ValueError("Texts and IDs must have the same length.")
        self.collection.add(documents=texts, ids=ids)

    def similarity_search(self, query: str, top_k: int = 5):
        """
        Perform similarity search for the query text and return the top-k results.
        :param query: Input query text.
        :param top_k: Number of top similar results to return.
        :return: List of tuples (id, text, similarity_score).
        """
        if not self.model:
            raise ValueError("Model is not loaded. Call `load_model()` first.")
        
        results = self.collection.query(
            query_texts=[query], # Chroma will embed this for you
            n_results=top_k # how many results to return
        )
        return results


# Example Usage
if __name__ == "__main__":
    retriever = Retriever()
    retriever.load_model()

    # Example texts to embed and store
    texts = [
        "This is the first document.",
        "This is the second document.",
        "Here is another piece of text."
    ]
    ids = ["doc11", "doc12", "doc13"]

    # Store embeddings
    retriever.embed_and_store(texts, ids)

    # Perform a similarity search
    query = "first document"
    results = retriever.similarity_search(query, top_k=2)
    print(results)
