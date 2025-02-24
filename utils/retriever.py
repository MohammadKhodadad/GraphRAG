from sentence_transformers import SentenceTransformer
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings

from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import QueryParser
from whoosh.scoring import BM25F
from whoosh.query import Or, Term
import os
import re
# import spacy
from transformers import pipeline
from sentence_transformers import CrossEncoder
from nltk.tokenize import sent_tokenize

class EmbeddingFunc(EmbeddingFunction):
    def __init__(self, model_name):
        super().__init__()
        self.model = SentenceTransformer(model_name)
    def __call__(self, input: Documents) -> Embeddings:
        embeddings = self.model.encode(input, convert_to_tensor=False).tolist() # ChromaDB does not work with np.array. Make it a list.
        return embeddings



class Retriever:
    def __init__(self, model_name: str = "all-mpnet-base-v2", chroma_dir: str = "chroma_db",
                  whoosh_dir: str = "whoosh_index", ner_model_name: str ="pruas/BENT-PubMedBERT-NER-Chemical",
                 cross_encoder_model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"):
        """
        Initialize the retriever class.
        :param model_name: Name of the sentence transformer model to use.
        :param chroma_dir: Directory where ChromaDB will store its data.
        """
        self.model_name = model_name
        self.chroma_dir = chroma_dir
        self.whoosh_dir = whoosh_dir
        self.model= None
        self.client= None
        self.collection = None
        self.whoosh_index = None
        self.ner_pipeline = pipeline("ner", model=ner_model_name)
        self.cross_encoder_model = CrossEncoder(cross_encoder_model_name)

    def load_model(self):
        """
        Load the embedding model and initialize ChromaDB.
        """
        self.client = chromadb.PersistentClient(path=self.chroma_dir)
        self.model= EmbeddingFunc(self.model_name)
        existing_collections = self.client.list_collections()
        if not 'docs' in [col for col in existing_collections]:
            print('"docs" does not exist. So, we will create it.')
            self.collection = self.client.create_collection( name="docs",embedding_function=self.model)
        else:
            print('"docs" already exists. So, we will load it.')
            self.collection = self.client.get_collection(name='docs',embedding_function=self.model)

        schema = Schema(id=ID(stored=True, unique=True), content=TEXT(stored=True))
        if not os.path.exists(self.whoosh_dir):
            os.makedirs(self.whoosh_dir)
            self.whoosh_index = create_in(self.whoosh_dir, schema)
        else:
            self.whoosh_index = open_dir(self.whoosh_dir)


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

        writer = self.whoosh_index.writer()
        for text, doc_id in zip(texts, ids):
            writer.add_document(id=doc_id, content=text)
        writer.commit()


    def extract_entities(self, query):
        """Use a transformer model to extract entities from the query."""
        
        # Use the NER pipeline to extract entities
        ner_results = self.ner_pipeline(query)
        # print(ner_results)
        entities = []
        current_entity = ""
        last_end = -1  # Track the last token's ending position

        for i, result in enumerate(ner_results):
            word = result['word']
            entity_type = result['entity']
            start = result.get('start', None)  # Get start position in original text
            end = result.get('end', None)  # Get end position in original text

            # Remove "##" from subwords (e.g., "##yl" → "yl")
            if word.startswith("##"):
                word = word[2:]

            # If it's the start of a new entity
            if entity_type.startswith("B"):
                if start==last_end:
                    current_entity += word 
                else:
                    if current_entity:
                        entities.append(current_entity.lower())  # Store previous entity
                    current_entity = word  # Start new entity
            
            # If it's a continuation (Inside)
            elif entity_type.startswith("I") and current_entity:
                if start == last_end:  
                    current_entity += word  # Merge directly if no space exists in the original text
                else:
                    current_entity += " " + word  # Add a space if there's a gap in the text
            
            # If it's an "O" tag (outside entity), save the entity
            else:
                if current_entity:
                    entities.append(current_entity.lower())
                    current_entity = ""

            # Update the last token end position
            last_end = end

        # Ensure last entity is stored
        if current_entity:
            entities.append(current_entity.lower())

        return entities

    
    def split_text(self, text, max_words=128):
        """Splits text into chunks by grouping sentences while keeping each chunk under `max_words`."""
        sentences = sent_tokenize(text)  # Tokenize into sentences
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence.split())

            if current_length + sentence_length > max_words:
                # Save the current chunk and start a new one
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                # Add the sentence to the current chunk
                current_chunk.append(sentence)
                current_length += sentence_length

        # Add any remaining text as the last chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def extract_paragraphs(self, documents, max_words=128):
        """
        Extracts and improves paragraphization of documents by splitting text into controlled chunks.
        
        Args:
            documents (dict): A dictionary containing a 'documents' key with a list of text documents.
            max_words (int): Maximum number of words allowed per paragraph.
        
        Returns:
            list: A list of paragraphs extracted from the documents.
        """
        if 'documents' not in documents:
            raise ValueError("The input dictionary must contain a 'documents' key.")
        
        paragraphized_documents = []
        for doc in documents['documents'][0]:
            if not isinstance(doc, str):
                # print(f"Skipping invalid document: {doc}")
                continue
            paragraphs = self.split_text(doc, max_words=max_words)
            paragraphized_documents.extend(paragraphs)
        
        return paragraphized_documents

    def cross_encoder_similarity_search(self, query, documents, top_k=1):
        """
        Perform similarity search using the cross-encoder model.
        """
        paragraphs = self.extract_paragraphs(documents)
        text_pairs = [(query, paragraph) for paragraph in paragraphs]
        scores=[self.cross_encoder_model.predict([text_pairs[i]])[0] for i in range(len(text_pairs))]
        scored_paragraphs = sorted(zip(paragraphs, scores), key=lambda x: x[1], reverse=True)
        return scored_paragraphs[:top_k]



    def similarity_search(self, query: str, top_k: int = 5, hybrid: bool=True, reranker:bool =True, reranker_top_k: int=3):
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
        old_count=len(results['ids'][0])
        if hybrid:
            with self.whoosh_index.searcher(weighting=BM25F()) as searcher:
                entities = self.extract_entities(query)
                if not entities:
                    # Use the full query text if no entities are detected
                    whoosh_query = QueryParser("content", self.whoosh_index.schema).parse(query)
                else:
                    whoosh_query = Or([Term("content", entity) for entity in entities])
                print('WHOOSH QUERY: ',whoosh_query)
                results_lexical = searcher.search(whoosh_query, limit=top_k)
                results_lexical = [(hit["id"], hit["content"], hit.score) for hit in results_lexical]
            # print(results_lexical)
            for item in results_lexical:
                if item[0] not in results['ids'][0]:
                    results['documents'][0].append(item[1])
                    results['ids'][0].append(item[0])
            new_count=len(results['ids'][0])
            print('Whoosh added:',new_count-old_count, f' Whoosh found {old_count}')
        if reranker:
            reranker_results= self.cross_encoder_similarity_search(query, results, top_k=reranker_top_k)
            results['reranker']=[res[0] for res in reranker_results]
            results['reranker_score']=[res[1] for res in reranker_results]
        return results


# Example Usage
if __name__ == "__main__":
    retriever = Retriever()
    retriever.load_model()

    # Example texts to embed and store
    texts = [
        "Aspirin (acetylsalicylic acid) is commonly used as an anti-inflammatory drug.",
        "Methanol (CH3OH) is a simple alcohol that is highly toxic to humans.",
        "The reaction between sodium chloride (NaCl) and silver nitrate (AgNO3) forms a white precipitate of silver chloride.",
        "Benzene is an aromatic hydrocarbon with a six-membered ring structure.",
        "Sulfuric acid (H2SO4) is a strong acid used in industrial chemical synthesis.",
        "DNA is composed of nucleotides that include adenine, thymine, cytosine, and guanine.",
        "Photosynthesis converts carbon dioxide and water into glucose and oxygen using sunlight.",
        "The Haber process synthesizes ammonia (NH3) from nitrogen and hydrogen under high pressure.",
        "Acetic acid (CH3COOH) is responsible for the sour taste of vinegar.",
        "Ethanol (C2H5OH) is a widely used solvent and the main component of alcoholic beverages.",
        "Polyethylene (PE) is the most commonly produced plastic, made from polymerization of ethylene.",
        "Catalysts like platinum and palladium are used in automotive catalytic converters to reduce emissions.",
        "Hydrochloric acid (HCl) is a strong acid used in laboratories and the production of PVC.",
        "Enzymes act as biological catalysts that speed up chemical reactions in living organisms.",
        "Ozone (O3) is a reactive gas that protects Earth from UV radiation in the stratosphere."
    ]

    ids = [
        "doc_aspirin",
        "doc_methanol",
        "doc_nacl_agno3",
        "doc_benzene",
        "doc_sulfuric_acid",
        "doc_dna",
        "doc_photosynthesis",
        "doc_haber_process",
        "doc_acetic_acid",
        "doc_ethanol",
        "doc_polyethylene",
        "doc_catalyst",
        "doc_hcl",
        "doc_enzymes",
        "doc_ozone"
]
    # Store embeddings
    retriever.embed_and_store(texts, ids)

    # Perform a similarity search
    query = "acetylsalicylic acid"
    results = retriever.similarity_search(query, top_k=4)
    print(results)
