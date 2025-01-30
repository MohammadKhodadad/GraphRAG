import os
import sys
import tqdm
import json
# Ensure the script runs correctly when executed directly
if __name__ == "__main__" and __package__ is None:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from entity_extraction import EntityExtractor, extract_relations, extract_entity_descriptions
    from graph import GraphManager
    from text_extraction import extract_text_from_pdf, clean_text, split_text
else:
    from .entity_extraction import EntityExtractor, extract_relations, extract_entity_descriptions
    from .graph import GraphManager
    from .text_extraction import extract_text_from_pdf, clean_text, split_text

def graph_pipeline(directory, graph_directory, api_key):
    G=GraphManager()
    entity_extractor = EntityExtractor()
    names=os.listdir(directory)
    for name in tqdm.tqdm(names):
        
        address=os.path.join(directory,name)
        try:
            text=extract_text_from_pdf(address)
            cleaned_text= clean_text(text)
        except Exception as e:
            print(f'Error: {e}')
            continue
    
        chunks = split_text(cleaned_text,max_words=128)
        print(f'Num chunks: {len(chunks)}')
        for chunk in chunks[:16]:
            try:
                extracted_entities = entity_extractor.extract_entities(chunk)
                descriptions = extract_entity_descriptions(chunk, extracted_entities, api_key)                
                relations = extract_relations(chunk, extracted_entities, api_key)
                for entity1, relation, entity2 in relations:
                    G.add_node(entity1, name, "")
                    G.add_node(entity2, name, "")
                    G.add_edge(entity1, entity2, name, relation)
            except Exception as e:
                print('Error:',e)
                print(chunk)
        # except Exception as e:
        #     print(e)
        G.save_graph(graph_directory)

# Example Usage
if __name__ == "__main__":
    import dotenv
    dotenv.load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    graph_pipeline("D:\jobs\Jobs\BASF\RAG\GraphRAG\data_loader\data\chemrxiv_papers", 'graph.json', api_key)
