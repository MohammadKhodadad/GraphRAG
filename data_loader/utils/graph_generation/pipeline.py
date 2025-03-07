import os
import sys
import tqdm
import json
import pandas as pd
# Ensure the script runs correctly when executed directly
if __name__ == "__main__" and __package__ is None:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from entity_extraction import EntityExtractor, extract_relations, extract_entity_descriptions, verify_entities_from_text
    from graph_manager import GraphManager
    from text_extraction import extract_text_from_pdf, clean_text, split_text, extract_introduction_with_limit
    from graph_explorer import GraphExplorer
    from description_extraction.wikipedia import wiki_fetch_combined_text
    from description_extraction.pubchem import pubchem_fetch_compound
else:
    from .entity_extraction import EntityExtractor, extract_relations, extract_entity_descriptions, verify_entities_from_text
    from .graph_manager import GraphManager
    from .text_extraction import extract_text_from_pdf, clean_text, split_text, extract_introduction_with_limit
    from .graph_explorer import GraphExplorer
    from .description_extraction.extractor import extract_meta_info

def graph_pipeline(directory, graph_directory, api_key):
    G=GraphManager()
    if os.path.exists(graph_directory):
        G.load_graph(graph_directory)
    entity_extractor = EntityExtractor()
    names=os.listdir(directory)
    for name in tqdm.tqdm(names):
        
        address=os.path.join(directory,name)
        try:
            if '.pdf' in address:
                text=extract_text_from_pdf(address)
            elif '.jsonl' in address:
                raise Exception('Not Implemented.')
            elif '.csv' in address:
                raise Exception('Not Implemented.')
            else:
                raise Exception('No supported input.')
            cleaned_text= extract_introduction_with_limit(clean_text(text),2000)
        except Exception as e:
            print(f'Error: {e}')
            continue
    
        chunks = split_text(cleaned_text,max_words=128)
        print(f'Num chunks: {len(chunks)}')
        for chunk in chunks[:16]:
            try:
                extracted_entities = entity_extractor.extract_entities(chunk)
                verified_entities = verify_entities_from_text(chunk ,extracted_entities, api_key)
                descriptions = extract_entity_descriptions(chunk, verified_entities, api_key)                
                relations = extract_relations(chunk, verified_entities, api_key)
                for entity, description in descriptions:
                    G.add_node(entity, name, description, chunk, meta_description = extract_meta_info(entity))
                for entity1, relation, entity2 in relations:
                    G.add_node(entity1, name, "",  meta_description = extract_meta_info(entity1))
                    G.add_node(entity2, name, "", meta_description = extract_meta_info(entity2))
                    G.add_edge(entity1, entity2, name, relation, chunk)
            except Exception as e:
                print('Error:',e)
                print(chunk)
        # except Exception as e:
        #     print(e)
        G.save_graph(graph_directory)


def graph_pipeline_from_csv(file_address, graph_directory, api_key,source_column='source',text_column='text'):
    G=GraphManager()
    if os.path.exists(graph_directory):
        G.load_graph(graph_directory)
    entity_extractor = EntityExtractor()
    data = pd.read_csv(file_address)
    print('Num Records: ',len(data))
    data=data.iloc[:1000]
    for index, row in tqdm.tqdm(data.iterrows()):
        try:
            cleaned_text= clean_text(row[text_column])
            name = row[source_column]
        except Exception as e:
            print(f'Error: {e}')
            break
    
        chunks = split_text(cleaned_text,max_words=128)
        print(f'Num chunks: {len(chunks)}')
        for chunk in chunks:
            try:
                extracted_entities = entity_extractor.extract_entities(chunk)
                verified_entities = verify_entities_from_text(chunk ,extracted_entities, api_key)
                descriptions = extract_entity_descriptions(chunk, verified_entities, api_key)                
                relations = extract_relations(chunk, verified_entities, api_key)
                for entity, description in descriptions:
                    G.add_node(entity, name, description, chunk, meta_description = extract_meta_info(entity))
                for entity1, relation, entity2 in relations:
                    G.add_node(entity1, name, "", meta_description = extract_meta_info(entity))
                    G.add_node(entity2, name, "", meta_description = extract_meta_info(entity))
                    G.add_edge(entity1, entity2, name, relation, chunk)
            except Exception as e:
                print('Error:',e)
                print(chunk)
        # except Exception as e:
        #     print(e)
        if index%10==0:
            G.save_graph(graph_directory)


def sample_graph_pipeline(graph_directory,sample_legnths = {2:2, 3:2 }, api_key= None):
    G=GraphManager()
    G.load_graph(graph_directory)
    E = GraphExplorer(G)
    samples={}
    for key, value in sample_legnths.items():
        samples[key]= E.sample_random_paths(key, value)
        # for path in samples[key]:
        #     E.display_path(path)
    
    return samples
# Example Usage
if __name__ == "__main__":
    import dotenv
    dotenv.load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    graph_pipeline("D:\jobs\Jobs\BASF\RAG\GraphRAG\data_loader\data\chemrxiv_papers", 'graph.json', api_key)
