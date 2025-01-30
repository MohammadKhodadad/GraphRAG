# Function to create a keyword-document mapping
from transformers import pipeline
import re
import tqdm
import os
import openai
from openai import OpenAI
import json


class EntityExtractor:
    def __init__(self, model_name="pruas/BENT-PubMedBERT-NER-Chemical"):
        """Initialize the entity extractor with a specified model."""
        self.ner_pipeline = pipeline("ner", model=model_name, tokenizer=model_name)
    
    def extract_entities(self, query):
        """Use a transformer model to extract entities from the query."""
        # Clean the query by removing punctuation and converting to lowercase
        clean_query = re.sub(r'[^\w\s]', '', query)
        
        # Use the NER pipeline to extract entities
        ner_results = self.ner_pipeline(clean_query)
        entities = []
        current_entity = ""
        # print(ner_results)
        for result in ner_results:
            word = result['word']
            if result['entity'].startswith("B") and (word.startswith("##")):  # Begin a new entity
                current_entity += word[2:]  # Append to entity phrase
            elif result['entity'].startswith("B") and (not word.startswith("##")):  # Begin a new entity
                if current_entity:  # If an entity was being built, save it
                    entities.append(current_entity.lower())
                current_entity = word  # Start a new entity
            
            elif result['entity'].startswith("I") and current_entity:  # Continue entity
                current_entity += " " + word  # Append to entity phrase
            
            else:  # If we hit an "O" (outside entity), save the last entity
                if current_entity:
                    entities.append(current_entity.lower())
                    current_entity = ""  # Reset

        if current_entity:  # Save last entity if it wasn't stored
            entities.append(current_entity.lower())

        # Fallback to simple word extraction if no entities were found
        if not entities:
            entities = [word.lower() for word in clean_query.split() if word.isalpha()]
        
        return entities


def extract_relations(text, entities, api_key, model="gpt-4o", max_facts=20):
    client = OpenAI(api_key=api_key)
    prompt = f"""
        You are an expert in chemical text analysis. Your task is to extract **only chemically meaningful relationships** between a given set of entities from the provided text. 

        ### **Guidelines for Relation Extraction:**
        1. **Entity Matching:** Consider only the entities provided in the given set. If an entity appears in the text but has no meaningful chemical relationship with another entity in the set, ignore it.
        2. **Chemically Significant Relations Only:** Extract relations that describe actual **chemical interactions, transformations, or properties** (e.g., "reacts with," "catalyzes," "dissolves in," "produces").
        3. **Tuple Format:** Output extracted facts in the form of **(entity1, relation, entity2)**.
        4. **Avoid Generic Relations:** Exclude weak relations like "is," "are," "exists," "relates to." Focus on **specific interactions**.

        ### **Valid Relation Types (Examples)**
        ✅ "reacts with"  
        ✅ "catalyzes"  
        ✅ "binds to"  
        ✅ "dissolves in"  
        ✅ "oxidizes"  
        ✅ "inhibits"  
        ✅ "precipitates with"  
        ✅ "acts as a solvent for"  
        ✅ "is synthesized from"  

        🚫 **Avoid These Weak Relations** (e.g., "is," "are," "has," "exists")  

        #### **Entities Provided:**
        {entities}

        #### **Text:**
        {text}

        #### **Example Output:**
        [
            ("HCl", "dissolves in", "Water"),
            ("HCl", "reacts with", "Sodium hydroxide")
        ]

        Extract **at most {max_facts}** factual statements.

        Provide the output as a **Python list of tuples**, containing only the extracted relationships without any code formatting, backticks, or markdown.
        """
    response = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=0.2,
                model=model,
                max_tokens=500
            )
    return json.dumps(eval(response.choices[0].message.content))



if __name__=='__main__':
    import dotenv
    dotenv.load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")  # Ensure this is set in your environment
    extractor = EntityExtractor()
    text = "Aspirin (C9H8O4) is widely used as an anti-inflammatory drug. Acetic anhydride reacts with salicylic acid to form it."
    # text = "Aspirin and ibuprofen are common nonsteroidal anti-inflammatory drugs (NSAIDs)."
    extracted_entities = extractor.extract_entities(text)
    print("Extracted Entities:", extracted_entities)
    relations = extract_relations(text,extracted_entities,api_key)
    print("Extracted Relations:",relations)