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
        
        # Use the NER pipeline to extract entities
        ner_results = self.ner_pipeline(query)
        print(ner_results)
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
    return eval(response.choices[0].message.content)


def extract_entity_descriptions(text, entities, api_key, model="gpt-4o", max_descriptions=20):
    client = OpenAI(api_key=api_key)
    prompt = f"""
        You are an expert in chemical text analysis. Your task is to extract **concise yet informative descriptions** of a given set of entities from the provided text. 

        ### **Guidelines for Entity Description Extraction:**
        1. **Entity Matching:** Extract descriptions **only** for the entities listed in the provided set. Ignore any entity that is not explicitly mentioned in the set.
        2. **Concise & Relevant Descriptions:** Each description should be **factual, chemically relevant, and no longer than a sentence or two**. Avoid unnecessary explanations.
        3. **Meaningful Chemical Properties:** Focus on essential chemical **properties, behaviors, or roles** (e.g., acidity, solubility, reactivity, catalytic function).
        4. **Tuple Format:** Output extracted facts as a Python list of tuples in the form of **(entity, description)**.
        5. **Avoid General/Vague Information:** Descriptions should be precise and **chemically informative** rather than generic (e.g., "is a compound" is too weak).
        6. **Only Output from the Given Text:** Descriptions have to come from the text. If there is no description for an entity, don't output anything for that entity.
        
        ### **Examples of Valid Descriptions:**
        ✅ ("HCl", "A strong acid that ionizes completely in water.")  
        ✅ ("Sodium hydroxide", "A strong base that reacts with acids to form salts and water.")  
        ✅ ("Ethanol", "A polar solvent commonly used in organic synthesis and pharmaceuticals.")  
        ✅ ("Copper sulfate", "A blue crystalline compound used as a fungicide and electrolyte.")  
        
        🚫 **Avoid These Weak Descriptions:**
        ❌ ("HCl", "A chemical compound.") → Too vague  
        ❌ ("Sodium hydroxide", "A substance used in labs.") → Lacks chemical detail  

        #### **Entities Provided:**
        {entities}

        #### **Text:**
        {text}

        #### **Example Output:**
        [
            ("HCl", "A strong acid that ionizes completely in water."),
            ("Sodium hydroxide", "A strong base that reacts with acids to form salts and water.")
        ]

        Extract **at most {max_descriptions}** descriptions.

        Provide the output as a **Python list of tuples**, containing only the extracted entity descriptions without any code formatting, backticks, or markdown.
        """
    response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                model=model,
                max_tokens=500
            )
    return eval(response.choices[0].message.content)



if __name__=='__main__':
    import dotenv
    dotenv.load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")  # Ensure this is set in your environment
    extractor = EntityExtractor()
    text = "Aspirin (C9H8O4) is widely used as an anti-inflammatory drug. Acetic anhydride reacts with salicylic acid to form it."
    # text = "Aspirin and ibuprofen are common nonsteroidal anti-inflammatory drugs (NSAIDs)."
    text = '2-Benzyl-1-(4-methylphenyl)-3-(4-prop-2-enoxyphenyl)guanidine.'
    # text = 'N,N-Dimethylformamide'
    extracted_entities = extractor.extract_entities(text)
    print("Extracted Entities:", extracted_entities)
    relations = extract_relations(text,extracted_entities,api_key)
    print("Extracted Relations:",relations)
    descriptions= extract_entity_descriptions(text,extracted_entities,api_key)
    print('Extracted Descriptions:', descriptions)