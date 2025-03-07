# Function to create a keyword-document mapping
from transformers import pipeline
import re
import tqdm
import os
import openai
import dotenv
from openai import OpenAI
import json
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from openai import OpenAI
from pubchempy import get_compounds
from chemspipy import ChemSpider



class EntityExtractor:
    def __init__(self, model_name="pruas/BENT-PubMedBERT-NER-Chemical"):
        """Initialize the entity extractor with a specified model."""
        self.ner_pipeline = pipeline("ner", model=model_name, tokenizer=model_name)
    
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



def verify_entities_from_text(text, entities, api_key, model="gpt-4o"):
    """
    Extract valid chemical entities from text after NER and filter incorrect ones.

    Args:
        text (str): The input text containing chemical information.
        entities (list): List of extracted entities from an NER model.
        api_key (str): OpenAI API key.
        model (str): OpenAI model to use.

    Returns:
        list: Filtered list of correct chemical entities.
    """
    client = OpenAI(api_key=api_key)

    # Validate extracted entities
    filtered_entities = entities # validate_chemical_entities(entities)

    if not filtered_entities:
        return []

    prompt = f"""
You are a chemistry expert specializing in entity recognition. Your task is to **validate and filter** the extracted entities, ensuring they are **chemically meaningful** based on the provided text. Remove any irrelevant terms, including general descriptors, numerical values, reaction conditions, and vague terms.

### **Entities Extracted by NER:**  
{entities}

### **Text for Context:**  
{text}

### **Criteria for Valid Entities:**  
✅ Chemical compounds (e.g., "HCl", "Sodium hydroxide", "Ethanol", "Benzene")  
✅ Chemical elements (e.g., "Carbon", "Oxygen", "Cesium")  
✅ Specific catalysts, solvents, reagents (e.g., "Cs₂CO₃", "Toluene", "Palladium")  

### **Remove the Following Types of Entities:**  
🚫 Generic terms (e.g., "Reaction", "Solvent", "Acid", "Base", "Solution")  
🚫 Experimental conditions (e.g., "pH", "Temperature", "2 M", "Strong acid")  
🚫 Measurement terms (e.g., "X-ray diffraction", "NMR")  
🚫 General descriptors (e.g., "High concentration", "Low efficiency")  

### **Output Format:**  
Return only a **Python list** of valid chemical entities, with no explanations, markdown, or extra formatting.
    """


    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        model=model,
        max_tokens=300
    )

    return eval(response.choices[0].message.content)

def extract_relations(text, entities, api_key, model="gpt-4o", max_facts=20):
    client = OpenAI(api_key=api_key)
    prompt = f"""
        You are an expert in chemical text analysis. Your task is to extract **only chemically meaningful relationships** between a given set of entities from the provided text. 

        ### **Guidelines for Relation Extraction:**
        1. **Entity Matching:** Consider only the entities provided in the given set. If an entity appears in the text but has no meaningful chemical relationship with another entity in the set, ignore it.
        2. **Chemically Significant Relations Only:** Extract relations that describe actual **chemical interactions, transformations, or properties** (e.g., "reacts with," "catalyzes," "dissolves in," "produces").
        3. **Factual Relations** Only extract factual relations. Avoid observations, opinions, and findings.
        4. **Tuple Format:** Output extracted facts in the form of **(entity1, relation, entity2)**.
        5. **Avoid Generic Relations:** Exclude weak relations like "is," "are," "exists," "relates to." Focus on **specific interactions**.

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
        3. **Factual Descriptions** Only extract factual descriptions. Avoid observations, opinions, and findings.
        5. **Tuple Format:** Output extracted facts as a Python list of tuples in the form of **(entity, description)**.
        6. **Avoid General/Vague Information:** Descriptions should be precise and **chemically informative** rather than generic (e.g., "is a compound" is too weak).
        7. **Only Output from the Given Text:** Descriptions have to come from the text. If there is no description for an entity, don't output anything for that entity.
        
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
#     extractor = EntityExtractor()
#     text = """Cesium carbonate (Cs₂CO₃) is a widely used inorganic base in organic synthesis. It dissolves in water and is often used as a mild base in various catalytic reactions. In Suzuki coupling reactions, Cs₂CO₃ acts as a catalyst by facilitating the deprotonation of boronic acids. However, its efficiency is questionable, as many researchers prefer stronger bases like potassium tert-butoxide (t-BuOK).

# Additionally, Cs₂CO₃ precipitates at concentrations above 8 M, limiting its application in high-concentration reactions. While some scientists believe it is an inferior catalyst compared to organic bases, others argue that its solubility advantages outweigh its lower catalytic efficiency.

# Moreover, Cs₂CO₃ is a better choice than K₂CO₃ in reactions that require higher solubility in polar solvents. However, in my experience, reactions catalyzed by Cs₂CO₃ take much longer to complete, making it an impractical choice for time-sensitive experiments.

# Interestingly, a recent study found that Cs₂CO₃ accelerates some esterification reactions in dimethyl sulfoxide (DMSO), but its role as a catalyst in these systems is still debated. Some researchers claim that Cs₂CO₃ plays no direct role and that the solvent itself may be responsible for the acceleration."""
#     # entities = ["HCl", "Sodium hydroxide", "Reaction", "Solution", "Water"]
#     extracted_entities = extractor.extract_entities(text)
#     print("Extracted Entities:", extracted_entities)
#     filtered_entities = verify_entities_from_text(text, extracted_entities, api_key)
#     print("Verified Entities:", extracted_entities)
#     relations = extract_relations(text,extracted_entities,api_key)
#     print("Extracted Relations:",relations)
#     descriptions= extract_entity_descriptions(text,extracted_entities,api_key)
#     print('Extracted Descriptions:', descriptions)