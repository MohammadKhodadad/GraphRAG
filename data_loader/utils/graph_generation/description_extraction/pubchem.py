import requests
import tqdm
import json
from collections import deque
import pandas as pd
import time
import os

# Function to fetch and parse the JSON file
import requests
import time

def fetch_compound_json(cid):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON/"
    retries = 3  # Number of retries
    delay = 2  # Delay in seconds between retries

    for attempt in range(retries):
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)
            data = response.json()  # Parse the JSON data
            return data
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(delay)
    
    print("All retry attempts failed.")
    return {}

def extract_name(json_data):
    name=json_data.get('Record',{}).get('RecordTitle',None)
    return name

def extract_name_and_identifier(sections):
    temp=[section for section in sections if section.get("TOCHeading",None)=='Names and Identifiers']
    if len(temp)>0:
        names_and_identifiers=temp[0].get('Section',{})
    else:
        names_and_identifiers={}
    return names_and_identifiers

def extract_record_description(names_and_identifiers):
    temp=[section for section in names_and_identifiers if section.get("TOCHeading",None)=='Record Description']
    if len(temp)>0:
        record_description=temp[0].get('Information',[])
    else:
        record_description=[]
    return record_description

def extract_description(record_description,cid=None):
    text=""
    mentioned_entities=[]
    for record in record_description:
        value=record.get('Value',{})
        string_with_markup=value.get('StringWithMarkup',[])
        for string_with_markup_case in string_with_markup:
            string=string_with_markup_case.get('String','')
            if string:
                text=text+'\n'+string
            markup=string_with_markup_case.get('Markup',[])
            for markup_item in markup:
                extra=markup_item.get('Extra','')
                if extra:
                    if 'CID-' in extra:
                        extra=extra.replace('CID-','')
                        if (extra!=str(cid)) or (not cid):
                            mentioned_entities.append(extra)
    return text,list(set(mentioned_entities))
def extract_safety(sections):
    safety=[]
    # print( [section.get("TOCHeading",'None') for section in sections])
    temp=[section for section in sections if section.get("TOCHeading",None)=='Chemical Safety']
    if len(temp)>0:
        chemical_safety=temp[0]
    else:
        chemical_safety={}
    chemical_safety_info=chemical_safety.get('Information',[])
    if len(chemical_safety_info)>0:
        string_with_markup=chemical_safety_info[0].get('Value',{}).get('StringWithMarkup',[])
    else:
        string_with_markup=[]
    for string_with_markup_case in string_with_markup:
        markup=string_with_markup_case.get('Markup',[])
        for markup_item in markup:
            extra=markup_item.get('Extra','')
            if extra:
                safety.append(extra)
    return ' and '.join(safety)

def extract_smiles(names_and_identifiers):
    temp=[section for section in names_and_identifiers if section.get("TOCHeading",None)=='Computed Descriptors']
    if len(temp)>0:
        computed_discriptors=temp[0].get('Section',[])
    else:
        computed_discriptors=[]
    temp=[section for section in computed_discriptors if section.get("TOCHeading",None)=='SMILES']
    if len(temp)>0:
        smiles=temp[0].get('Information',[])
    else:
        smiles=[]
    if len(smiles)>0:
        string_with_markup= smiles[0].get('Value',{}).get('StringWithMarkup',[])
    else:
        string_with_markup=[]
    if len(string_with_markup)>0:
        smiles_string=string_with_markup[0].get('String','')
    else:
        smiles_string=''
    return smiles_string

def extract_formula(names_and_identifiers):
    temp=[section for section in names_and_identifiers if section.get("TOCHeading",None)=="Molecular Formula"]
    if len(temp)>0:
        molecular_formula=temp[0].get('Information',[])
    else:
        molecular_formula=[]
    if len(molecular_formula)>0:
        string_with_markup=molecular_formula[0].get('Value',{}).get('StringWithMarkup',[])
    else:
        string_with_markup=[]
    if len(string_with_markup)>0:
        formula_string=string_with_markup[0].get('String','')
    else:
        formula_string=''
    return formula_string

def chemical_properties(sections):
    chem_prot=''
    temp=[section for section in sections if section.get("TOCHeading",None)=='Chemical and Physical Properties']
    if len(temp)>0:
        chemical_and_physical_properties=temp[0].get('Section',[])
    else:
        chemical_and_physical_properties=[]
    temp=[section for section in chemical_and_physical_properties if section.get("TOCHeading",None)=='Computed Properties']
    if len(temp)>0:
        computed_properties=temp[0].get('Section',[])
    else:
        computed_properties=[]
    for prop in computed_properties:
        chem_prot+=(prop.get('TOCHeading','Unknown')+': ')
        information=prop.get("Information",[])
        if len(information)>0:
            value=information[0].get('Value',{})
            number=value.get('Number',[''])
            if len(number)>0:
                chem_prot+=str(number[0])
            string_with_markup = value.get('StringWithMarkup',[])
            if len(string_with_markup)>0:
                number_string=string_with_markup[0].get('String','')
                chem_prot+=(number_string+' ')
                chem_prot+=value.get("Unit",'')
        chem_prot+='\n'
    return chem_prot
            

        
def extract_properties(sections):
    safety = extract_safety(sections)
    names_and_identifiers = extract_name_and_identifier(sections)
    smiles = extract_smiles(names_and_identifiers)
    formula=extract_formula(names_and_identifiers)
    chem_properties=chemical_properties(sections)
    if safety:
        properties=f'safety: {safety}\n'
    if smiles:
        properties+=f'smiles: {smiles}\n'
    if formula:
        properties+=f'formula: {formula}\n'
    if chem_properties:
        properties+=f'chem_properties: {chem_properties}\n'

    return properties

def extract_pubchem_data(json_data,cid=None):
    name = extract_name(json_data)
    sections=json_data.get('Record',{}).get('Section',{})
    names_and_identifiers = extract_name_and_identifier(sections)
    record_description = extract_record_description(names_and_identifiers)
    text,mentioned_entities=extract_description(record_description,cid)
    properties = extract_properties(sections)
    return mentioned_entities,text,name, properties

def get_cid_from_name(chemical_name):
    """Fetches CID (Compound ID) from PubChem using a chemical name."""
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{chemical_name}/cids/JSON"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        cids = data.get("IdentifierList", {}).get("CID", [])
        return cids[0] if cids else None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching CID for {chemical_name}: {e}")
        return None

def pubchem_fetch_compound(id_or_name):
    if isinstance(id_or_name,int):
        cid= id_or_name
    elif isinstance(id_or_name,str):
        cid = get_cid_from_name(id_or_name)
    else:
        return ""
    if cid:
        json_data = fetch_compound_json(cid)
        mentioned_entities,text,name, properties  = extract_pubchem_data(json_data,cid)
        return f"Name: {name}\nText:{text}\nProperties:{properties}"   
    else:
        return ''     



if __name__=='__main__':
    print(pubchem_fetch_compound('aspirin'))