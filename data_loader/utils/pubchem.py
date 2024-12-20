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



def extract_edges_and_text(json_data,cid=None):
    text=""
    edges=[]
    sections=json_data.get('Record',{}).get('Section',{})
    name=json_data.get('Record',{}).get('RecordTitle',None)
    temp=[section for section in sections if section.get("TOCHeading",None)=='Names and Identifiers']
    if len(temp)>0:
        names_and_identifiers=temp[0].get('Section',{})
    else:
        names_and_identifiers={}
    temp=[section for section in names_and_identifiers if section.get("TOCHeading",None)=='Record Description']
    if len(temp)>0:
        record_description=temp[0].get('Information',[])
    else:
        record_description=[]

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
                            edges.append(extra)
    return list(set(edges)),text,name


def fetch_compound(cid):
    json_data = fetch_compound_json(cid)
    edges,text,name = extract_edges_and_text(json_data,cid)
    return edges,text,name            

def download_and_store_pubchem(address='pubchem_dump.csv'):
    cids=[]
    texts=[]
    edges=[]
    names=[]
    start=1
    if os.path.exists(address):
        data=pd.read_csv(address)
        cids=list(data.cid)
        texts=list(data.text)
        edges=list(data.edge)
        names=list(data.name)
        start=cids[-1]+1
    print(f"starting from {start}")
    for i in tqdm.tqdm(range(start,50001)):
        if i%10==0:
            time.sleep(0.1)
        edge,text,name = fetch_compound(i)
        cids.append(i)
        texts.append(text)
        edges.append(json.dumps(edge))
        names.append(name)
        if i%1000==0 and i>0:
            pd.DataFrame({'text':texts,'edge':edges,'cid':cids,'name':names}).to_csv(address)

if __name__=='__main__':
    fetch_compound(22)