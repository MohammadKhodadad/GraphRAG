from .wikipedia import wiki_fetch_combined_text
from .pubchem import pubchem_fetch_compound

def extract_meta_info(name):
    return {'pubchem':pubchem_fetch_compound(name),'wikipedia':wiki_fetch_combined_text(name)}
