import re
import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
    
def process_document(doc, pattern, terms, doc_idx):
    """Process a single document to find matching terms."""
    matches = re.findall(pattern, doc, re.IGNORECASE)
    matched_terms = set(term.lower() for term in matches)
    doc_result = {term: [] for term in terms}
    for term in terms:
        if term.lower() in matched_terms:
            doc_result[term].append(doc_idx)  # Store the index of the document
    return doc_result

def keyword_document_mapping(documents, terms, num_threads=16):
    # Sort terms by length to prioritize longer terms
    sorted_terms = sorted(terms, key=len, reverse=True)
    
    # Create a regex pattern for exact matching
    pattern = r'\b(' + '|'.join(re.escape(term) for term in sorted_terms) + r')\b'
    
    # Initialize the dictionary to store results
    keyword_to_documents = {term: [] for term in terms}
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_doc = {
            executor.submit(process_document, doc, pattern, terms, idx): idx
            for idx, doc in enumerate(documents)
        }
        for future in tqdm.tqdm(as_completed(future_to_doc), total=len(documents)):
            result = future.result()
            for term, doc_indices in result.items():
                keyword_to_documents[term].extend(doc_indices)

    return keyword_to_documents

def keyword_document_mapping_old(documents, terms):
    # Sort terms by length to prioritize longer terms
    sorted_terms = sorted(terms, key=len, reverse=True)
    
    # Create a regex pattern for exact matching
    pattern = r'\b(' + '|'.join(re.escape(term) for term in sorted_terms) + r')\b'
    
    # Initialize the dictionary to store results
    keyword_to_documents = {term: [] for term in terms}
    
    # Process each document
    for idx, doc in tqdm.tqdm(enumerate(documents)):
        matches = re.findall(pattern, doc, re.IGNORECASE)
        # Normalize matches to original case in `terms`
        matched_terms = set(term.lower() for term in matches)
        for term in terms:
            if term.lower() in matched_terms:
                keyword_to_documents[term].append(idx)  # Use document index (1-based)
    
    return keyword_to_documents

if __name__=='__main__':
    # List of keywords
    # chemical_terms = ["acet","2-3 acet", "acetyl", "acetyl group", "chemical A", "compound B", "reaction D"]

    # # List of documents
    # documents = [
    #     "Acet is 2-3 acet often used in reactions.",
    #     "2-3 acet often used in reactions.",
    #     "The acetyl group is a key part of biochemistry.",
    #     "Chemical A reacts with compound B in reaction D.",
    #     "Both acet and acetyl can be found in organic compounds."
    # ]


    # # Generate the dictionary
    # result = keyword_document_mapping(documents, chemical_terms)

    # # Display the result
    # for keyword, doc_list in result.items():
    #     print(f"Keyword '{keyword}' found in documents: {doc_list}")

    extractor = EntityExtractor()
    text = "Aspirin (C9H8O4) is widely used as an anti-inflammatory drug. Acetic anhydride reacts with salicylic acid to form it."
    extracted_entities = extractor.extract_entities(text)
    print("Extracted Entities:", extracted_entities)
