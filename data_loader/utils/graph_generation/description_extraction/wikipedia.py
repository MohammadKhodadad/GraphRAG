import wikipediaapi
from concurrent.futures import ThreadPoolExecutor
import time
import tqdm
# Initialize Wikipedia API
wiki_wiki = wikipediaapi.Wikipedia("Anonymous")

def wiki_exists(title):
    page = wiki_wiki.page(title)
    if page.exists():
        return True
    else:
        return False
def wiki_fetch_combined_text(title):
    """
    Fetch all text from a Wikipedia page combined into a single string.
    
    Args:
        title (str): Title of the Wikipedia page.
    
    Returns:
        str: Combined text of the entire page.
    """
    page = wiki_wiki.page(title)
    if not page.exists():
        return ''

    # Combine text from all sections recursively
    # def extract_text(sections):
    #     combined_text = ""
    #     for section in sections:
    #         combined_text += ( f'{section.title}: '+section.text + "\n\n")
    #         combined_text += extract_text(section.sections)  # Recursively add subsections
    #     return combined_text

    return page.summary# + '\n\n'+  extract_text(page.sections)


if __name__=='__main__':
    print(wiki_fetch_combined_text('Aspirin'))