import wikipediaapi

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
        return None

    # Combine text from all sections recursively
    def extract_text(sections):
        combined_text = ""
        for section in sections:
            combined_text += ( f'{section.title}: '+section.text + "\n\n")
            combined_text += extract_text(section.sections)  # Recursively add subsections
        return combined_text

    return page.summary+ '\n\n'+  extract_text(page.sections)


def wiki_fetch_pages_in_category_recursive_combined(category_name, max_pages=100, max_depth=3, current_depth=0):
    """
    Recursively fetch pages in a Wikipedia category and combine all their text.
    
    Args:
        category_name (str): Name of the Wikipedia category.
        max_pages (int): Maximum number of pages to retrieve.
        max_depth (int): Maximum depth for recursive retrieval of subcategories.
        current_depth (int): Current recursion depth.
    
    Returns:
        dict: Dictionary with page titles as keys and combined text and IDs as values.
    """
    if current_depth > max_depth:
        return {}
    
    category = wiki_wiki.page(f"Category:{category_name}")
    if not category.exists():
        return {}

    documents = {}
    pages = category.categorymembers

    for title, page in pages.items():
        if page.ns == 0:  # Main namespace pages (articles)
            combined_text = wiki_fetch_combined_text(title)
            if combined_text:
                documents[title] = {
                    "text": combined_text,
                    "id": title.replace(" ", "_").lower()  # Generate unique ID by normalizing the title
                }
            if len(documents) >= max_pages:
                break
        elif page.ns == 14:  # Subcategories
            subcategory_documents = wiki_fetch_pages_in_category_recursive_combined(
                page.title.replace("Category:", ""),
                max_pages=max_pages - len(documents),
                max_depth=max_depth,
                current_depth=current_depth + 1
            )
            documents.update(subcategory_documents)
            if len(documents) >= max_pages:
                break
    
    return documents


# Example Usage
if __name__ == "__main__":
    # Specify the category to fetch
    category_name = "Medicine"
    max_pages = 20
    max_depth = 2

    # Fetch pages recursively
    documents = wiki_fetch_pages_in_category_recursive_combined(category_name, max_pages=max_pages, max_depth=max_depth)
    print(f"Retrieved {len(documents)} pages from the '{category_name}' category.")

    # Example: Display the ID and snippet of text for each page
    for title, content in documents.items():
        print(f"\nTitle: {title}")
        print(f"ID: {content['id']}")
        print(f"Text Snippet: {content['text'][:200]}...")
