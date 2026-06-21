import wikipediaapi

def scrape_wikipedia_page(url: str) -> str:
    """
    Takes a full Wikipedia URL, extracts the topic title, 
    and returns the clean, raw text content of the page.
    """
    # 1. Parse the page title from the URL
    if "wiki/" not in url:
        raise ValueError("Invalid Wikipedia URL. Please provide a standard Wikipedia link.")
    
    # Extracts everything after 'wiki/' and replaces underscores with spaces
    page_title = url.split("wiki/")[-1].replace("_", " ")

    # 2. Initialize Wikipedia API with a mandatory custom User-Agent
    wiki = wikipediaapi.Wikipedia(
        user_agent="WikiRAGChatbot/1.0 (your_email@example.com)",
        language="en"
    )

    # 3. Fetch the page object
    page = wiki.page(page_title)

    # 4. Verify if the page actually exists
    if not page.exists():
        raise FileNotFoundError(f"The Wikipedia page for '{page_title}' could not be found.")

    # 5. Return the clean text (automatically excludes sidebars, templates, and HTML clutter)
    return page.text