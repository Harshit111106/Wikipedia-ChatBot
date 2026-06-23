import requests
from bs4 import BeautifulSoup

def scrape_wikipedia(url: str):
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("Failed to fetch Wikipedia page data")
        
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Extract the true Wikipedia article headline
    title_element = soup.find('h1', id='firstHeading')
    article_title = title_element.text if title_element else "Unknown Article"
    
    # Pull text content from all valid paragraphs
    paragraphs = soup.find_all('p')
    text_content = "\n".join([p.text for p in paragraphs if p.text.strip()])
    
    return {"title": article_title, "text": text_content}