import requests
from bs4 import BeautifulSoup

def scrape_wikipedia(url: str):
    # ✨ ADDED: The "ID Card" header to bypass Wikipedia's bot blocker
    headers = {
        "User-Agent": "WikipediaRAGApp/1.0 (goyalharshit006@gmail.com)" 
    }
    
    # ✨ ADDED: Pass the headers directly into the GET request
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        raise Exception(f"Failed to fetch Wikipedia page data (Status Code: {response.status_code})")
        
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Extract the true Wikipedia article headline
    title_element = soup.find('h1', id='firstHeading')
    article_title = title_element.text if title_element else "Unknown Article"
    
    # Pull text content from all valid paragraphs
    paragraphs = soup.find_all('p')
    text_content = "\n".join([p.text for p in paragraphs if p.text.strip()])
    
    return {"title": article_title, "text": text_content}