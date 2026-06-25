import requests
from bs4 import BeautifulSoup
from typing import List, Dict

# ── helpers ──────────────────────────────────────────────────────────────────

def _full_image_url(src: str) -> str:
    """Convert a Wikipedia thumbnail src to a full-resolution URL."""
    if src.startswith("//"):
        src = "https:" + src
    if "/thumb/" in src:
        # e.g. .../thumb/a/a7/file.jpg/320px-file.jpg  →  .../a/a7/file.jpg
        src = src.replace("/thumb/", "/")
        src = src.rsplit("/", 1)[0]
    return src

def _extract_tables(soup: BeautifulSoup) -> List[str]:
    """
    Convert every wikitable on the page into a plain-text block that can be
    chunked and embedded just like paragraph text.

    Format:
        TABLE: <caption>
        <Header1>: <value> | <Header2>: <value> | ...
        ...
    """
    table_blocks: List[str] = []
    for table in soup.find_all("table", class_="wikitable"):
        caption_el = table.find("caption")
        caption = caption_el.get_text(" ", strip=True) if caption_el else "Data Table"

        # Extract header row (th elements in first tr)
        header_row = table.find("tr")
        headers: List[str] = []
        if header_row:
            headers = [th.get_text(" ", strip=True) for th in header_row.find_all(["th", "td"])]

        rows_text: List[str] = []
        data_rows = table.find_all("tr")[1:]
        for row in data_rows:
            cells = row.find_all(["td", "th"])
            if not cells:
                continue
            cell_values = [c.get_text(" ", strip=True) for c in cells]
            if headers and len(headers) >= len(cell_values):
                row_str = " | ".join(
                    f"{h}: {v}" for h, v in zip(headers, cell_values) if h and v
                )
            else:
                row_str = " | ".join(v for v in cell_values if v)
            if row_str.strip():
                rows_text.append(row_str)

        if rows_text:
            block = f"TABLE: {caption}\n" + "\n".join(rows_text)
            table_blocks.append(block)

    return table_blocks

def _extract_images(soup: BeautifulSoup) -> List[Dict[str, str]]:
    """
    Scrape Wikipedia image thumbnails and their captions.
    Returns a list of {url, caption} dicts.
    Deduplicates by URL.
    """
    seen_urls: set = set()
    images: List[Dict[str, str]] = []

    # Modern Wikipedia uses <figure> with <figcaption>
    for figure in soup.find_all("figure"):
        img = figure.find("img")
        cap_el = figure.find("figcaption")
        if img and cap_el:
            src = _full_image_url(img.get("src", ""))
            caption = cap_el.get_text(" ", strip=True)
            if src and caption and src not in seen_urls:
                seen_urls.add(src)
                images.append({"url": src, "caption": caption})

    # Older Wikipedia markup uses div.thumbinner / div.thumbcaption
    for div in soup.find_all("div", class_="thumbinner"):
        img = div.find("img")
        cap_el = div.find("div", class_="thumbcaption")
        if img and cap_el:
            src = _full_image_url(img.get("src", ""))
            caption = cap_el.get_text(" ", strip=True)
            if src and caption and src not in seen_urls:
                seen_urls.add(src)
                images.append({"url": src, "caption": caption})

    return images

# ── public API ────────────────────────────────────────────────────────────────

def scrape_wikipedia(url: str) -> Dict:
    """
    Scrape a Wikipedia article and return:
      - title: str
      - text:  str  (paragraphs + tables combined, ready for chunking)
      - images: list of {url, caption}
    """
    headers = {"User-Agent": "WikipediaRAGApp/1.0 (goyalharshit006@gmail.com)"}
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        raise Exception(f"Failed to fetch Wikipedia page data (Status Code: {response.status_code})")

    soup = BeautifulSoup(response.content, "html.parser")

    # ── Title ────────────────────────────────────────────────────────────────
    title_el = soup.find("h1", id="firstHeading")
    article_title = title_el.text.strip() if title_el else "Unknown Article"

    # ── Paragraphs ───────────────────────────────────────────────────────────
    paragraphs = soup.find_all("p")
    para_text = "\n".join(p.get_text() for p in paragraphs if p.get_text(strip=True))

    # ── Tables ───────────────────────────────────────────────────────────────
    table_blocks = _extract_tables(soup)
    tables_text = "\n\n".join(table_blocks)

    # Combine: paragraph text first, then table blocks (separated clearly)
    full_text = para_text
    if tables_text:
        full_text += "\n\n" + tables_text

    # ── Images ───────────────────────────────────────────────────────────────
    images = _extract_images(soup)
    print(f"[SCRAPER] Extracted {len(table_blocks)} table(s) and {len(images)} image(s) from '{article_title}'")

    return {"title": article_title, "text": full_text, "images": images}