import re
import requests
from bs4 import BeautifulSoup
from typing import List, Dict

# ── noise classification ──────────────────────────────────────────────────────

# Filename / URL terms that are dead giveaways for structural UI images
_NOISE_FILENAME_TERMS: frozenset = frozenset({
    "icon", "logo", "sprite", "button", "stub", "magnif",
    "search", "arrow", "edit", "oojs", "disambig",
    "commons", "wiktionary", "wikisource", "wikibooks",
    "wikinews", "wikiquote", "wikiversity", "wikivoyage",
    "mediawiki", "protection", "shackle", "padlock", "lock-",
    "pog.", "placeholder", "blank", "transparent",
    "separator", "bullet", "star_full", "star_empty",
    "question_book", "refimprove",
})

# Caption / alt-text phrases that indicate a non-content image
_NOISE_CAPTION_TERMS: frozenset = frozenset({
    "icon", "logo", "stub", "edit this at wikidata",
    "wikimedia commons", "semi-protected", "fully protected",
    "maintenance", "disambiguation", "move protected",
})


def _is_noise_image(url: str, caption: str, alt: str = "") -> bool:
    """
    Return True when the image is a UI / structural / layout element
    that should be excluded from the knowledge base.

    Criteria (any one is sufficient to reject):
      1. File extension is .svg  (almost always icons / logos / badges)
      2. The URL filename contains a known noise keyword
      3. The caption is missing or shorter than 8 characters
      4. The caption or alt text contains a known noise phrase
    """
    url_lower = url.lower()

    # 1. SVG files — icons, logos, badges, nearly always structural
    if url_lower.endswith(".svg"):
        return True

    # 2. Filename keyword check
    filename = url_lower.rsplit("/", 1)[-1]
    if any(term in filename for term in _NOISE_FILENAME_TERMS):
        return True

    # 3. Empty / trivially short caption
    clean_caption = caption.strip()
    if not clean_caption or len(clean_caption) < 8:
        return True

    # 4. Caption / alt-text noise phrase check
    caption_lower = clean_caption.lower()
    alt_lower = alt.lower() if alt else ""
    if any(term in caption_lower for term in _NOISE_CAPTION_TERMS):
        return True
    if alt and any(term in alt_lower for term in _NOISE_FILENAME_TERMS):
        return True

    return False


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


def _extract_main_text(soup: BeautifulSoup) -> str:
    """
    Extract ALL readable text from the Wikipedia article's main content body.

    Strategy:
      1. Locate #mw-content-text (or fall back to #bodyContent / <body>).
      2. Decompose known noise elements (navboxes, TOC, reference lists,
         [edit] buttons, citation superscripts, maintenance banners, scripts).
      3. Insert newlines around block elements so paragraphs/lists are separated,
         while inline elements (like <a> links) remain seamlessly part of their sentence.
    """
    content = (
        soup.find("div", id="mw-content-text")
        or soup.find("div", id="bodyContent")
        or soup.body
    )
    if not content:
        return ""

    # ── noise decomposition ───────────────────────────────────────────────────
    # Citation superscripts  [1] [2] …
    for el in content.find_all("sup", class_="reference"):
        el.decompose()
    # [edit] section links
    for el in content.find_all("span", class_="mw-editsection"):
        el.decompose()
    # Table of contents
    for el in content.find_all(id="toc"):
        el.decompose()
    for el in content.find_all("div", class_="toc"):
        el.decompose()
    # Navigation boxes (footer link clusters)
    for el in content.find_all("div", class_=lambda c: c and "navbox" in c):
        el.decompose()
    # References / footnotes section
    for el in content.find_all("div", class_=["reflist", "mw-references-wrap"]):
        el.decompose()
    # Category links bar
    for el in content.find_all(id="catlinks"):
        el.decompose()
    # Stub / maintenance article-message banners
    for el in content.find_all("table", class_=lambda c: c and "ambox" in c):
        el.decompose()
    # Sister-project boxes
    for el in content.find_all("div", class_=lambda c: c and "sister" in c):
        el.decompose()
    # Scripts and style blocks
    for el in content.find_all(["script", "style"]):
        el.decompose()

    # ── text extraction ───────────────────────────────────────────────────────
    # Replace <br> with newline
    for br in content.find_all("br"):
        br.replace_with("\n")
        
    # Insert newlines around block elements so they form distinct paragraphs
    block_elements = ["p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table", "tr", "ul", "ol", "dl", "dt", "dd"]
    for block in content.find_all(block_elements):
        block.insert_before("\n")
        block.insert_after("\n")

    # Get raw text without stripping (so inline spaces survive)
    raw = content.get_text(separator="", strip=False)
    
    # Clean up whitespace:
    # 1. Split into lines and strip edge whitespace
    # 2. Collapse internal runs of spaces within each line
    # 3. Rejoin with newlines and collapse multiple newlines
    lines = [line.strip() for line in raw.split("\n")]
    cleaned_lines = [re.sub(r"\s+", " ", line) for line in lines if line.strip()]
    
    text = "\n\n".join(cleaned_lines)
    return text.strip()


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
    Noise images (icons, logos, SVGs, structural UI elements, blank captions)
    are excluded via _is_noise_image() before anything reaches the vector store.
    Returns a list of clean {url, caption} dicts, deduplicated by URL.
    """
    seen_urls: set = set()
    images: List[Dict[str, str]] = []

    # Modern Wikipedia: <figure> with <figcaption>
    for figure in soup.find_all("figure"):
        img = figure.find("img")
        cap_el = figure.find("figcaption")
        if img and cap_el:
            src = _full_image_url(img.get("src", ""))
            alt = img.get("alt", "")
            caption = cap_el.get_text(" ", strip=True)
            if src and src not in seen_urls and not _is_noise_image(src, caption, alt):
                seen_urls.add(src)
                images.append({"url": src, "caption": caption})

    # Legacy Wikipedia markup: div.thumbinner / div.thumbcaption
    for div in soup.find_all("div", class_="thumbinner"):
        img = div.find("img")
        cap_el = div.find("div", class_="thumbcaption")
        if img and cap_el:
            src = _full_image_url(img.get("src", ""))
            alt = img.get("alt", "")
            caption = cap_el.get_text(" ", strip=True)
            if src and src not in seen_urls and not _is_noise_image(src, caption, alt):
                seen_urls.add(src)
                images.append({"url": src, "caption": caption})

    return images


# ── public API ────────────────────────────────────────────────────────────────

def scrape_wikipedia(url: str) -> Dict:
    """
    Scrape a Wikipedia article and return:
      - title:  str
      - text:   str  (full article body + structured wikitables, ready for chunking)
      - images: list of noise-filtered {url, caption} dicts
    """
    headers = {"User-Agent": "WikipediaRAGApp/1.0 (goyalharshit006@gmail.com)"}
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        raise Exception(f"Failed to fetch Wikipedia page data (Status Code: {response.status_code})")

    soup = BeautifulSoup(response.content, "html.parser")

    # ── Title ────────────────────────────────────────────────────────────────
    title_el = soup.find("h1", id="firstHeading")
    article_title = title_el.text.strip() if title_el else "Unknown Article"

    # ── Images  (before any soup mutation) ───────────────────────────────────
    images = _extract_images(soup)

    # ── Structured table text  (before soup mutation) ─────────────────────
    table_blocks = _extract_tables(soup)

    # ── Full-body text  (may decompose noise nodes in-place) ─────────────────
    main_text = _extract_main_text(soup)

    # Append structured table blocks after the body text so the LLM gets
    # both the prose context AND the precise Header: Value rows.
    full_text = main_text
    if table_blocks:
        full_text += "\n\n" + "\n\n".join(table_blocks)

    print(
        f"[SCRAPER] '{article_title}': "
        f"{len(full_text):,} chars | "
        f"{len(table_blocks)} table(s) | "
        f"{len(images)} content image(s) (noise filtered)"
    )

    return {"title": article_title, "text": full_text, "images": images}



    return {"title": article_title, "text": full_text, "images": images}