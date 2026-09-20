import pandas as pd
import requests
import urllib.parse
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "text/html,application/json"
})

def fetch_jisho_definition(term: str) -> tuple:
    """Queries Jisho's API, falling back to direct HTML scraping if necessary."""
    clean_term = term.strip()
    api_url = f"https://jisho.org/api/v1/search/words?keyword={urllib.parse.quote(clean_term)}"
    jisho_link = f"https://jisho.org/search/{urllib.parse.quote(clean_term)}"
    
    try:
        response = session.get(api_url, timeout=3)
        if response.status_code == 200:
            data = response.json().get("data", [])
            if data:
                first = data[0]
                reading = clean_term
                japanese_array = first.get("japanese", [])
                if japanese_array:
                    reading = japanese_array[0].get("reading", japanese_array[0].get("word", clean_term))
                    
                defs = None
                pos = "Vocabulary"
                senses = first.get("senses", [])
                if senses:
                    for sense in senses:
                        english_defs = sense.get("english_definitions", [])
                        if english_defs:
                            defs = "; ".join(english_defs)
                            break
                    parts = senses[0].get("parts_of_speech", [])
                    if parts:
                        pos = ", ".join(parts)
                        
                if defs:
                    return clean_term, reading, pos, defs, jisho_link

        html_response = session.get(jisho_link, timeout=3)
        if html_response.status_code == 200:
            soup = BeautifulSoup(html_response.text, 'html.parser')
            
            furigana_el = soup.find('span', class_='furigana')
            reading = furigana_el.text.strip() if furigana_el else clean_term
            
            meaning_el = soup.find('span', class_='meaning-meaning')
            if meaning_el:
                defs = meaning_el.text.strip()
                pos_el = soup.find('span', class_='part-of-speech')
                pos = pos_el.text.strip() if pos_el else "Vocabulary"
                return clean_term, reading, pos, defs, jisho_link

    except Exception:
        pass
            
    return clean_term, clean_term, "Vocabulary", "See Jisho entry", jisho_link

def _lookup_single_word(task: dict) -> dict:
    term_lemma = task["lemma"]
    term_clean, reading, pos, definition, jisho_link = fetch_jisho_definition(term_lemma)
    return {
        "term": term_clean,
        "reading": reading,
        "part_of_speech": pos,
        "definition": definition,
        "jisho_link": jisho_link,
    }

def enrich_vocabulary(df_vocab: pd.DataFrame, line: str = "", top_n: int = 20) -> pd.DataFrame:
    """Enriches vocabulary concurrently using Jisho."""
    if df_vocab.empty:
        return pd.DataFrame()

    tasks = df_vocab.head(top_n)[["lemma", "surface"]].to_dict("records")
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(_lookup_single_word, tasks))
        
    return pd.DataFrame(results)