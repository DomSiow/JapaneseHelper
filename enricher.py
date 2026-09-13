import pandas as pd
import requests
import time
from tokenizer import extract_vocabulary
from analyzer import rank_vocabulary

def get_jisho_data(word: str) -> tuple:
    """
    Sends a request to the Jisho.org API and returns 
    the pronunciation (reading) and English definition.
    """
    url = f"https://jisho.org/api/v1/search/words?keyword={word}"
    
    try:
        response = requests.get(url, timeout=5)
        data = response.json()
        
        if data['data']:
            # Grab the very first (most common) search result
            first_result = data['data'][0]
            japanese_info = first_result['japanese'][0]
            
            # 1. Extract the reading (furigana)
            reading = japanese_info.get('reading', '')
            
            # 2. Extract the first English definition
            senses = first_result.get('senses', [])
            if senses:
                english_defs = senses[0].get('english_definitions', [])
                definition = ", ".join(english_defs)
            else:
                definition = "No definition found"
                
            return reading, definition
            
    except Exception as e:
        print(f"Error fetching {word}: {e}")
        
    return "", "Not found"

def enrich_vocabulary(df_vocab: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """
    Takes the ranked vocabulary, slices the top N words, 
    and adds dictionary definitions and readings via API.
    """
    # Slice the top N words so we don't spam the API with 1000 requests
    df_top = df_vocab.head(top_n).copy()
    
    readings = []
    definitions = []
    
    print(f"Fetching dictionary data for the top {top_n} words...")
    
    for lemma in df_top['lemma']:
        reading, definition = get_jisho_data(lemma)
        readings.append(reading)
        definitions.append(definition)
        
        # Be polite to the API server! Pause for half a second between requests.
        time.sleep(0.5) 
        
    # Add our new data as columns to the scoreboard
    df_top['reading'] = readings
    df_top['definition'] = definitions
    
    # Reorder columns to make it look like a study sheet
    columns_order = ['lemma', 'reading', 'definition', 'pos', 'specificity_score']
    return df_top[columns_order]

# --- Quick Test ---
if __name__ == "__main__":
    sample_lyrics = "走り出すバスの窓から、君の姿が見えた。君は走っていた。私は泣いた。"
    
    # Run the full pipeline!
    df_raw = extract_vocabulary(sample_lyrics)
    df_ranked = rank_vocabulary(df_raw)
    
    # Enrich the top 3 most important words
    df_final = enrich_vocabulary(df_ranked, top_n=3)
    
    print("\nFinal Enriched Study Sheet:")
    print("-" * 70)
    print(df_final)