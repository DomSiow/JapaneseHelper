import pandas as pd
from sudachipy import tokenizer
from sudachipy import dictionary

# Initialize the Sudachi tokenizer
tokenizer_obj = dictionary.Dictionary().create()
mode = tokenizer.Tokenizer.SplitMode.C

def extract_vocabulary(text: str) -> pd.DataFrame:
    """
    Tokenizes Japanese text left-to-right.
    Filters out basic particles and punctuation.
    """
    if not text.strip():
        return pd.DataFrame()
        
    tokens = tokenizer_obj.tokenize(text, mode)
    
    vocab_list = []
    for t in tokens:
        pos = t.part_of_speech()[0]
        # Filter out punctuation, basic particles, and auxiliary verbs 
        if pos not in ["助詞", "補助記号", "助動詞", "空白"]:
            vocab_list.append({
                "surface": t.surface(),
                "lemma": t.normalized_form(),
                "pos": pos
            })
            
    return pd.DataFrame(vocab_list)