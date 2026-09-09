import pandas as pd
from collections import Counter
from sudachipy import dictionary, tokenizer

# 1. Setup the Tokenizer
tokenizer_obj = dictionary.Dictionary().create()
mode = tokenizer.Tokenizer.SplitMode.C 

def extract_vocabulary(text: str) -> pd.DataFrame:
    """
    Takes raw Japanese text, extracts meaningful vocabulary (lemmas),
    and returns a pandas DataFrame with raw frequency counts.
    """
    # 2. Chop the text into pieces
    tokens = tokenizer_obj.tokenize(text, mode)
    
    valid_lemmas = []
    
    # 3. Define the "VIP" words we care about
    target_pos = {"名詞", "動詞", "形容詞", "副詞"} 
    
    # 4. Filter and Lemmatize
    for token in tokens:
        pos = token.part_of_speech()[0]  
        lemma = token.dictionary_form()  
        
        if pos in target_pos:
            valid_lemmas.append((lemma, pos))
            
    # 5. Count the words
    counts = Counter(valid_lemmas)
    
    # 6. Build the Scoreboard (DataFrame)
    df = pd.DataFrame([
        {"lemma": lemma, "pos": pos, "text_count": count}
        for (lemma, pos), count in counts.items()
    ])
    
    # 7. Sort from most to least frequent
    if not df.empty:
        df = df.sort_values(by="text_count", ascending=False).reset_index(drop=True)
        
    return df

# 8. Test Block
if __name__ == "__main__":
    sample_lyrics = "走り出すバスの窓から、君の姿が見えた。君は走っていた。私は泣いた。"
    df_vocab = extract_vocabulary(sample_lyrics)
    print(df_vocab)