import pandas as pd
from tokenizer import extract_vocabulary

# A mock frequency dictionary of everyday Japanese 
# (For a final portfolio, you would load a real CSV dataset like the BCCWJ corpus here)
MOCK_BASELINE = {
    "私": 50000,       # Very common
    "走る": 10000,     # Common
    "君": 5000,        # Common
    "泣く": 2000,      # Less common
    "バス": 1000,      # Specific
    "走り出す": 100    # Very specific (High impact)
}
BASELINE_TOTAL_WORDS = 1_000_000

def rank_vocabulary(df_vocab: pd.DataFrame) -> pd.DataFrame:
    """
    Compares text word frequencies against a baseline Japanese corpus
    to rank them by specificity (how unique they are to this text).
    """
    if df_vocab.empty:
        return df_vocab
        
    total_text_words = df_vocab["text_count"].sum()
    
    def get_baseline_freq(lemma):
        # If the word isn't in our baseline, assume it's rare (count of 1)
        return MOCK_BASELINE.get(lemma, 1) / BASELINE_TOTAL_WORDS
        
    # 1. Text Frequency: What percentage of our text is this word?
    df_vocab["text_freq"] = df_vocab["text_count"] / total_text_words
    
    # 2. Baseline Frequency: What percentage of normal Japanese is this word?
    df_vocab["baseline_freq"] = df_vocab["lemma"].apply(get_baseline_freq)
    
    # 3. Specificity Score: The ratio of text frequency to baseline frequency
    df_vocab["specificity_score"] = df_vocab["text_freq"] / df_vocab["baseline_freq"]
    
    # 4. Sort by our new advanced metric instead of raw counts
    df_vocab = df_vocab.sort_values(by="specificity_score", ascending=False).reset_index(drop=True)
    
    # Clean up the output for readability
    df_vocab["specificity_score"] = df_vocab["specificity_score"].round(2)
    
    return df_vocab

# --- Quick Test ---
if __name__ == "__main__":
    sample_lyrics = "走り出すバスの窓から、君の姿が見えた。君は走っていた。私は泣いた。"
    
    print("Extracting and analyzing vocabulary...")
    df_raw = extract_vocabulary(sample_lyrics)
    df_ranked = rank_vocabulary(df_raw)
    
    print("\nRanked Vocabulary DataFrame:")
    print("-" * 50)
    print(df_ranked[['lemma', 'pos', 'text_count', 'specificity_score']])