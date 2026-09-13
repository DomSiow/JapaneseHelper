import pandas as pd

def rank_vocabulary(df_vocab: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the frequency of each lemma in the text.
    Sorts ascending so Rarest = Head (top) and Most Common = Tail (bottom).
    """
    if df_vocab.empty:
        return pd.DataFrame()
        
    # Count how many times each base word appears
    freq_counts = df_vocab['lemma'].value_counts().reset_index()
    freq_counts.columns = ['lemma', 'frequency']
    
    # Merge the frequency back with the original data
    df_merged = pd.merge(df_vocab, freq_counts, on='lemma', how='left')
    
    # Sort by frequency (ascending). 
    # Words that appear only 1 time will be at the very top (rarest).
    df_ranked = df_merged.sort_values(by='frequency', ascending=True)
    
    return df_ranked