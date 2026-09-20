import pandas as pd
from wordfreq import zipf_frequency

def rank_vocabulary(df_vocab: pd.DataFrame) -> pd.DataFrame:
    """Ranks extracted vocabulary by taking the most common Zipf score across available forms."""
    if df_vocab.empty:
        return df_vocab
        
    def get_true_rarity(row):
        # 1. Get the Zipf score for the strict dictionary lemma (e.g., 為る, 奇麗)
        lemma_score = zipf_frequency(row['lemma'], 'ja')
        
        # 2. Get the Zipf score for how it actually appeared in the text (e.g., する, きれい)
        # We use .get() so it doesn't break if your tokenizer hasn't added a 'surface' column yet
        surface_score = zipf_frequency(row.get('surface', row['lemma']), 'ja')
        
        # 3. Take the HIGHEST score (judging the word by its most commonly recognized spelling)
        return max(lemma_score, surface_score)

    # Apply the function row-by-row (axis=1) instead of just on the lemma column
    df_vocab['zipf'] = df_vocab.apply(get_true_rarity, axis=1)
    
    # Sort by lowest zipf score first (Most Rare)
    df_ranked = df_vocab.sort_values(by='zipf', ascending=True).reset_index(drop=True)
    
    return df_ranked