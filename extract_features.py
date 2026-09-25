import pandas as pd
from sudachipy import tokenizer, dictionary
from wordfreq import zipf_frequency
import numpy as np

# 1. Initialize Sudachi Tokenizer
tokenizer_obj = dictionary.Dictionary().create()
mode = tokenizer.Tokenizer.SplitMode.C

def extract_sentence_features(text: str) -> pd.Series:
    """
    Parses a Japanese sentence and engineers numeric features 
    based on word frequency, density, and grammatical structure.
    """
    if not isinstance(text, str) or not text.strip():
        # Return baseline zeroes for empty data
        return pd.Series([0, 0.0, 0.0, 7.0, 7.0])
        
    tokens = tokenizer_obj.tokenize(text, mode)
    
    total_valid_words = 0
    unique_words = set()
    grammar_count = 0
    zipf_scores = []
    
    for t in tokens:
        pos = t.part_of_speech()[0]
        
        # Filter out punctuation and spaces
        if pos not in ["補助記号", "空白"]:
            total_valid_words += 1
            surface = t.surface()
            lemma = t.normalized_form()
            unique_words.add(surface)
            
            # Track particles and auxiliary verbs (structural complexity)
            if pos in ["助詞", "助動詞"]:
                grammar_count += 1
                
            # Zipf Rarity: max() logic from your analyzer.py
            score = max(zipf_frequency(surface, 'ja'), zipf_frequency(lemma, 'ja'))
            if score > 0:
                zipf_scores.append(score)
    
    # 2. Compute Statistical Metrics
    # Type-Token Ratio (Lexical Richness)
    unique_ratio = len(unique_words) / total_valid_words if total_valid_words > 0 else 0.0
    
    # Grammatical Density
    particle_ratio = grammar_count / total_valid_words if total_valid_words > 0 else 0.0
    
    # Distribution of Rarity (lower Zipf = rarer)
    if zipf_scores:
        median_zipf = np.median(zipf_scores)
        min_zipf = min(zipf_scores)  # Captures the single rarest word as a ceiling
    else:
        median_zipf = 7.0  # Safe default for extremely common text
        min_zipf = 7.0

    return pd.Series([
        total_valid_words, 
        unique_ratio, 
        particle_ratio, 
        median_zipf, 
        min_zipf
    ], index=['total_words', 'unique_ratio', 'particle_ratio', 'median_zipf', 'min_zipf'])

# 3. Execution Block
if __name__ == "__main__":
    print("Loading training data...")
    df = pd.read_csv("jlpt_training_data.csv")
    
    print(f"Extracting features for {len(df)} sentences. This may take a moment...")
    
    # Apply the feature extraction function across the text column
    features_df = df['sentence'].apply(extract_sentence_features)
    
    # Concatenate the new features alongside the original text and targets
    df_final = pd.concat([df, features_df], axis=1)
    
    # Drop rows where Sudachi couldn't find any valid words
    df_final = df_final[df_final['total_words'] > 0]
    
    # Save the finalized training matrix
    df_final.to_csv("jlpt_features_matrix.csv", index=False)
    
    print("\nFeature extraction complete!")
    print("Final Feature Matrix:")
    print(df_final[['jlpt_level', 'total_words', 'unique_ratio', 'median_zipf', 'min_zipf']].head(10))