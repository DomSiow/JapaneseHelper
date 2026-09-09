import pandas as pd
import scipy
from sudachipy import dictionary, tokenizer

print("--- Environment Check ---")
print(f"Pandas version: {pd.__version__}")
print(f"SciPy version:  {scipy.__version__}")

# Initialize Japanese morphological tokenizer
tok = dictionary.Dictionary().create()
mode = tokenizer.Tokenizer.SplitMode.C

sample_text = "日本語を勉強して、面白い曲の歌詞を分析します。"
print(f"\nOriginal text: {sample_text}\n")

print(f"{'Surface (Original)':<20} | {'Dictionary Form (Lemma)':<25} | {'Part of Speech'}")
print("-" * 65)

tokens = tok.tokenize(sample_text, mode)
for token in tokens:
    surface = token.surface()
    lemma = token.dictionary_form()
    pos = token.part_of_speech()[0]  # Primary POS category
    print(f"{surface:<20} | {lemma:<25} | {pos}")

print("\nEnvironment is working correctly!")