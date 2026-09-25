import pandas as pd

def fetch_jlpt_data():
    levels = ['n5', 'n4', 'n3', 'n2', 'n1']
    dataframes = []
    
    # We will assign a numerical target value (N5 = 1, ..., N1 = 5)
    # This allows the model to treat difficulty as an ordinal scale.
    level_mapping = {'n5': 1, 'n4': 2, 'n3': 3, 'n2': 4, 'n1': 5}
    
    for level in levels:
        print(f"Fetching {level.upper()} data...")
        # Direct link to the raw CSV files hosted on GitHub
        url = f"https://raw.githubusercontent.com/jamsinclair/open-anki-jlpt-decks/main/src/{level}.csv"
        
        try:
            # The CSVs don't have standard headers, so we assign our own
            df = pd.read_csv(url, names=['vocab', 'reading', 'meaning', 'sentence', 'sentence_reading', 'sentence_meaning'])
            
            # We only need the Japanese example sentences for our text classifier
            df = df[['sentence']].dropna()
            
            # Filter out entries where the sentence column is empty or just whitespace
            df = df[df['sentence'].str.strip() != '']
            
            # Assign the target variable
            df['difficulty_target'] = level_mapping[level]
            df['jlpt_level'] = level.upper()
            
            dataframes.append(df)
            
        except Exception as e:
            print(f"Failed to fetch {level}: {e}")
            
    # Combine all levels into one master dataset
    master_df = pd.concat(dataframes, ignore_index=True)
    return master_df

if __name__ == "__main__":
    df_training = fetch_jlpt_data()
    
    # Save a local copy so we don't have to fetch it every time
    df_training.to_csv("jlpt_training_data.csv", index=False)
    
    print("\nDataset successfully built!")
    print(f"Total sentences extracted: {len(df_training)}")
    print("\nSample Data:")
    print(df_training.head())
    print("\nDistribution of levels:")
    print(df_training['jlpt_level'].value_counts())