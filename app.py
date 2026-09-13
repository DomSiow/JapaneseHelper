import streamlit as st
from tokenizer import extract_vocabulary
from analyzer import rank_vocabulary
from enricher import enrich_vocabulary

# 1. Set up the web page title and description
st.title("🎌 Japanese Vocabulary Analyzer")
st.write("Paste your text below to generate a custom vocabulary study sheet.")

# 2. Create the user interface components
# Using a classic J-pop lyric as the default text to test it out!
default_text = "沈むように溶けてゆくように、二人だけの空が広がる夜に"

text_input = st.text_area("Input Japanese Text:", value=default_text, height=150)
top_n = st.slider("How many top words to fetch?", min_value=1, max_value=15, value=5)

# 3. Connect the UI to your data science pipeline
if st.button("Generate Study Sheet"):
    if text_input.strip() == "":
        st.warning("Please enter some Japanese text first.")
    else:
        # st.spinner shows a loading animation while the API works
        with st.spinner("Analyzing grammar and fetching definitions..."):
            
            # Step 1: Tokenize
            df_raw = extract_vocabulary(text_input)
            
            # Step 2: Rank
            df_ranked = rank_vocabulary(df_raw)
            
            # Step 3: Enrich
            df_final = enrich_vocabulary(df_ranked, top_n=top_n)
            
            # Step 4: Display the scoreboard
            st.success("Study Sheet Generated!")
            st.dataframe(df_final, use_container_width=True)