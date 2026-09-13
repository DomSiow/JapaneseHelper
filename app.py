import streamlit as st
from tokenizer import extract_vocabulary
from analyzer import rank_vocabulary       
from enricher import enrich_vocabulary

# 1. Setup the Web Page
st.set_page_config(page_title="Japanese Lyric Analyzer", page_icon="🎌", layout="wide")

st.title("🎌 Japanese Lyric Analyzer")
st.write("Read lyrics in order with inline definitions, or extract vocabulary by statistical rarity.")

default_text = "走り出すバスの窓から\n君の姿が見えた\n君は走っていた\n私は泣いた"
text_input = st.text_area("Paste Lyrics Here:", value=default_text, height=200)

# 2. UI Toggle
analysis_mode = st.radio(
    "Choose Analysis Mode:",
    ["Line-by-Line Analysis (Reading Order)", "Global Vocabulary Filter (By Rarity)"],
    horizontal=True
)

st.divider()

# ==========================================
# MODE 1: LINE-BY-LINE (READING ORDER)
# ==========================================
if analysis_mode == "Line-by-Line Analysis (Reading Order)":
    
    if st.button("Analyze Line-by-Line"):
        if not text_input.strip():
            st.error("Please enter some text first.")
        else:
            lines = text_input.split('\n')
            valid_lines = [line for line in lines if line.strip()]
            total_lines = len(valid_lines)
            
            progress_text = "Analysis in progress. Please wait."
            progress_bar = st.progress(0.0, text=progress_text)
            
            for idx, line in enumerate(valid_lines):
                st.markdown(f"### 🎵 {line}")
                
                df_raw = extract_vocabulary(line)
                
                if df_raw.empty:
                    st.write("*No core vocabulary found.*")
                    st.divider()
                    progress_bar.progress((idx + 1) / total_lines, text=progress_text)
                    continue
                    
                df_ordered = df_raw.drop_duplicates(subset=['lemma'], keep='first')
                df_final = enrich_vocabulary(df_ordered, line=line, top_n=20)
                
                with st.expander("View Vocabulary & Examples"):
                    st.dataframe(
                        df_final, 
                        width="stretch",
                        column_config={
                            "jisho_link": st.column_config.LinkColumn(
                                "Jisho Dictionary",
                                help="Click to open entry on Jisho.org",
                                display_text="🔗 View on Jisho"
                            )
                        }
                    )
                st.divider()
                
                progress_bar.progress((idx + 1) / total_lines, text=progress_text)
                
            progress_bar.empty()
            st.success("Line-by-line analysis completed successfully!")

# ==========================================
# MODE 2: GLOBAL VOCABULARY FILTER
# ==========================================
elif analysis_mode == "Global Vocabulary Filter (By Rarity)":
    
    col1, col2 = st.columns(2)
    with col1:
        sort_order = st.radio("Sort by:", ["Most Rare", "Most Common"])
    with col2:
        top_n_global = st.slider("How many words to extract?", min_value=1, max_value=50, value=10)
    
    if st.button("Find Words"):
        if not text_input.strip():
            st.error("Please enter some text first.")
        else:
            with st.spinner("Analyzing statistical frequency..."):
                df_raw = extract_vocabulary(text_input)
                
                if df_raw.empty:
                    st.error("No vocabulary found.")
                else:
                    df_ranked = rank_vocabulary(df_raw)
                    df_ranked = df_ranked.drop_duplicates(subset=['lemma'], keep='first')
                    
                    if sort_order == "Most Rare":
                        df_filtered = df_ranked.head(top_n_global)
                    else:
                        df_filtered = df_ranked.tail(top_n_global).iloc[::-1]
                        
                    df_final = enrich_vocabulary(df_filtered, line="", top_n=top_n_global)
                    
                    st.success(f"Extracted the {top_n_global} {sort_order.lower()} words from the text!")
                    st.dataframe(
                        df_final, 
                        width="stretch",
                        column_config={
                            "jisho_link": st.column_config.LinkColumn(
                                "Jisho Dictionary",
                                help="Click to open entry on Jisho.org",
                                display_text="🔗 View on Jisho"
                            )
                        }
                    )