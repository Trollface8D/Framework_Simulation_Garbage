import streamlit as st
import pandas as pd
import json
import os
import re

# Define the base directories
BASE_DIR = r"Causal_extractor\lib\output"
REFERENCE_DIR = r"Causal_extractor\lib\reference"

# --- NEW: Define the score storage path (inside the base directory) ---
SCORE_FILE_NAME = "validation_scores.json"
SCORE_FILE_PATH = os.path.join(BASE_DIR, SCORE_FILE_NAME)

# Define column names based on your JSON structure (Main Data)
COLUMNS = ["pattern", "causal type", "causal", "note", "Named entity/Object in causal", "original reference"]
# Rename columns for cleaner display and charting
DISPLAY_COLUMNS = {
    "pattern": "Pattern",
    "causal type": "Causal Type",
    "causal": "Causal Statement",
    "note": "Note",
    "Named entity/Object in causal": "Named Entity/Object",
    "original reference": "Original Reference"
}
REF_COL_NAME = DISPLAY_COLUMNS['original reference']

# ------------------------------------------------------------------
# 0. Score Management Functions (NEW)
# ------------------------------------------------------------------

def load_scores():
    """Loads the validation scores from a JSON file."""
    if not os.path.exists(SCORE_FILE_PATH):
        return {}
    try:
        with open(SCORE_FILE_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading scores: {e}")
        return {}

def save_scores(scores):
    """Saves the current validation scores to a JSON file."""
    try:
        os.makedirs(os.path.dirname(SCORE_FILE_PATH), exist_ok=True)
        with open(SCORE_FILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(scores, f, indent=4)
        st.toast("✅ Scores saved successfully!", icon='💾')
    except Exception as e:
        st.error(f"Error saving scores: {e}")

# ------------------------------------------------------------------
# 1. JSON Data Loading (Main Data) - MODIFIED to include scores
# ------------------------------------------------------------------
@st.cache_data(hash_funcs={dict: lambda x: json.dumps(x, sort_keys=True)})
def load_json_data(file_path, selected_file_name):
    if not os.path.exists(file_path):
        st.error(f"File not found at: {file_path}")
        return pd.DataFrame(columns=[v for v in DISPLAY_COLUMNS.values()] + ["Score"])
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        if isinstance(raw_data, list) and all(isinstance(item, list) for item in raw_data):
            if raw_data and len(raw_data[0]) != len(COLUMNS):
                st.error(f"Column mismatch in file {file_path}.")
                return pd.DataFrame(columns=[v for v in DISPLAY_COLUMNS.values()] + ["Score"])
            
            df = pd.DataFrame(raw_data, columns=COLUMNS)
            df = df.rename(columns=DISPLAY_COLUMNS)

            df.insert(0, 'Unique_ID', df.index)
            df['Score'] = ""

            all_scores = load_scores()
            file_scores = all_scores.get(selected_file_name, {})
            
            def get_score(row):
                score_key = str(row['Unique_ID'])
                return str(file_scores.get(score_key, "")) 

            df['Score'] = df.apply(get_score, axis=1)
            
            return df
        else:
            st.error("JSON must be list-of-lists format.")
            return pd.DataFrame(columns=[v for v in DISPLAY_COLUMNS.values()] + ["Score"])
            
    except Exception as e:
        st.error(f"Error reading: {e}")
        return pd.DataFrame(columns=[v for v in DISPLAY_COLUMNS.values()] + ["Score"])

# ------------------------------------------------------------------
# 2. CSV Reference Input Loading
# ------------------------------------------------------------------
@st.cache_data
def load_csv_reference(file_path):
    if not os.path.exists(file_path):
        return pd.DataFrame(columns=['input'])
    try:
        df = pd.read_csv(file_path)
        if "input" in df.columns:
            return df[["input"]].drop_duplicates().astype(str).reset_index(drop=True)
        else:
            st.sidebar.error("Missing column 'input'")
            return pd.DataFrame(columns=['input'])
    except Exception as e:
        st.sidebar.error(f"CSV error: {e}")
        return pd.DataFrame(columns=['input'])

# ------------------------------------------------------------------
# Highlighting
# ------------------------------------------------------------------

def highlight_references(row, selected_reference_input):
    style = [''] * len(row)
    if selected_reference_input and selected_reference_input != 'None':
        original_reference_text = str(row[REF_COL_NAME]).strip()
        if original_reference_text == selected_reference_input.strip():
            style = ['background-color: #ffd700; color: #333333'] * len(row)
    return style

# ------------------------------------------------------------------

st.set_page_config(
    page_title="Causal Data Visualization Dashboard",
    layout="wide"
)

st.title("📄 Causal Extractor Data Analyzer (JSON + CSV Reference)")
st.markdown(f"Main data loaded from: `{BASE_DIR}`")

# ----------------------------------------------------
# Sidebar CSV Selection
# ----------------------------------------------------

st.sidebar.header("🔍 CSV Input Comparison") 
csv_files = []
if os.path.isdir(REFERENCE_DIR):
    try:
        csv_files = [f for f in os.listdir(REFERENCE_DIR) if f.endswith('.csv')]
    except OSError as e:
        st.sidebar.error(f"Permission: {e}")

selected_csv_file_name = None
df_reference_input = pd.DataFrame(columns=['input'])

if csv_files:
    default = csv_files.index("generation_log.csv") if "generation_log.csv" in csv_files else 0
    selected_csv_file_name = st.sidebar.selectbox(
        "Select CSV",
        options=csv_files,
        index=default,
    )
    
    if selected_csv_file_name:
        csv_ref_path = os.path.join(REFERENCE_DIR, selected_csv_file_name)
        df_reference_input = load_csv_reference(csv_ref_path)
    reference_inputs = df_reference_input['input'].tolist()
else:
    st.sidebar.warning("No CSV found.")
    reference_inputs = []

selected_reference_input = 'None'

if reference_inputs:
    display_options = ['None'] + reference_inputs
    selected_reference_input = st.sidebar.selectbox(
        "Match Original Reference:",
        display_options,
        index=0
    )

# ----------------------------------------------------
# JSON Selection
# ----------------------------------------------------

json_files = []
if os.path.isdir(BASE_DIR):
    json_files = [f for f in os.listdir(BASE_DIR) if f.endswith('.json') and f != SCORE_FILE_NAME]

selected_file_name = None
df = pd.DataFrame(columns=[v for v in DISPLAY_COLUMNS.values()] + ["Score"]) 

if json_files:
    selected_file_name = st.selectbox("Select JSON", options=json_files)
    
    if selected_file_name:
        full_path = os.path.join(BASE_DIR, selected_file_name)
        df = load_json_data(full_path, selected_file_name)
else:
    st.info("No JSON files found.")

# ----------------------------------------------------
# Main Data
# ----------------------------------------------------

if not df.empty:
    
    pattern_col = DISPLAY_COLUMNS['pattern']
    causal_type_col = DISPLAY_COLUMNS['causal type']

    col1, col2 = st.columns(2)

    with col1:
        selected_patterns = st.multiselect(
            f"Filter by {pattern_col}:",
            df[pattern_col].unique(),
            default=df[pattern_col].unique()
        )

    with col2:
        selected_causal_types = st.multiselect(
            f"Filter by {causal_type_col}:",
            df[causal_type_col].unique(),
            default=df[causal_type_col].unique()
        )

    df_filtered = df[
        df[pattern_col].isin(selected_patterns) & 
        df[causal_type_col].isin(selected_causal_types)
    ].reset_index(drop=True)
    
    st.subheader(f"Filtered Data ({len(df_filtered)} rows)")

    if selected_reference_input != 'None':
        styled_df = df_filtered.style.apply(
            lambda row: highlight_references(row, selected_reference_input),
            axis=1
        )
        st.dataframe(styled_df, use_container_width=True)
    else:
        st.dataframe(df_filtered, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ----------------------------------------------------
    # Detail View
    # ----------------------------------------------------
    st.header("Causal Statement Detail View")

    cols_for_selection = [
        'Unique_ID',
        DISPLAY_COLUMNS['pattern'],
        DISPLAY_COLUMNS['causal type'],
        DISPLAY_COLUMNS['causal'],
        'Score',
        DISPLAY_COLUMNS['original reference']
    ]
    
    df_selection_view = df_filtered[cols_for_selection].copy()
    df_selection_view.insert(0, 'Select', False)
    
    edited_df_view = st.data_editor(
        df_selection_view.drop(columns=['Unique_ID']), 
        use_container_width=True, 
        column_config={
            "Select": st.column_config.CheckboxColumn("Select"),
            "Score": st.column_config.SelectboxColumn(
                "Score",
                options=["", "1", "2", "3", "4", "5"]
            ),
        },
        key="detail_view_editor"
    )

    edited_df_with_id = pd.merge(
        edited_df_view.reset_index(names=['original_index']),
        df_filtered[['Unique_ID']].reset_index(names=['original_index']),
        on='original_index',
    )
    
    selected_rows = edited_df_with_id[edited_df_with_id['Select'] == True]
    
    if len(selected_rows) >= 1:
        # ---------------------------
        # INLINE SCORE EDIT SECTION
        # ---------------------------
        if len(selected_rows) == 1:
            selected_row_data = selected_rows.iloc[0]
            selected_unique_id = str(selected_row_data['Unique_ID'])

            st.write("### ✍️ Edit Score Inline")
            inline_value = st.radio(
                "Score for this statement:",
                options=["1", "2", "3", "4", "5"],
                index=["1", "2", "3", "4", "5"].index(selected_row_data['Score']) if selected_row_data['Score'] in ["1", "2", "3", "4", "5"] else 0,
                horizontal=True,
                key=f"inline_{selected_unique_id}"
            )

            if st.button("💾 Save Inline Score", type="primary"):
                all_scores = load_scores()
                file_scores = all_scores.get(selected_file_name, {})
                file_scores[selected_unique_id] = inline_value.strip()
                all_scores[selected_file_name] = file_scores
                save_scores(all_scores)
                st.cache_data.clear()
                st.rerun()

            st.markdown("---")

            # ---------------------------
            # Comparison Display
            # ---------------------------
            causal_statement = selected_row_data[DISPLAY_COLUMNS['causal']]
            original_reference_text = str(selected_row_data[DISPLAY_COLUMNS['original reference']]).strip()
            
            st.subheader("Selected Item Details and Comparison")
            st.markdown(f"**Extracted:** *{causal_statement}*")
            st.markdown(f"**Current Saved Score:** `{selected_row_data['Score'] or 'None'}`")
            st.markdown("---")
            
            st.markdown("### Source Text Comparison")

            if df_reference_input is not None and len(df_reference_input) > 0:
                raw_text = original_reference_text
                
                highlight_style = 'background-color: #981ca3; font-weight: bold; color: white; padding: 2px; border-radius: 2px;' 
                best_csv_match = None
                
                for csv_input in df_reference_input['input'].tolist():
                    if raw_text.lower() in csv_input.lower():
                        best_csv_match = csv_input
                        break
                
                if best_csv_match:
                    styled_segment = f"<span style='{highlight_style}'>{raw_text}</span>"
                    final = best_csv_match.replace(raw_text, styled_segment, 1)
                    
                    st.markdown("**CSV Input (Source Document)**")
                    st.markdown(final, unsafe_allow_html=True)
                else:
                    st.warning("Original snippet not found inside CSV input text.")
                    st.markdown(raw_text)
            else:
                st.warning("No CSV Loaded.")

    else:
        st.info("Select a row above to view details.")
