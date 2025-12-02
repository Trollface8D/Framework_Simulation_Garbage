import streamlit as st
import pandas as pd
import json
import os
import re

# Define the base directories (use data_extract output and generation_log as requested)
BASE_DIR = os.path.join("Causal_extractor", "data_extract", "output")
REFERENCE_DIR = os.path.join("Causal_extractor", "data_extract")

# --- NEW: Define the score storage path (inside the base directory) ---
SCORE_FILE_NAME = "validation_scores.json"
SCORE_FILE_PATH = os.path.join(BASE_DIR, SCORE_FILE_NAME)

# Define column names based on your JSON structure (Main Data)
# V4 schema columns (all fields from JSON)
COLUMNS_V4 = [
    "pattern_type", "sentence_type", "marked_type", "explicit_type",
    "relationship", "marker", "subject", "object", "source_text", "reasoning"
]
DISPLAY_COLUMNS_V4 = {
    "pattern_type": "Pattern Type",
    "sentence_type": "Sentence Type",
    "marked_type": "Marked Type",
    "explicit_type": "Explicit Type",
    "relationship": "Relationship",
    "marker": "Marker",
    "subject": "Subject",
    "object": "Object",
    "source_text": "Source Text",
    "reasoning": "Reasoning"
}

# Legacy V3 schema columns
COLUMNS_V3 = ["pattern", "causal type", "causal", "note", "Named entity/Object in causal", "original reference"]
DISPLAY_COLUMNS_V3 = {
    "pattern": "Pattern",
    "causal type": "Causal Type",
    "causal": "Causal Statement",
    "note": "Note",
    "Named entity/Object in causal": "Named Entity/Object",
    "original reference": "Original Reference"
}

# Default to V4 for display references
REF_COL_NAME = "Source Text"  # V4 uses source_text

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
    """Load JSON data from file, auto-detect V3 or V4 schema, return DataFrame with all columns."""
    if not os.path.exists(file_path):
        st.error(f"File not found at: {file_path}")
        return pd.DataFrame(columns=list(DISPLAY_COLUMNS_V4.values()) + ["Score"]), "v4"
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        # Helper to safely get a value or empty string
        def get_val(d, key):
            v = d.get(key)
            return v if v is not None else ""

        # Case A: legacy format - list of lists (V3)
        if isinstance(raw_data, list) and raw_data and isinstance(raw_data[0], list):
            if len(raw_data[0]) != len(COLUMNS_V3):
                st.error(f"Column mismatch in file {file_path}.")
                return pd.DataFrame(columns=list(DISPLAY_COLUMNS_V3.values()) + ["Score"]), "v3"

            df = pd.DataFrame(raw_data, columns=COLUMNS_V3)
            df = df.rename(columns=DISPLAY_COLUMNS_V3)
            schema_version = "v3"

        # Case B: modern format - list of dicts
        elif isinstance(raw_data, list) and raw_data and isinstance(raw_data[0], dict):
            first_item = raw_data[0]
            # Detect V4 schema by checking for v4-specific keys
            is_v4 = any(k in first_item for k in ['pattern_type', 'sentence_type', 'marked_type', 'explicit_type', 'relationship', 'source_text'])
            
            if is_v4:
                # V4 schema: extract all fields directly from JSON keys
                rows = []
                for item in raw_data:
                    row = [get_val(item, col) for col in COLUMNS_V4]
                    rows.append(row)
                df = pd.DataFrame(rows, columns=COLUMNS_V4)
                df = df.rename(columns=DISPLAY_COLUMNS_V4)
                schema_version = "v4"
            else:
                # V3-like dict format: map to legacy columns
                rows = []
                for item in raw_data:
                    pattern = item.get('pattern', item.get('pattern_type', ''))
                    causal_type = item.get('causal type', item.get('sentence_type', ''))
                    causal = item.get('causal', item.get('causal_statement', item.get('relationship', '')))
                    note = item.get('note', item.get('notes', item.get('reasoning', '')))
                    named_entity = item.get('Named entity/Object in causal', item.get('named_entity', item.get('object', '')))
                    original_reference = item.get('original reference', item.get('original_reference', item.get('source_text', '')))
                    rows.append([pattern, causal_type, causal, note, named_entity, original_reference])
                df = pd.DataFrame(rows, columns=COLUMNS_V3)
                df = df.rename(columns=DISPLAY_COLUMNS_V3)
                schema_version = "v3"

        # Empty list case
        elif isinstance(raw_data, list) and not raw_data:
            return pd.DataFrame(columns=list(DISPLAY_COLUMNS_V4.values()) + ["Score"]), "v4"

        else:
            st.error("Unsupported JSON format: expected list-of-lists or list-of-dicts.")
            return pd.DataFrame(columns=list(DISPLAY_COLUMNS_V4.values()) + ["Score"]), "v4"

        # Add Unique_ID and populate Score from saved scores (if any)
        df.insert(0, 'Unique_ID', df.index)
        df['Score'] = ""

        all_scores = load_scores()
        file_scores = all_scores.get(selected_file_name, {})

        def get_score(row):
            score_key = str(row['Unique_ID'])
            return str(file_scores.get(score_key, ""))

        df['Score'] = df.apply(get_score, axis=1)

        return df, schema_version

    except Exception as e:
        st.error(f"Error reading: {e}")
        return pd.DataFrame(columns=list(DISPLAY_COLUMNS_V4.values()) + ["Score"]), "v4"

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
df = pd.DataFrame(columns=list(DISPLAY_COLUMNS_V4.values()) + ["Score"])
schema_version = "v4"  # default

if json_files:
    selected_file_name = st.selectbox("Select JSON", options=json_files)
    
    if selected_file_name:
        full_path = os.path.join(BASE_DIR, selected_file_name)
        df, schema_version = load_json_data(full_path, selected_file_name)
        st.caption(f"Detected schema: **{schema_version.upper()}**")
else:
    st.info("No JSON files found.")

# ----------------------------------------------------
# Main Data
# ----------------------------------------------------

if not df.empty:
    # Determine column names based on detected schema
    if schema_version == "v4":
        pattern_col = DISPLAY_COLUMNS_V4['pattern_type']
        causal_type_col = DISPLAY_COLUMNS_V4['sentence_type']
    else:
        pattern_col = DISPLAY_COLUMNS_V3['pattern']
        causal_type_col = DISPLAY_COLUMNS_V3['causal type']

    col1, col2 = st.columns(2)

    with col1:
        if pattern_col in df.columns:
            selected_patterns = st.multiselect(
                f"Filter by {pattern_col}:",
                df[pattern_col].unique(),
                default=df[pattern_col].unique()
            )
        else:
            selected_patterns = []

    with col2:
        if causal_type_col in df.columns:
            selected_causal_types = st.multiselect(
                f"Filter by {causal_type_col}:",
                df[causal_type_col].unique(),
                default=df[causal_type_col].unique()
            )
        else:
            selected_causal_types = []

    # Apply filters only if columns exist
    df_filtered = df.copy()
    if pattern_col in df.columns and selected_patterns:
        df_filtered = df_filtered[df_filtered[pattern_col].isin(selected_patterns)]
    if causal_type_col in df.columns and selected_causal_types:
        df_filtered = df_filtered[df_filtered[causal_type_col].isin(selected_causal_types)]
    df_filtered = df_filtered.reset_index(drop=True)
    
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

    # Build detail view columns based on schema
    if schema_version == "v4":
        cols_for_selection = [
            'Unique_ID',
            DISPLAY_COLUMNS_V4['pattern_type'],
            DISPLAY_COLUMNS_V4['sentence_type'],
            DISPLAY_COLUMNS_V4['marked_type'],
            DISPLAY_COLUMNS_V4['explicit_type'],
            DISPLAY_COLUMNS_V4['relationship'],
            'Score',
            DISPLAY_COLUMNS_V4['source_text']
        ]
    else:
        cols_for_selection = [
            'Unique_ID',
            DISPLAY_COLUMNS_V3['pattern'],
            DISPLAY_COLUMNS_V3['causal type'],
            DISPLAY_COLUMNS_V3['causal'],
            'Score',
            DISPLAY_COLUMNS_V3['original reference']
        ]
    # Filter to only include columns that exist in df
    cols_for_selection = [c for c in cols_for_selection if c in df_filtered.columns]
    
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
            if schema_version == "v4":
                causal_statement = selected_row_data.get(DISPLAY_COLUMNS_V4['relationship'], '')
                original_reference_text = str(selected_row_data.get(DISPLAY_COLUMNS_V4['source_text'], '')).strip()
            else:
                causal_statement = selected_row_data.get(DISPLAY_COLUMNS_V3['causal'], '')
                original_reference_text = str(selected_row_data.get(DISPLAY_COLUMNS_V3['original reference'], '')).strip()
            
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
