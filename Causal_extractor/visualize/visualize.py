import streamlit as st
import pandas as pd
import json
import os 
import re # Import re for regex operations used in highlighting

# Define the base directories
BASE_DIR = r"Causal_extractor\lib\output"
REFERENCE_DIR = r"Causal_extractor\lib\reference" # New directory for reference PDFs

# Define column names based on your new structure
COLUMNS = ["pattern", "causal type", "causal", "note", "Named entity/Object in causal", "original reference"]
# Rename columns for cleaner display and charting
DISPLAY_COLUMNS = {
    "pattern": "Pattern",
    "causal type": "Causal Type",
    "causal": "Causal Statement",
    "note": "Note",
    "Named entity/Object in causal": "Named Entity/Object",
    "original reference": "Original Reference" # This column contains the full source text snippet
}

# Function to load and prepare the data from the file path
@st.cache_data
def load_data(file_path):
    """Loads data from a local JSON file specified by path and converts it to a Pandas DataFrame."""
    if not os.path.exists(file_path):
        st.error(f"File not found at: {file_path}") 
        return pd.DataFrame(columns=[v for v in DISPLAY_COLUMNS.values()])
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        if isinstance(raw_data, list) and all(isinstance(item, list) for item in raw_data):
            if raw_data and len(raw_data[0]) != len(COLUMNS):
                 st.error(f"Data structure mismatch in {file_path}. Expected {len(COLUMNS)} columns, but found {len(raw_data[0])} in the first row.")
                 return pd.DataFrame(columns=[v for v in DISPLAY_COLUMNS.values()])
                 
            df = pd.DataFrame(raw_data, columns=COLUMNS)
            df = df.rename(columns=DISPLAY_COLUMNS)
            return df
        else:
            st.error("JSON content is not in the expected list-of-lists format (must be a list of lists).")
            return pd.DataFrame(columns=[v for v in DISPLAY_COLUMNS.values()])
            
    except json.JSONDecodeError:
        st.error(f"Error decoding JSON from file: {file_path}. Check file content for errors.")
        return pd.DataFrame(columns=[v for v in DISPLAY_COLUMNS.values()])
    except Exception as e:
        st.error(f"An unexpected error occurred while reading the file: {e}")
        return pd.DataFrame(columns=[v for v in DISPLAY_COLUMNS.values()])


def highlight_references(row, selected_reference_file):
    """
    Applies a CSS style to highlight rows where the 'Original Reference'
    column contains text related to the selected PDF file name, using normalized strings.
    """
    style = [''] * len(row)
    ref_col_name = DISPLAY_COLUMNS['original reference']

    if selected_reference_file != 'None':
        # 1. Normalize the selected PDF file name
        normalized_pdf_name = selected_reference_file.lower().replace('.pdf', '').strip()
        
        # 2. Normalize the reference cell value
        ref_cell_value = str(row[ref_col_name]).lower()
        
        # Simple check: if the PDF name (or part of it) is in the source snippet text.
        # This is a heuristic that works well for direct comparisons.
        if normalized_pdf_name in ref_cell_value or ref_cell_value in normalized_pdf_name:
            style = ['background-color: #ffd700; color: #333333'] * len(row) 
            
    return style

def find_best_reference_match(original_reference_text, reference_files):
    """
    Attempts to find the most probable PDF file name from the available files
    based on the text snippet in the 'Original Reference' column.
    """
    best_match = "N/A (No strong PDF match found)"
    max_score = 0
    text_to_search = original_reference_text.lower()
    
    # Iterate through all available PDF files
    for pdf_name in reference_files:
        normalized_pdf_name = pdf_name.lower().replace('.pdf', '').strip()
        
        # We use a simple scoring heuristic: count how many words from the PDF name
        # appear in the reference text.
        score = 0
        pdf_name_parts = [p.strip() for p in normalized_pdf_name.split('_') if p.strip()]
        
        for part in pdf_name_parts:
            # Use regex for whole-word counting
            score += len(re.findall(r'\b' + re.escape(part) + r'\b', text_to_search))
            
        if score > max_score:
            max_score = score
            best_match = pdf_name
            
    return best_match


# --- Streamlit App Layout ---

st.set_page_config(
    page_title="Causal Data Visualization Dashboard",
    layout="wide"
)

st.title("📄 Causal Extractor Data Analyzer")
st.markdown(f"Files are loaded from the local directory: `{BASE_DIR}`")

# ----------------------------------------------------
## File Selection Logic (Main Data)
# ----------------------------------------------------

json_files = []
if os.path.isdir(BASE_DIR):
    try:
        json_files = [f for f in os.listdir(BASE_DIR) if f.endswith('.json')]
    except OSError as e:
        st.error(f"Permission error or cannot list directory contents: {e}")
        
selected_file_name = None
df = pd.DataFrame(columns=[v for v in DISPLAY_COLUMNS.values()]) 

if json_files:
    selected_file_name = st.selectbox(
        "Select a JSON file to analyze:",
        options=json_files,
        index=0
    )
    
    if selected_file_name:
        full_path = os.path.join(BASE_DIR, selected_file_name)
        df = load_data(full_path)
        
        if not df.empty:
            st.success(f"Successfully loaded and processed **{len(df)}** records from `{selected_file_name}`.")

else:
    if os.path.isdir(BASE_DIR):
        st.info(f"No JSON files found in the directory: `{BASE_DIR}`.")
    else:
        st.error(f"The specified directory does not exist: `{BASE_DIR}`. Please ensure the path is correct.")
        

# ----------------------------------------------------
## Reference File Selection (Sidebar)
# ----------------------------------------------------

selected_reference = 'None' # Default selection
reference_files = [] # Initialize reference_files list here
if os.path.isdir(REFERENCE_DIR):
    st.sidebar.header("🔍 Reference Comparison")
    st.sidebar.markdown("Highlight rows that match an existing PDF file.")
    
    try:
        # Find all PDF files in the reference directory
        reference_files = [f for f in os.listdir(REFERENCE_DIR) if f.lower().endswith('.pdf')]
    except OSError as e:
        st.sidebar.error(f"Permission error for reference directory: {e}")

    if reference_files:
        selected_reference = st.sidebar.selectbox(
            "Select a reference PDF to highlight:",
            options=['None'] + reference_files,
            index=0
        )
    else:
        st.sidebar.info(f"No PDF files found in: `{REFERENCE_DIR}`")
else:
    st.sidebar.warning(f"Reference directory not found: `{REFERENCE_DIR}`")


# Handle case where DataFrame is empty after attempted load
if df.empty:
    pass
else:
    # ----------------------------------------------------
    ## Data Filtering
    # ----------------------------------------------------
    st.header("Interactive Data Filter")

    # The columns to use for filtering and charting are the DISPLAY_COLUMNS values (renamed columns)
    pattern_col = DISPLAY_COLUMNS['pattern']
    causal_type_col = DISPLAY_COLUMNS['causal type']

    col1, col2 = st.columns(2)

    with col1:
        selected_patterns = st.multiselect(
            f"Filter by {pattern_col} (e.g., F, C, A):",
            options=df[pattern_col].unique(),
            default=df[pattern_col].unique()
        )

    with col2:
        selected_causal_types = st.multiselect(
            f"Filter by {causal_type_col}:",
            options=df[causal_type_col].unique(),
            default=df[causal_type_col].unique()
        )

    # Apply filters
    df_filtered = df[
        df[pattern_col].isin(selected_patterns) & 
        df[causal_type_col].isin(selected_causal_types)
    ].reset_index(drop=True)

    st.subheader(f"Filtered Data ({len(df_filtered)} rows)")
    
    # Apply highlighting if a reference is selected
    if selected_reference and selected_reference != 'None':
        st.markdown(f"**Rows referencing `{selected_reference}` are highlighted in gold.**")
        
        # Apply the styling function to the filtered DataFrame
        styled_df = df_filtered.style.apply(highlight_references, 
                                             axis=1, 
                                             selected_reference_file=selected_reference)
        st.dataframe(styled_df, use_container_width=True)
    else:
        # Display the unstyled DataFrame if no reference is selected
        st.dataframe(df_filtered, use_container_width=True)

    # ----------------------------------------------------
    ## Data Distribution Charts
    # ----------------------------------------------------
    st.header("Data Distribution")

    col3, col4 = st.columns(2)

    # Visualization 1: Count by Pattern
    with col3:
        st.subheader(f"Count by {pattern_col}")
        pattern_counts = df[pattern_col].value_counts().reset_index()
        pattern_counts.columns = [pattern_col, 'Count']
        st.bar_chart(pattern_counts.set_index(pattern_col))
        st.caption("Common patterns: F=Fact, C=Causal Relation, A=Action Item")

    # Visualization 2: Count by Causal Type
    with col4:
        st.subheader(f"Count by {causal_type_col}")
        type_counts = df[causal_type_col].value_counts().reset_index()
        type_counts.columns = [causal_type_col, 'Count']
        st.bar_chart(type_counts.set_index(causal_type_col))

    # ----------------------------------------------------
    ## Debugging/Raw Data View
    # ----------------------------------------------------
    st.markdown("---")
    st.subheader("Raw Data Preview")
    st.dataframe(df.head())
    
    # ----------------------------------------------------
    ## Labeling and Text Comparison
    # ----------------------------------------------------
    st.header("Labeling and Text Comparison")
    st.markdown("""
        Select a row below using the checkbox to see the **Original Reference** text snippet and the most probable source PDF file.
    """)

    # Select only the columns relevant for context and referencing
    cols_for_selection = [
        DISPLAY_COLUMNS['pattern'],
        DISPLAY_COLUMNS['causal type'],
        DISPLAY_COLUMNS['causal'],
        # FIX APPLIED HERE: Changed from 'Original Reference' to 'original reference'
        DISPLAY_COLUMNS['original reference'] 
    ]
    df_selection_view = df[cols_for_selection].copy()
    
    # Add a temporary index column to track the original row index
    df_selection_view['__index'] = df_selection_view.index
    
    # FIX: Initialize the 'Select' column *before* passing to st.data_editor
    df_selection_view['Select'] = False 
    
    # Use st.data_editor to allow row selection via checkbox
    edited_df = st.data_editor(
        df_selection_view,
        hide_index=True,
        use_container_width=True,
        column_config={
            # Add a selection column
            "Select": st.column_config.CheckboxColumn("Select", default=False),
            # Hide the temporary index
            "__index": st.column_config.Column(disabled=True, width="min")
        },
    )

    # Filter to find the indices of the selected rows
    selected_indices_in_editor = edited_df[edited_df['Select'] == True].index
    
    if len(selected_indices_in_editor) == 1:
        selected_row_data = edited_df.loc[selected_indices_in_editor[0]]
        
        causal_statement = selected_row_data[DISPLAY_COLUMNS['causal']]
        original_reference_text = selected_row_data[DISPLAY_COLUMNS['original reference']] 
        
        # --- NEW LOGIC: Match Reference Text to PDF File ---
        matched_pdf = find_best_reference_match(original_reference_text, reference_files)
        
        st.subheader("Selected Item Details")
        
        st.markdown("### Source Comparison View")
        
        # Display the extracted causal statement
        st.markdown(f"**Extracted Causal Statement (from JSON):** *{causal_statement}*")
        st.markdown("---")
        
        st.markdown(f"**Most Probable Source File:** **`{matched_pdf}`**")
        st.markdown("---")
        
        # Display the original reference text with the causal statement highlighted
        highlighted_reference_text = original_reference_text
        match = re.search(re.escape(causal_statement), original_reference_text, re.IGNORECASE)
        
        if match:
            match_text = match.group(0)
            highlight_style = 'background-color: yellow; font-weight: bold; padding: 2px; border-radius: 4px;'
            
            highlighted_reference_text = re.sub(
                re.escape(match_text), 
                f"<span style='{highlight_style}'>{match_text}</span>", 
                original_reference_text, 
                1, 
                flags=re.IGNORECASE
            )
            st.markdown(f"**Original Source Text Snippet (with Causal Statement highlighted):** {highlighted_reference_text}", unsafe_allow_html=True)
        else:
            st.markdown(f"**Original Source Text Snippet:** {original_reference_text}")
        
        st.warning("""
            **Verification Note:** The file above is the system's best guess based on keywords. 
            The highlighted text shows exactly where the **Causal Statement** was sourced from within the snippet.
        """)

    elif len(selected_indices_in_editor) > 1:
        st.warning("Please select only **one** row for detailed text comparison.")
    else:
        st.info("Select a row in the table above to see the comparison details.")
