import streamlit as st
import pandas as pd
import json
import os # Import the os module for path manipulation

# Define the full file path
FILE_PATH = r"lib\output\exp1_output_v1.json"
# Define column names for clarity
COLUMNS = ["Type", "Category", "Description", "Qualifier", "Source"]

# Function to load and prepare the data, with error handling
@st.cache_data
def load_data(path):
    """Loads data from a local JSON file and converts it to a Pandas DataFrame."""
    if not os.path.exists(path):
        st.error(f"File not found at: {path}")
        return pd.DataFrame(columns=COLUMNS)
    
    try:
        with open(path, 'r') as f:
            raw_data = json.load(f)
        
        # Ensure the loaded data is in the expected list-of-lists format
        if isinstance(raw_data, list) and all(isinstance(item, list) for item in raw_data):
            df = pd.DataFrame(raw_data, columns=COLUMNS)
            return df
        else:
            st.error("JSON content is not in the expected list-of-lists format.")
            return pd.DataFrame(columns=COLUMNS)
            
    except json.JSONDecodeError:
        st.error("Error decoding JSON from file. Check file content for errors.")
        return pd.DataFrame(columns=COLUMNS)
    except Exception as e:
        st.error(f"An unexpected error occurred while reading the file: {e}")
        return pd.DataFrame(columns=COLUMNS)

# --- Streamlit App Layout ---

st.set_page_config(
    page_title="Garbage Management Data Dashboard",
    layout="wide"
)

st.title("🗑️ Garbage Management Data Analysis")
st.markdown(f"Data loaded from: `{FILE_PATH}`")

# Convert the data into a DataFrame
df = load_data(FILE_PATH)

# Handle case where DataFrame is empty
if df.empty:
    st.warning("The data frame is empty. Please check the file path and content.")
else:
    
    # ----------------------------------------------------
    ## Data Filtering
    # ----------------------------------------------------
    st.header("Interactive Data Filter")

    col1, col2 = st.columns(2)

    with col1:
        selected_types = st.multiselect(
            "Filter by Type (F=Fact, C=Causal, A=Action):",
            options=df['Type'].unique(),
            default=df['Type'].unique()
        )

    with col2:
        selected_categories = st.multiselect(
            "Filter by Category:",
            options=df['Category'].unique(),
            default=df['Category'].unique()
        )

    # Apply filters
    df_filtered = df[
        df['Type'].isin(selected_types) & 
        df['Category'].isin(selected_categories)
    ].reset_index(drop=True)

    st.subheader(f"Filtered Data ({len(df_filtered)} rows)")
    # Display the filtered data as an interactive table
    st.dataframe(df_filtered, use_container_width=True)

    # ----------------------------------------------------
    ## Data Distribution Charts
    # ----------------------------------------------------
    st.header("Data Distribution")

    col3, col4 = st.columns(2)

    # Visualization 1: Count by Type
    with col3:
        st.subheader("Count by Type")
        type_counts = df['Type'].value_counts().reset_index()
        type_counts.columns = ['Type', 'Count']
        st.bar_chart(type_counts.set_index('Type'))

    # Visualization 2: Count by Category
    with col4:
        st.subheader("Count by Category")
        category_counts = df['Category'].value_counts().reset_index()
        category_counts.columns = ['Category', 'Count']
        st.bar_chart(category_counts.set_index('Category'))