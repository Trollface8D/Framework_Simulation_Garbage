import streamlit as st
from Framework_Simulation_Garbage.Causal_extractor.utils.gemini import GeminiClient
from config import API_KEY, out_as_json
import pandas as pd
import os

from datetime import datetime
import json

# --- Page Configuration ---
st.set_page_config(
    page_title="Gemini JSON Generator",
    page_icon="🤖",
    layout="wide"
)

# --- Constants ---
OUTPUT_DIR = "output"
LOG_FILE = "generation_log.csv"

# --- Load API Key and Configure Gemini ---

try:
    if not API_KEY:
        st.error("🚨 GOOGLE_API_KEY not found. Please set it in your .env file.")
        st.stop()
    model = GeminiClient(key=API_KEY)
except Exception as e:
    st.error(f"🚨 Error configuring Gemini: {e}")
    st.stop()


# --- Helper Functions ---
def initialize_log_file():
    """Creates the log file with headers if it doesn't exist."""
    if not os.path.exists(LOG_FILE):
        df = pd.DataFrame(columns=["prompt_template", "input", "output_filename", "timestamp"])
        df.to_csv(LOG_FILE, index=False)

def append_to_log(template, user_input, filename):
    """Appends a new record to the CSV log file."""
    new_log_entry = pd.DataFrame([{
        "prompt_template": template,
        "input": user_input,
        "output_filename": filename,
        "timestamp": datetime.now().isoformat()
    }])
    new_log_entry.to_csv(LOG_FILE, mode='a', header=False, index=False)

# --- Streamlit App UI ---
st.title("📄✨ Gemini JSON Generator")
st.markdown("Provide a prompt template and an input to generate a structured JSON response.")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
initialize_log_file()

# --- Input Fields ---
with st.form("prompt_form"):
    st.subheader("1. Define Your Prompt")
    
    # Use a two-column layout for inputs
    col1, col2 = st.columns(2)
    
    with col1:
        prompt_template = st.text_area(
            "**Prompt Template**",
            height=200,
            value="Generate a JSON object for a user profile. The user's name is {name}. The JSON should include fields for 'fullName', 'username', 'email', and 'isActive'. The username should be a lowercase version of the name without spaces.",
            help="Use curly braces `{}` for your input variable, e.g., `{name}`."
        )

    with col2:
        user_input = st.text_area(
            "**Input Value**",
            height=200,
            value="John Doe",
            help="This value will replace the placeholder in your template."
        )

    submitted = st.form_submit_button("🚀 Generate JSON", type="primary", use_container_width=True)

# --- Generation Logic ---
if submitted:
    if not prompt_template or not user_input:
        st.warning("⚠️ Please provide both a prompt template and an input value.")
    # A simple check to find at least one placeholder
    elif '{' not in prompt_template or '}' not in prompt_template:
        st.warning("⚠️ Your template must contain a placeholder like `{input}`.")
    else:
        with st.spinner("🧠 Gemini is thinking..."):
            try:
                # Dynamically find the placeholder key (e.g., 'name' from '{name}')
                # placeholder = prompt_template.split('{')[1].split('}')[0]
                final_prompt = prompt_template.format(user_input)

                # --- Call Gemini API ---
                text, response = model.generate(prompt=final_prompt, generation_config=out_as_json, model_name="gemini-2.5-pro", google_search=False)
                
                # Clean the response to extract only the JSON part
                cleaned_response_text = response.text.strip().replace("```json", "").replace("```", "")
                
                # --- Process and Save JSON ---
                parsed_json = json.loads(cleaned_response_text)
                
                # Generate a unique filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"response_{timestamp}.json"
                filepath = os.path.join(OUTPUT_DIR, filename)

                with open(filepath, 'w') as f:
                    json.dump(parsed_json, f, indent=4)
                
                # --- Log and Show Success ---
                append_to_log(prompt_template, user_input, filename)
                st.success(f"✅ Success! Response saved to: `{filepath}`")
                
                # Display the generated JSON in the app
                st.subheader("Generated JSON Output")
                st.json(parsed_json)

            except json.JSONDecodeError:
                st.error("🚨 Failed to decode JSON from Gemini's response. The model might not have returned valid JSON.")
                st.code(response.text, language="text")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

# --- Display Log File ---
st.divider()
st.subheader("📜 Generation History")
if os.path.exists(LOG_FILE):
    log_df = pd.read_csv(LOG_FILE)
    st.dataframe(log_df.tail(10), use_container_width=True)
else:
    st.info("No logs found yet. Generate a response to start logging.")