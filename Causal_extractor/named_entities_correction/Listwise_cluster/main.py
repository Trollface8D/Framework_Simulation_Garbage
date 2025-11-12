import json
import os
import pandas as pd
import csv
import sys # <-- Add this

# --- Add these lines to fix the import ---
# Get the absolute path of the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up two levels to the project root (D:\)
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
# Add the project root to the sys.path
sys.path.append(project_root)
# --- End of fix ---

from utils.prompt import out_as_json
from utils.gemini import GeminiClient
from config import API_KEY
with open("prompt.json") as f:
    prompts = json.load(f)
prompt_template = prompts["listwise_clustering"]
client = GeminiClient(key=API_KEY)
path = "./output/"


if __name__ == "__main__":
    filelist = os.listdir(path)
    print(filelist)
    test = int(input("file index: "))
    # with open(os.path.join(path, test)) as f:
    #     raw_data = json.load(f)
    selected_file =  filelist[test]
    raw_data = pd.read_json(os.path.join(path, selected_file))
    named_entities = raw_data.iloc[:,-2].to_list()
    request = prompt_template.format(named_entities)
    print(f"request: {request}")
    output, response = client.generate(request, out_as_json, model_name = "gemini-2.5-pro", google_search=False)
    print(f"\noutput: {output}")
    # save output as json file at (./output/exp2)
    with open(os.path.join("./output/exp2", selected_file), "w") as f:
        f.write(output) 
    # save log at nec_log.csv as header: output_path, input_path, prompt_template, inference time
    # check if the file exist if not we write header file
    with open("output/nec_log.csv", "a+") as f:
        f.writelines(f"./output/exp2/{filelist[test]},./output/{selected_file}")
    print("wrote log...")
    print(response)
    

def manual_input():
    test = input("named entity list: ")
    request = prompt_template.format(test)
    print(f"request: {request}")
    output, response = client.generate(prompt_template.format(test), out_as_json, model_name = "gemini-2.5-pro", google_search=False)
    print(f"\noutput: {output}")
    