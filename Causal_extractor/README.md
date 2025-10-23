# About
this is tool for ewvaluating extracted causal structure
# folder structure
- source
- lib/output
- gemini.py
- .env
- visualize
# tech stack
- streamlit
# functional
- add prompt and source to UI -> generate output (call from gemini.py) and save to output folder
- read json file from output folder (can select file with drop down)
- pair output and source -> causal extracted
- have button for label (5 level completeness) horizontal radiobox
    - 5: flawless
    - 4: small missing points
    - 3: noticable missing detail
    - 2: critical missing part
    - 1: completely wrong
- localhost only
## UI
1. show raw source
2. list of pair of each raw source with output, buttons
3. OnClick auto save labeled output -> csv

## run command
streamlit run causal_extractor/visualize/visualize.py