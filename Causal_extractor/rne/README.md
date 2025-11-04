# About
this compoonent will use the embedding model to form cluster of texy representation
# model used
- Embedding gemma3
    - link: https://huggingface.co/google/embeddinggemma-300m
    - term of uses: https://ai.google.dev/gemma/terms
    - property
        - Maximum input context length of 2048 tokens
        - output dim: 768, 512, 256, or 128
## to download nay model
1. ```pip install huggingface-hub```
2. cd to desired directory 
3. ```huggingface-cli download --local-dir ./model_name reponame(e.g google/embeddinggemma-300m)```