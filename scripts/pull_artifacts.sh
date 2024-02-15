# Install huggingface-cli
pip install -U "huggingface_hub[cli]"

# Download the models and outputs from hugging face. This might take a while.
huggingface-cli download surajpaib/fmcib --local_dir ..

# If you want to download only the models,
# huggingface-cli download surajpaib/fmcib models ../models

# If you want to download only the outputs,
# huggingface-cli download surajpaib/fmcib outputs ../outputs
