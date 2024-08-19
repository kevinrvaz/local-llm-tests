from huggingface_hub import hf_hub_download

model_name = "TheBloke/Mistral-7B-OpenOrca-GGUF"
model_file = "mistral-7b-openorca.Q4_K_M.gguf"

model_path = hf_hub_download(model_name, filename=model_file)

# Take the model path from here and use in jupyter notebook
print(model_path)
