import os
model_path = os.path.join(os.environ["HOME"], ".cache/huggingface/hub/models--TheBloke--Mistral-7B-OpenOrca-GGUF/snapshots/fbd9cd059e5fa0bba72a0abe8aea7e127d7994cb/mistral-7b-openorca.Q4_K_M.gguf")

from llama_cpp import Llama

model_kwargs = {
  "n_ctx":4096,    # Context length to use
  "n_threads":4,   # Number of CPU threads to use
  "n_gpu_layers":0,# Number of model layers to offload to GPU. Set to 0 if only using CPU
}

## Instantiate model from downloaded file
llm = Llama(model_path=model_path, **model_kwargs)

## Generation kwargs
generation_kwargs = {
    "max_tokens":200, # Max number of new tokens to generate
    "stop":["<|endoftext|>", "</s>"], # Text sequences to stop generation on
    "echo":False, # Echo the prompt in the output
    "top_k":1 # This is essentially greedy decoding, since the model will always return the highest-probability token. Set this value > 1 for sampling decoding
}

## Run inference
prompt = "what do you know about kevin rohan vaz? "
res = llm(prompt, **generation_kwargs) # Res is a dictionary

## Unpack and the generated text from the LLM response dictionary and print it
print(res["choices"][0]["text"])