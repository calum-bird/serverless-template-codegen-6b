import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Init is ran on server startup
# Load your model to GPU as a global variable here.
def init():
    global model
    global tokenizer

    print("loading to CPU...")
    model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-6B-mono", low_cpu_mem_usage=True)
    print("done")

    # conditionally load to GPU
    if device == "cuda:0":
        print("loading to GPU...")
        model.cuda()
        print("done")

    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-6B-mono")


# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model
    global tokenizer

    # Parse out your arguments
    prompt = model_inputs.get('prompt', None)
    if prompt == None:
        return {'message': "No prompt provided"}
    
    # Tokenize inputs
    input_tokens = tokenizer(prompt, return_tensors="pt").to(device)

    # Run the model
    generated_ids = model.generate(input_tokens)

    # Decode output tokens
    output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    result = {"output": output_text}

    # Return the results as a dictionary
    return result
