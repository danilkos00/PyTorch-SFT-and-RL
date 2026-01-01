import os
import tarfile
import yaml
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        StoppingCriteria,
        StoppingCriteriaList
    )
import torch
import gdown


class MultiTokenStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, stop_sequences):
        self.tokenizer = tokenizer
        self.stop_sequences = stop_sequences

    def __call__(self, input_ids, scores, **kwargs):
        sequence = input_ids[0].tolist()

        for stop_seq in self.stop_sequences:
            stop_ids = self.tokenizer.encode(stop_seq, add_special_tokens=False)
            if len(sequence) >= len(stop_ids):
                if sequence[-len(stop_ids):] == stop_ids:
                    return True
        return False
    

class InputData(BaseModel):
    prompt: str


app = FastAPI()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

params_url = 'https://drive.google.com/uc?id=1UBL86kdVBHiCkXvOl-OxoDcpim7ft7fR'
output_path = './qwen.tar'
gdown.download(params_url, output_path, quiet=True)

with tarfile.open(output_path, 'r:') as file:
    file.extractall('./models')

os.remove(output_path)

model = AutoModelForCausalLM.from_pretrained('./models/Qwen2.5-Math-1.5B').to(device)
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Math-1.5B')

with open('generation_config.yml', 'r') as file:
    generation_params = yaml.safe_load(file)['generate']

stop_sequences = generation_params['stop_sequences']
del generation_params['stop_sequences']
stopping_criteria = StoppingCriteriaList([
    MultiTokenStoppingCriteria(tokenizer, stop_sequences)
])

@app.get("/")
async def read_root():
    return {"health_check": "OK", "model_version": 1}


@app.post("/generate")
async def predict(input_data: InputData):
    prompt = input_data.prompt
    processed_prompt = 'Question: ' + prompt + 'Answer: <think>'
    inputs = tokenizer([processed_prompt], return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}
    prompt_len = inputs['input_ids'].size(1)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **generation_params,
            stopping_criteria=stopping_criteria,
        )
    
    return {'answer' : '<think>' + tokenizer.decode(outputs[0][prompt_len:])}
