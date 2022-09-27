from fastapi import FastAPI
from pydantic import BaseModel
import torch
from torch.quantization import quantize_dynamic
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from mangum import Mangum
model_ckpt = "sshleifer/distill-pegasus-cnn-16-4"


class Model:
    def __init__(self, checkpoint = model_ckpt, quantize = True):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(device)
        if quantize:
            model = quantize_dynamic(model, {torch.nn.Linear}, dtype = torch.qint8).to(device)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint).to(device)
        self.pipe = pipeline('summarization', model = model, tokenizer = tokenizer)

    def predict(self, text):
        return self.pipe(text)

model = Model()
app = FastAPI()
handler = Mangum(app)


class request_body(BaseModel):
    article: str

@app.get("/")
async def root():
    return {"status": "ok", "message": "Hello World"}

    
@app.post('/generate')
async def generate_summary(data : request_body):
    summary = model.predict(data.article)
    return {"summary": summary}
