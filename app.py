from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

app = FastAPI()

model_path = "./models/llama-2-7b-chat"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

qa_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

@app.post("/generate")
async def generate_text(request: Request):
    data = await request.json()
    question = data.get("question", "")
    response = qa_pipeline(question, max_length=512, temperature=0.7, top_p=0.9)
    return {"response": response[0]["generated_text"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
