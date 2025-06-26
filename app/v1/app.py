from fastapi import FastAPI, Request
from pydantic import BaseModel
from vllm import LLM, SamplingParams
from rag.retriever import retrieve


# Step 1: Load the model ONCE
model_path = "./Baichuan2-13B-Chat-v2.0.1"  # 模型名称
llm = LLM(model=model_path, 
          trust_remote_code=True,
          tensor_parallel_size=2,  # 设置张量并行度
          dtype="float16",
          gpu_memory_utilization=0.8,  # GPU内存利用率
          )
params = SamplingParams(
    max_tokens=512,
    temperature=0.7,
    top_p=0.9)

# Step 2: Create API app
app = FastAPI()

class ChatRequest(BaseModel):
    prompt: str

@app.get("/")
async def root():
    """
    Root endpoint to check if the server is running.
    """
    return {"message": "Hello, this is the Baichuan2 API!"}

@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Chat endpoint to process user prompts.
    """
    # Step 3: Process the request
    docs = retrieve(request.prompt.strip())
    context = "\n".join(docs) if docs else "No relevant documents found."
    prompt = f"""你是一名专业的作业指导顾问，请结合以下参考资料和用户问题，生成结构化的回答。
    【参考资料】
    {context}
    【用户问题】
    {request.prompt.strip()}
    """
    response = llm.generate([prompt], sampling_params=params)
    
    # Step 4: Return the response
    return {"response": response[0].outputs[0].text.strip() if response else "No response generated."}