import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import json

from router import SmartRouter
from config import MODEL_MAP, GROQ_API_KEY, CEREBRAS_API_KEY, SAMBANOVA_API_KEY

app = FastAPI(title="Accelera Router")
router = SmartRouter()

# Request Body (OpenAI Compatible)
class ChatRequest(BaseModel):
    model: str
    messages: List[dict]
    stream: Optional[bool] = False

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    
    # 1. Decide which hardware provider to use
    provider = router.decide_route(request.model, request.messages, request.stream)
    
    # 2. Get the specific model name for that provider
    target_model = MODEL_MAP.get(request.model, {}).get(provider)
    if not target_model:
        return {"error": f"Model {request.model} not supported"}

    # 3. Prepare the upstream request
    headers = {
        "Content-Type": "application/json",
    }
    
    url = ""
    api_key = ""
    
    if provider == "groq":
        url = "https://api.groq.com/openai/v1/chat/completions"
        api_key = GROQ_API_KEY
    elif provider == "cerebras":
        url = "https://api.cerebras.ai/v1/chat/completions"
        api_key = CEREBRAS_API_KEY
    elif provider == "sambanova":
        url = "https://api.sambanova.ai/v1/chat/completions"
        api_key = SAMBANOVA_API_KEY

    headers["Authorization"] = f"Bearer {api_key}"
    
    payload = {
        "model": target_model,
        "messages": request.messages,
        "stream": request.stream
    }

    # 4. Forward the request (Streaming Proxy)
    # We use httpx to stream the response back to the user in real-time
    async def upstream_generator():
        async with httpx.AsyncClient() as client:
            async with client.stream("POST", url, headers=headers, json=payload, timeout=60.0) as response:
                async for chunk in response.aiter_bytes():
                    yield chunk

    if request.stream:
        return StreamingResponse(upstream_generator(), media_type="text/event-stream")
    else:
        # Non-streaming fallback
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, headers=headers, json=payload, timeout=60.0)
            return resp.json()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)