from fastapi.responses import StreamingResponse, JSONResponse
from fastapi import FastAPI, HTTPException
from typing import Dict, List, Optional
from model import AI_Assistant
from pydantic import BaseModel
from functools import partial
import asyncio
import uvicorn



class LLM_Object:
    def generate(self, prompt: str):
        for i in range(10):
            yield f"Response chunk {i} for prompt: {prompt}\n"
    
    def reset(self):
        pass


class PromptRequest(BaseModel):
    prompt: str
    uid: str


class ResetRequest(BaseModel):
    uid: str


class ResetResponse(BaseModel):
    status: str

class GetResponse(BaseModel):
    uids: List


llm_instances: Dict[str, AI_Assistant] = {}


app = FastAPI(
    title="LLM API Service",
    description="API for managing LLM instances and streaming responses",
    version="1.0.0"
)


@app.post(
    "/generate",
    tags=["LLM Generation"],
    summary="Generate streaming response from LLM",
    response_description="Streaming text response"
)
async def generate_stream(request: PromptRequest):
    if request.uid not in llm_instances:
        llm_instances[request.uid] = AI_Assistant()
    
    llm = llm_instances[request.uid]
    loop = asyncio.get_event_loop()
    
    async def async_generator():
        generator = await loop.run_in_executor(
            None, 
            lambda: llm.generate(request.prompt)
        )
        for chunk in generator:
            yield chunk
    
    return StreamingResponse(
        async_generator(),
        media_type="text/plain"
    )


@app.post(
    "/reset",
    tags=["LLM Generation"],
    summary="Reset LLM instance",
    response_model=ResetResponse,
    response_description="Reset status"
)
async def reset_llm(request: ResetRequest):
    if request.uid not in llm_instances:
        raise HTTPException(status_code=404, detail="LLM instance not found")
    
    llm = llm_instances[request.uid]
    loop = asyncio.get_event_loop()
    
    await loop.run_in_executor(
        None,
        lambda: llm.reset()
    )
    
    return JSONResponse(content={"status": "success"})

@app.get(
    "/uid_list",
    tags=["LLM Generation"],
    summary="Get LLM instance list",
    response_model=GetResponse,
    response_description="LLM instance List"
)
async def get_llm_list():
    return JSONResponse(content={"uids": list(llm_instances.keys())})
    

@app.get(
    "/health",
    tags=["Health"],
    summary="Health check endpoint",
    response_description="Service health status"
)
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False
    )