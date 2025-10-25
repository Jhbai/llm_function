from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials # 加入FastAPI的安全性驗證
from fastapi import FastAPI, HTTPException, Depends
from model import AI_Assistant, AI_Agent
from typing import Dict, List, Optional
from rag import add_memory, get_memory
from model import AI_Assistant
from pydantic import BaseModel
from functools import partial
from db import conn, cursor
import asyncio
import uvicorn
import secrets # 用於安全比較字串
import uuid

tools = {
    "add_memory": add_memory,
    "get_memory": get_memory
}



class LLM_Object:
    def generate(self, prompt: str):
        for i in range(10):
            yield f"Response chunk {i} for prompt: {prompt}\n"
    
    def reset(self):
        pass

class ChatRequest(BaseModel):
    status: str

class UserRequest(BaseModel):
    user: str

class ChatResponse(BaseModel):
    message: str

class HistoryRequest(BaseModel):
    start: str
    end: str

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

security = HTTPBasic()
SECRET_USERNAMES = ["bear", "dab", "mom"]
SECRET_PASSWORDS = ["1015", "1015", "1015"]

def authenticate_user(credentials: HTTPBasicCredentials = Depends(security)):
    for i in range(len(SECRET_USERNAMES)):
        SECRET_USERNAME, SECRET_PASSWORD = SECRET_USERNAMES[i], SECRET_PASSWORDS[i]
        correct_username = secrets.compare_digest(credentials.username, SECRET_USERNAME)
        correct_password = secrets.compare_digest(credentials.password, SECRET_PASSWORD)
        if (correct_username and correct_password): break
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return {"user": credentials.username}

auth_dependency = Depends(authenticate_user)


app = FastAPI(
    title="LLM API Service",
    description="API for managing LLM instances and streaming responses",
    version="1.0.0"
)

@app.post(
    "/chat",
    tags=["chat box"],
    summary="Each user can send message for group chat",
    response_description="Group Chat",
)
async def group_chat(request: ChatRequest, auth: UserRequest=auth_dependency):
    try:
        res = cursor.execute("INSERT INTO ChatTable (user, message) VALUES (?, ?)",
                             (auth, request.message)
                             )
        conn.commit()
    except Exception as e:
        print("Error:", e)
        conn.rollback()
    finally:
        conn.close()
    return JSONResponse(content={"status": "success"})

@app.post(
    "/chat_history",
    tags=["chat box"],
    summary="Search message from group chat",
    response_description="Group Chat History",
    dependencies=[auth_dependency]
)
async def history_chat(request: HistoryRequest):
    st, ed = request.start, request.end
    cursor.execute("""
    SELECT user, message, dateTime
    FROM ChatTable
    WHERE dateTime BETWEEN ? AND ?
    ORDER BY dateTime ASC
    """, (st, ed))
    return cursor.fetchall()

@app.post(
    "/generate",
    tags=["LLM Generation"],
    summary="Generate streaming response from LLM",
    response_description="Streaming text response",
    dependencies=[auth_dependency]
)
async def generate_stream(request: PromptRequest):
    if request.uid not in llm_instances:
        llm_instances[request.uid] = AI_Assistant()
    
    llm = llm_instances[request.uid]
    loop = asyncio.get_event_loop()
    
    async def async_generator():
        instruction = await loop.run_in_executor(
            None,
            lambda: AI_Agent(request.prompt)
        )
        instruction = eval(instruction)
        if instruction["name"] == "no_call":
            prompt = request.prompt
        else:
            print("[instruction]", instruction)
            memory = tools[instruction["name"]](**instruction["parameters"])
            prompt = f"<tool_response>{memory}</tool_response>" + request.prompt
        generator = await loop.run_in_executor(
            None, 
            lambda: llm.generate(prompt)
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
    response_description="Reset status",
    dependencies=[auth_dependency]
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
    response_description="LLM instance List",
    dependencies=[auth_dependency]
)
async def get_llm_list():
    return JSONResponse(content={"uids": list(llm_instances.keys())})
    

@app.get(
    "/health",
    tags=["Health"],
    summary="Health check endpoint",
    response_description="Service health status",
    dependencies=[auth_dependency]
)
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    print(f"--- Starting server. Access Swagger UI at http://0.0.0.0:8000/docs ---")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False
    )