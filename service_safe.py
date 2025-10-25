from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials # 加入FastAPI的安全性驗證
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from model import AI_Assistant, AI_Agent
from typing import Dict, List, Optional
from rag import add_memory, get_memory
from model import AI_Assistant
from pydantic import BaseModel
from functools import partial
from db import conn, cursor
import datetime
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
    message: str

class UserRequest(BaseModel):
    user: str

class ChatResponse(BaseModel):
    status: str

class HistoryRequest(BaseModel):
    start: str
    end: str

class DeleteRequest(BaseModel):
    id: int

class DeleteResponse(BaseModel):
    status: str

class PromptRequest(BaseModel):
    prompt: str
    uid: str

class MemorizeRequest(BaseModel):
    prompt: str

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

login_log = {}

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
    now = datetime.datetime.now().replace(microsecond=0)
    login_log[credentials.username] = str(now)
    return {"user": credentials.username}

auth_dependency = Depends(authenticate_user)


app = FastAPI(
    title="LLM API Service",
    description="API for managing LLM instances and streaming responses",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 或是更嚴格的來源，例如 "http://localhost:8080"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get(
    "/login/log",
    tags=["Login Log"],
    summary="Check what time is the user finally login",
    response_description="login time record",
    dependencies=[auth_dependency]
)
async def log_of_login():
    return JSONResponse(content=login_log)

@app.post(
    "/chat",
    tags=["chat box"],
    summary="Each user can send message for group chat",
    response_model=ChatResponse,
    response_description="Group Chat",
)
async def group_chat(request: ChatRequest, auth: UserRequest=auth_dependency):
    try:
        res = cursor.execute("INSERT INTO ChatTable (user, message) VALUES (?, ?)",
                             (auth["user"], request.message)
                             )
        conn.commit()
    except Exception as e:
        print("Error:", e)
        conn.rollback()
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
    "/delete_history",
    tags=["chat box"],
    summary="Delete message from group chat",
    response_description="Delete Chat History",
    response_model=DeleteResponse,
    dependencies=[auth_dependency]
)
async def history_deletion(request: DeleteRequest):
    cursor.execute("DELETE FROM ChatTable WHERE id = ?", (request.id,))
    conn.commit()
    return JSONResponse(content={"status": "success"})

@app.post(
    "/memorize",
    tags=["LLM Memorization"],
    summary="To Decide whether memorize the chat or not",
    response_description="Memorization",
    dependencies=[auth_dependency]
)
async def Memorize(request: MemorizeRequest):
    loop = asyncio.get_event_loop()
    instruction = await loop.run_in_executor(None, lambda: AI_Agent(request.prompt))
    instruction = "{" + instruction
    print(instruction)
    instruction = eval(instruction)
    if instruction["name"] == "add_memory":
        tools[instruction["name"]](**instruction["parameters"])
        return JSONResponse(content={"status": "success"})
    return JSONResponse(content={"status": "success"})

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
        instruction = "{" + instruction
        print("[instruction]", instruction)
        instruction = eval(instruction)
        if instruction["name"] == "no_call":
            prompt = request.prompt
        else:
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
        port=8080,
        reload=False
    )