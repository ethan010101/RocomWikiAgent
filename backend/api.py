import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.agent import build_agent

logger = logging.getLogger(__name__)

app = FastAPI(title="Rocom Wiki Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent_executor = None


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str


@app.on_event("startup")
def startup_event():
    global agent_executor
    try:
        agent_executor = build_agent()
        logger.info("Knowledge base loaded, agent ready.")
    except FileNotFoundError as e:
        agent_executor = None
        logger.warning(
            "Agent not started: %s — 请先运行: python backend/build_kb.py",
            e,
        )


@app.get("/health")
def health():
    return {"ok": True, "kb_ready": agent_executor is not None}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if agent_executor is None:
        raise HTTPException(
            status_code=503,
            detail="知识库未就绪。请在项目根目录执行: python backend/build_kb.py",
        )
    result = agent_executor.invoke({"input": req.message})
    return ChatResponse(answer=result.get("output", "未获取到回答"))
