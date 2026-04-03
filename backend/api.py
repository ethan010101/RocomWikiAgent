import json
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.agent import build_agent

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_FRONTEND_DIR = _PROJECT_ROOT / "frontend"
_ASSETS_DIR = _FRONTEND_DIR / "assets"

app = FastAPI(title="Rocom Wiki Assistant API")

if _ASSETS_DIR.is_dir():
    app.mount(
        "/assets",
        StaticFiles(directory=str(_ASSETS_DIR)),
        name="assets",
    )

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


@app.get("/")
def serve_index():
    """浏览器访问 http://127.0.0.1:8000/ 加载对话页（静态图走 /assets/）。"""
    index = _FRONTEND_DIR / "index.html"
    if not index.is_file():
        raise HTTPException(status_code=404, detail="frontend/index.html 不存在")
    return FileResponse(index)


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


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """SSE：data 为 JSON。type=status | reasoning | content | error | done"""
    if agent_executor is None:
        raise HTTPException(
            status_code=503,
            detail="知识库未就绪。请在项目根目录执行: python backend/build_kb.py",
        )

    async def event_gen():
        try:
            async for evt in agent_executor.astream_sse_payloads(req.message):
                yield f"data: {json.dumps(evt, ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.exception("chat_stream failed")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
