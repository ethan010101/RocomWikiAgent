import json
import logging
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.agent import build_agent
from backend import conversation_store, online_eval

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
    history_messages: list = Field(default_factory=list)
    context_state: dict = Field(default_factory=dict)


class ChatResponse(BaseModel):
    answer: str


class ConversationsPayload(BaseModel):
    conversations: list = Field(default_factory=list)
    currentConvId: str | None = None


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


@app.get("/api/conversations")
def get_conversations():
    """读取服务端持久化的对话列表（与前端 localStorage 结构一致）。"""
    s = conversation_store.read_state()
    return {
        "conversations": s["conversations"],
        "currentConvId": s["currentConvId"],
    }


@app.post("/api/conversations")
def save_conversations(body: ConversationsPayload):
    """保存完整对话状态；前端在每次变更后防抖同步。"""
    conversation_store.write_state(body.conversations, body.currentConvId)
    return {"ok": True}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if agent_executor is None:
        raise HTTPException(
            status_code=503,
            detail="知识库未就绪。请在项目根目录执行: python backend/build_kb.py",
        )
    trace = agent_executor.invoke_with_trace(
        {
            "input": req.message,
            "history_messages": req.history_messages,
            "context_state": req.context_state,
        }
    )
    answer = trace.get("output", "未获取到回答")
    online_eval.schedule_log(
        question=req.message,
        answer=answer if isinstance(answer, str) else str(answer),
        docs=trace.get("docs") or [],
        latency_ms=float(trace.get("latency_ms", 0)),
        route="POST /chat",
    )
    return ChatResponse(answer=answer)


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """SSE：data 为 JSON。type=status | reasoning | content | error | done"""
    if agent_executor is None:
        raise HTTPException(
            status_code=503,
            detail="知识库未就绪。请在项目根目录执行: python backend/build_kb.py",
        )

    async def event_gen():
        capture: dict = {}
        t0 = time.perf_counter()
        content_parts: list[str] = []
        try:
            async for evt in agent_executor.astream_sse_payloads(
                req.message,
                history_messages=req.history_messages,
                context_state=req.context_state,
                eval_capture=capture,
            ):
                if isinstance(evt, dict) and evt.get("type") == "content":
                    content_parts.append(evt.get("delta") or "")
                yield f"data: {json.dumps(evt, ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.exception("chat_stream failed")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"
            return
        finally:
            answer = "".join(content_parts)
            docs = capture.get("docs") or []
            online_eval.schedule_log(
                question=req.message,
                answer=answer,
                docs=docs,
                latency_ms=(time.perf_counter() - t0) * 1000,
                route="POST /chat/stream",
            )

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
