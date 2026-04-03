# 洛克王国 Wiki 智能助手（LangChain Agent + RAG）

本项目实现了一个可本地运行的智能助手：

- 使用 `LangChain Agent` 进行对话推理
- 通过 MediaWiki `api.php` 拉取页面列表（逻辑对齐 `import_lkwiki` / `import_pets`），再构建 `FAISS` 向量知识库（RAG）
- 通过 Web 页面与 Agent 对话

目标站点：`https://wiki.biligame.com/rocom/`

## 1) 环境准备

建议 Python 3.10+

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2) 配置模型

复制并修改环境变量文件：

```bash
copy .env.example .env
```

在 `.env` 中填写：

- `OPENAI_API_BASE=https://api.longcat.chat/openai`
- `OPENAI_API_KEY=你的key`
- `OPENAI_MODEL_NAME=LongCat-Flash-Thinking-2601`

## 3) 构建知识库

与 Coze 侧脚本一致：先用 `list=allpages` 枚举标题，再用 `action=parse` 取正文（失败则回退 GET 词条 URL）。

```bash
python backend/build_kb.py
```

若已有 `data/raw_pages.jsonl`（例如 1000+ 条），**只需重建向量库、不必再爬**：

```bash
python backend/build_kb.py --from-raw
```

环境变量（见 `.env.example`）：

- `WIKI_MODE=all`：全站列表 + `import_lkwiki.py` 的命名空间过滤
- `WIKI_MODE=pets`：仅宠物相关 + `import_pets.py` 的标题规则
- `WIKI_MAX_PAGES`：大于 0 时只处理前 N 个标题（API 量大时建议设限，如 `500`）
- **向量检索（长期默认）**：`EMBEDDING_MODEL_NAME` 默认 `BAAI/bge-small-zh-v1.5`，`build_kb` 与对话共用；**更换模型或从旧 MiniLM 索引升级时，必须删除 `data/kb` 后重建**（可保留 `raw_pages.jsonl`，执行 `python backend/build_kb.py --from-raw`）
- `RAG_SEARCH_TYPE=mmr`：可选，最大边际相关性检索；默认 `similarity`
- `RAG_USE_LEXICAL=true`：可选，在向量结果上叠加子串命中；中文 BGE 下一般保持 `false` 即可

Coze 云端加载项目环境变量可使用（需对应 SDK）：

```bash
eval $(python scripts/load_env.py)
```

执行后会生成：

- `data/raw_pages.jsonl`（url / title / 正文）
- `data/kb/`（FAISS 索引）

## 4) 启动后端 API

```bash
uvicorn backend.api:app --reload --host 127.0.0.1 --port 8000
```

接口：

- `GET /`：对话 Web 页（同机静态资源 `/assets`）
- `GET /health`
- `POST /chat`，请求体：`{"message":"你的问题"}`
- `POST /chat/stream`：SSE 流式对话（含 `reasoning_content` 等，供页面使用）

## 5) 打开 Web 页面

启动后端后，在浏览器打开：

`http://127.0.0.1:8000/`

（与 API 同一地址，无需再开 `http.server` 或 Live Server。）

## 6) 说明

- API 失败或过滤结果为空时，会回退到与 `import_lkwiki.py` / `import_pets.py` 相同的手写备选 URL 列表
- 嵌入模型由 `EMBEDDING_MODEL_NAME` 配置（默认 `BAAI/bge-small-zh-v1.5`），下载仍受 `HF_ENDPOINT` 等 Hugging Face 环境变量影响
- Agent 回答时优先检索知识库

## 参考

- 洛克王国wiki：[https://wiki.biligame.com/rocom/](https://wiki.biligame.com/rocom/)
- Coze 页面（用于参考项目形态）：[https://code.coze.cn/p/7623774469399527467/](https://code.coze.cn/p/7623774469399527467/)
