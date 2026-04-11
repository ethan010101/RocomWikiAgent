# 洛克王国 Wiki 智能助手（RAG + 多轮上下文流水线）

**核心思路**：使用 LLM 提取 `entities`、`info_type` 与 `recommend_entity`，通过 `entities` 与 `recommend_entity` 解析指代，再将三者做**笛卡尔积**组合后分别向量检索，以缓解指代消解、最优推荐与实体抽取误差等问题。

本项目是一个可本地运行的 BWIKI 问答助手：

- **检索 + 生成**：用 LangChain（FAISS 向量库、`ChatOpenAI`、提示模板）做「先检索再生成」，避免工具型 Agent 在部分网关上不调用检索而胡编。
- **多轮流水线**（`backend/rag_pipeline/`）：实体时间线抽取、问句类型与检索种子解析、指代/多实体消歧、可选检索门控与会话摘要，与单次向量检索组合使用。
- **建库**：通过 MediaWiki `api.php` 枚举页面（逻辑与历史脚本 `import_lkwiki` / `import_pets` 对齐，现收敛在 `backend/wiki_sources.py`），再切块写入 **FAISS**。
- **前端**：`frontend/index.html` 由 FastAPI 同机托管；支持流式 SSE（含推理字段等）。
- **评测与日志**：离线检索评测、半自动构造 golden、RAGAS 指标，以及可选的线上对话 JSONL 日志。

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

在 `.env` 中至少填写对话网关（OpenAI 兼容），例如：

- `OPENAI_API_BASE=https://api.longcat.chat/openai`
- `OPENAI_API_KEY=你的key`
- `OPENAI_MODEL_NAME=LongCat-Flash-Thinking-2601`

多轮流水线、指代消解、实体抽取等可选模型与预算变量见 `.env.example` 中「多轮上下文流水线」等章节。

## 3) 构建知识库

流程：先用 `list=allpages` 枚举标题，再用 `action=parse` 取正文（失败则回退 GET 词条 URL）。

```bash
python backend/build_kb.py
```

若已有 `data/raw_pages.jsonl`（例如 1000+ 条），**只需重建向量库、不必再爬**：

```bash
python backend/build_kb.py --from-raw
```

环境变量（见 `.env.example`）：

- `WIKI_MODE=all`：全站列表 + `wiki_sources` 中的命名空间过滤
- `WIKI_MODE=pets`：仅宠物相关 + 标题规则
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
- `GET /health`：返回 `kb_ready` 等状态
- `POST /chat`：非流式对话  
  - 请求体：`{"message":"你的问题","history_messages":[],"context_state":{}}`  
    - `history_messages`：多轮用户/助手消息列表（与前端一致）  
    - `context_state`：服务端返回的 `turn_context` 可回传，用于跨请求延续流水线状态  
  - 响应：`{"answer":"...","turn_context":{...}}`（`turn_context` 可能为 `null`）
- `POST /chat/stream`：SSE 流式对话；`data` 为 JSON，常见 `type`：`status`、`entity_extract`、`reasoning`、`content`、`error`、`done`（供页面展示进度与思考内容）
- `GET /api/conversations`：读取服务端持久化的对话列表与当前会话 id
- `POST /api/conversations`：保存完整状态，请求体 `{"conversations":[...],"currentConvId":"..."}`（页面在变更后防抖同步，关闭标签前 `sendBeacon` 再存一次）

对话默认写入项目根 `data/conversations.json`（已加入 `.gitignore`）；清除浏览器缓存后只要后端数据仍在，打开页面会从服务端恢复。可选环境变量 **`CONVERSATIONS_STORE_PATH`** 自定义路径。

## 5) 打开 Web 页面

启动后端后，在浏览器打开：

`http://127.0.0.1:8000/`

（与 API 同一地址，无需再开 `http.server` 或 Live Server。）

## 6) 多轮 RAG 流水线（概要）

核心代码在 `backend/rag_pipeline/`：

| 模块 | 作用 |
|------|------|
| `orchestrate.py` | 单轮 `prepare` / `run`，串联检索与生成 |
| `query_resolve.py` | 解析用户句，产出检索用种子、路由（代词/事实/多实体等） |
| `entity_extract_llm.py` | 从近期对话抽取实体时间线（可选独立模型） |
| `pronoun_resolve_llm.py` | 代词/多答案路径下的语义消歧与多路检索 |
| `session_state.py` / `turn_context.py` | 会话状态与轮次上下文合并、继承 |
| `retrieval_gate.py` | 可选：要求片段命中解析出的主体 |

环境与开关仍以 `.env` 为准（如 `RAG_CTX_*`、`ENTITY_EXTRACT_*`、`PRONOUN_*`、`RAG_CONTEXT_SUBJECT_GATE` 等）。

## 7) 测评 / 评测

以下均在**项目根目录**执行，且需已配置 `.env`、已构建 `data/kb/`（RAG 对话链与检索一致）。更全的环境变量说明见 `.env.example`。

### 7.1 离线检索评测（`eval_rag.py`）

衡量检索返回的片段是否包含你标注的**子串**与 **URL 片段**（不调用 RAGAS）。

```bash
python backend/eval_rag.py --dataset data/eval/golden.jsonl
python backend/eval_rag.py --dataset data/eval/golden.jsonl --with-generation
python backend/eval_rag.py --dataset data/eval/golden.jsonl --out data/eval/eval_last.json
```

| 参数 | 含义 |
|------|------|
| `--dataset` | JSONL 路径，默认 `data/eval/golden.jsonl` |
| `--with-generation` | 额外走完整 RAG 生成；若数据里有 `must_contain_in_answer` 则检查答案是否含这些子串 |
| `--out` | 将明细写入 JSON |

**数据集（每行一个 JSON）** 常用字段：

| 字段 | 说明 |
|------|------|
| `question` | 必填，用户问题 |
| `id` | 可选，便于对照日志 |
| `expected_substrings` | 可选，列表；每项须出现在至少一条检索片段的**标题或正文**中 |
| `expected_source_substrings` | 可选，列表；每项须出现在某条片段的 **source URL** 中 |
| `must_contain_in_answer` | 可选；仅与 `--with-generation` 一起用时检查生成答案 |
| `ground_truth` | 可选；本脚本不解析，供 RAGAS 或人工校对参考 |

未标注 `expected_*` 的样本在统计里记为「检索 SKIP」，不计入通过率。示例见 `data/eval/golden.example.jsonl`。

有检索标注时，若存在未通过样本，进程退出码为 **1**（便于 CI）。

### 7.2 半自动生成评测集（`gen_eval_golden.py`）

从 `raw_pages.jsonl` 按与建库相同的切分抽样，调用 LLM 生成 `question` 与 `expected_substrings`（子串会校验是否真在原文中）。生成后建议**人工抽检**再用于 7.1 / 7.3。

```bash
python backend/gen_eval_golden.py --count 20 --out data/eval/golden.generated.jsonl
python backend/gen_eval_golden.py --count 10 --seed 7 --raw data/raw_pages.jsonl
```

常用参数：`--seed`、`--max-chars`、`--sleep`（请求间隔）。可选环境变量 **`GEN_EVAL_TEMPERATURE`**（默认约 `0.35`）。

### 7.3 RAGAS 离线评测（`eval_rag_ragas.py`）

对每条样本：用当前 **RAG 检索 + 生成**，再计算 **faithfulness**（答案陈述是否被上下文支持）与 **answer_relevancy**（与问题的相关性代理）。**耗时与 API 调用次数远高于 7.1**，建议先用 `--limit` 试跑。

```bash
python backend/eval_rag_ragas.py --dataset data/eval/golden.generated.jsonl --limit 10
python backend/eval_rag_ragas.py --dataset data/eval/golden.generated.jsonl --out data/eval/ragas_last.json
```

| 参数 | 含义 |
|------|------|
| `--dataset` | JSONL，需含 `question` |
| `--limit` | 只评前 N 条；`0` 表示全部 |
| `--max-contexts` / `--max-chunk-chars` | 控制送入 RAGAS 的检索片段条数与单条长度 |
| `--out` | 合并明细 JSON（`ragas_last.json` 已加入 `.gitignore`，勿提交仓库） |

**依赖**：`requirements.txt` 中的 `ragas`、`datasets`、`pandas`。

**环境变量（摘录）**：

| 变量 | 作用 |
|------|------|
| `RAGAS_OPENAI_MODEL_NAME` | 评测用聊天模型，建议 **chat**（如 `deepseek-chat`）；未设则沿用 `OPENAI_MODEL_NAME` |
| `RAGAS_ANSWER_RELEVANCY_STRICTNESS` | 默认 `1`；仅支持 `n=1` 的网关（如 DeepSeek）若设过大可能 400 |
| `RAGAS_EVAL_TIMEOUT` | 单次相关超时（秒） |

实现上使用 **旧版** `ragas.metrics` 指标 + **LangChain `ChatOpenAI`** + 与建库相同的 **本地 BGE**（`EMBEDDING_*`），与 `metrics.collections` 新指标不混用。中文长回答下分数可能整体偏低，宜作**调参前后对比**而非绝对门槛。

### 7.4 线上测评日志（对话结束后异步写入）

开启后，在 **`POST /chat`** 与 **`POST /chat/stream`** 完成响应后，**后台线程**追加写入 JSONL，**不阻塞**用户。

在 `.env` 中设置：

| 变量 | 作用 |
|------|------|
| `ONLINE_EVAL_ENABLED` | `true` / `1` 开启 |
| `ONLINE_EVAL_LOG_PATH` | 日志路径，默认 `data/eval/online_eval.jsonl` |
| `ONLINE_EVAL_QUICK_SCORE` | 可选，用 BGE 算问句与答案前段的余弦相似度，作粗 relevancy 代理 |
| `ONLINE_EVAL_MAX_QUESTION_CHARS` | 日志中问题最大长度 |

日志含时间、`question`/`question_hash`、答案长度与预览、检索条数、去重后的来源列表、耗时等。**该文件默认已加入 `.gitignore`**（含用户问题片段，勿提交）。

## 8) 说明

- API 失败或过滤结果为空时，会回退到与 `wiki_sources` 中相同的手写备选 URL 列表
- 嵌入模型由 `EMBEDDING_MODEL_NAME` 配置（默认 `BAAI/bge-small-zh-v1.5`），下载受 `HF_ENDPOINT` 等 Hub 环境变量影响；若访问 `hf-mirror.com` 出现 WinError 10060，可去掉 `HF_ENDPOINT` 改用代理访问官方，或将模型放到本地目录并把 `EMBEDDING_MODEL_NAME` 设为该路径（见 `.env.example`）
- 回答始终基于检索到的 wiki 片段生成；多轮时由 `rag_pipeline` 决定检索查询如何展开

## 9) 参考

- 洛克王国wiki：[https://wiki.biligame.com/rocom/](https://wiki.biligame.com/rocom/)
- Coze 页面（用于参考项目形态）：[https://code.coze.cn/p/7623774469399527467/](https://code.coze.cn/p/7623774469399527467/)
