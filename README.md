# Little RAG

一个从零手写的教学版 RAG 项目，语料是《红楼梦》。项目目标不是追求复杂框架，而是把 RAG 的核心链路拆开，让初学者能看懂每一步：文本如何被切成 chunk，chunk 如何进入索引，问题如何被检索、融合、重排、生成，最后又如何评测。

教学网页：

- GitHub Pages: https://fairy2021.github.io/little-rag/
- 备用预览: https://htmlpreview.github.io/?https://github.com/Fairy2021/little-rag/blob/main/docs/index.html
- 本地文件: `docs/index.html`

## 这个项目覆盖什么

主流程：

```text
ingest -> normalize -> chunk -> embed -> index
-> retrieve(dense + bm25) -> fusion(RRF)
-> rerank -> context build(citations) -> generate
-> eval(retrieval + generation + grounding)
```

核心特点：

- 不使用 LangChain / LlamaIndex，核心逻辑手写。
- 默认完全离线运行：`FakeEmbedding + DummyLLM`。
- 可选接入 OpenAI-compatible chat/completions。
- 检索支持 dense、BM25、RRF hybrid。
- 生成答案强制带 citation。
- 评测支持 Recall@k、MRR、nDCG@k、latency、LLM judge、grounding judge。

## 目录结构

```text
rag_tutor/
  data/          # 数据结构、读取、清洗、chunk
  index/         # 本地索引输出目录，默认不提交
  models/        # FakeEmbedding / OpenAI-compatible embedding
  retrieval/     # DenseIndex / BM25 / RRF / rerank
  generation/    # context / prompt / DummyLLM / OpenAI-compatible LLM
  eval/          # eval metrics / judge / runner
scripts/         # 构建、查询、问答脚本
eval/            # eval CLI 和 QA 文件
docs/            # 教学网页，GitHub Pages 使用
orig_txt/        # 原始文本，本地使用，不提交
```

## 快速开始

生成 demo 数据：

```bash
python scripts/make_demo_data.py
```

把文本切成 chunks：

```bash
python scripts/prepare_chunks.py --input rag_tutor/data/demo_hongloumeng.txt --out rag_tutor/data/demo_chunks.jsonl --chunk-size 80 --overlap 20
```

构建 demo 索引：

```bash
python scripts/build_index.py --chunks rag_tutor/data/demo_chunks.jsonl --out-dir rag_tutor/index/demo_dense --dim 128
```

查询 dense 检索：

```bash
python scripts/query_dense.py "黛玉进京后拜见了谁" --index-dir rag_tutor/index/demo_dense --dim 128
```

查询 hybrid 检索：

```bash
python scripts/query_hybrid.py "黛玉进京后拜见了谁" --index-dir rag_tutor/index/demo_dense --dim 128
```

端到端问答：

```bash
python scripts/ask.py "黛玉进京后拜见了谁" --index-dir rag_tutor/index/demo_dense --dim 128
```

运行评测：

```bash
python eval/run_eval.py
```

## 使用整本《红楼梦》

把原始文本放在：

```text
orig_txt/hongloumeng.txt
```

然后运行：

```bash
python scripts/prepare_chunks.py --input orig_txt/hongloumeng.txt --out rag_tutor/data/chunks.jsonl
python scripts/build_index.py --chunks rag_tutor/data/chunks.jsonl --out-dir rag_tutor/index/dense --dim 256
python scripts/ask.py "黛玉进入荣府后见到了谁" --index-dir rag_tutor/index/dense --dim 256
```

`orig_txt/`、`rag_tutor/data/chunks.jsonl` 和 `rag_tutor/index/` 默认不提交到 GitHub。

## 可选：接入真实 LLM

默认不需要 API key。要使用 OpenAI-compatible chat/completions：

```powershell
$env:OPENAI_COMPAT_API_KEY="<your-key>"
$env:OPENAI_COMPAT_CHAT_URL="https://open.xiaojingai.com/v1/chat/completions"
$env:OPENAI_COMPAT_CHAT_MODEL="gpt-4o"

python scripts/ask.py "黛玉进京后拜见了谁" --llm openai
```

运行 LLM judge：

```powershell
python eval/run_eval.py --judge llm --support-judge llm
```

不要把 API key 写进代码、README 或 `.env` 以外的文件。

## 学习建议

先打开教学网页：

```text
docs/index.html
```

推荐阅读顺序：

1. 看第一页的 RAG 系统总览图。
2. 跑 demo 数据和 demo 索引。
3. 对照网页理解 `scripts/ask.py` 的完整调用链。
4. 再看 `rag_tutor/retrieval/` 和 `rag_tutor/generation/`。
5. 最后用 `eval/run_eval.py` 理解 RAG 怎么评测。

## GitHub Pages

本项目的教学网页位于 `docs/`，并带有 GitHub Actions 自动发布配置。优先访问：

```text
https://fairy2021.github.io/little-rag/
```

如果首次访问时 404，去 GitHub 仓库页面检查 Pages：

```text
Settings -> Pages -> Source: GitHub Actions
```

备用预览地址可以直接打开 HTML：

```text
https://htmlpreview.github.io/?https://github.com/Fairy2021/little-rag/blob/main/docs/index.html
```
