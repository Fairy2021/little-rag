# RAG Tutor

教学版 RAG 项目，按 Phase 逐步实现核心链路。

## 教学网页

打开 `docs/index.html` 查看函数调用路径、模块职责和核心原理。

## Phase 1

```bash
python scripts/make_demo_data.py
```

## Phase 2

```bash
python scripts/prepare_chunks.py --input rag_tutor/data/demo_hongloumeng.txt
python scripts/prepare_chunks.py --input orig_txt/hongloumeng.txt
```

## Phase 3

```bash
python scripts/build_index.py --chunks rag_tutor/data/chunks.jsonl
python scripts/query_dense.py "黛玉进京后拜见了谁" --top-k 5
```

## Phase 4

```bash
python scripts/query_hybrid.py "黛玉 初入 荣国府 贾母" --top-k 5
```

## Phase 5

```bash
python scripts/ask.py "黛玉进入荣府后见到了谁"
```

## Phase 6

```bash
$env:OPENAI_COMPAT_API_KEY="<your-key>"
$env:OPENAI_COMPAT_CHAT_URL="https://open.xiaojingai.com/v1/chat/completions"
$env:OPENAI_COMPAT_CHAT_MODEL="gpt-4o"
python scripts/ask.py "黛玉进入荣府后见到了谁" --llm openai

# Optional, only if your provider supports /v1/embeddings:
$env:OPENAI_COMPAT_BASE_URL="https://open.xiaojingai.com/v1"
python scripts/build_index.py --embedding openai --out-dir rag_tutor/index/openai_dense
```

## Phase 7

```bash
python eval/run_eval.py
python eval/make_qa50.py
python eval/run_eval.py --qa eval/qa_50.jsonl --index-dir rag_tutor/index/dense --dim 256 --k 5
```
