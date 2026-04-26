from __future__ import annotations


def build_prompt(question: str, context: str) -> str:
    return f"""你是一个严格基于证据回答的 RAG 助手。

规则：
1. 只能使用【证据】里的内容回答。
2. 如果证据不足，回答“不知道”。
3. 每条结论后必须标注引用，引用格式使用 [chunk_id]。

【问题】
{question}

【证据】
{context}

【答案】
"""

