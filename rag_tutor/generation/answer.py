from __future__ import annotations

from rag_tutor.data.schema import Answer
from rag_tutor.generation.context import build_context
from rag_tutor.generation.llm import LLM
from rag_tutor.generation.prompt import build_prompt


def answer_question(
    question: str,
    ranked_results: list[dict],
    llm: LLM,
    max_context_chars: int = 1800,
) -> Answer:
    context, evidences = build_context(ranked_results, max_chars=max_context_chars)
    prompt = build_prompt(question, context)
    text = llm.generate(question, evidences, prompt)
    citations = [evidence.chunk_id for evidence in evidences if f"[{evidence.chunk_id}]" in text]
    return Answer(
        question=question,
        answer=text,
        evidences=evidences,
        citations=citations,
        metadata={"prompt": prompt, "context_chars": len(context)},
    )

