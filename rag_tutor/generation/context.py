from __future__ import annotations

from rag_tutor.data.schema import Evidence


def build_context(results: list[dict], max_chars: int = 1800) -> tuple[str, list[Evidence]]:
    used_ids: set[str] = set()
    evidences: list[Evidence] = []
    blocks: list[str] = []
    total = 0

    for item in results:
        chunk = item["chunk"]
        chunk_id = chunk["chunk_id"]
        if chunk_id in used_ids:
            continue
        text = chunk["text"].strip()
        if not text:
            continue
        header = f"[{chunk_id}] {chunk.get('chapter') or 'unknown chapter'}"
        block = f"{header}\n{text}"
        if total + len(block) > max_chars and blocks:
            break
        if len(block) > max_chars:
            block = block[:max_chars].rstrip()
        blocks.append(block)
        total += len(block)
        used_ids.add(chunk_id)
        evidences.append(
            Evidence(
                chunk_id=chunk_id,
                doc_id=chunk["doc_id"],
                text=text,
                score=float(item.get("rerank_score", item["score"])),
                chapter=chunk.get("chapter"),
                chapter_index=chunk.get("chapter_index"),
                metadata={
                    "paragraph_index": chunk.get("paragraph_index"),
                    "sources": item.get("sources", {}),
                    "features": item.get("features", {}),
                },
            )
        )
    return "\n\n".join(blocks), evidences

