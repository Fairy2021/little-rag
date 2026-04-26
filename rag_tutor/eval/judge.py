from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod

from rag_tutor.generation import OpenAICompatibleLLM


class Judge(ABC):
    @abstractmethod
    def score_answer(self, question: str, prediction: str, reference: str) -> dict:
        raise NotImplementedError

    @abstractmethod
    def score_support(self, claim: str, evidence: str) -> dict:
        raise NotImplementedError


class KeywordJudge(Judge):
    def score_answer(self, question: str, prediction: str, reference: str) -> dict:
        del question
        ok = bool(reference and reference in prediction)
        return {"score": int(ok), "reason": "reference substring match" if ok else "missing reference"}

    def score_support(self, claim: str, evidence: str) -> dict:
        words = {w for w in re.findall(r"[\u4e00-\u9fff]{2,}", claim)}
        ok = not words or len(words - set(evidence)) / max(len(words), 1) <= 0.25
        return {"supported": int(ok), "reason": "keyword support heuristic"}


class LLMJudge(Judge):
    def __init__(self, model: str | None = None) -> None:
        self.llm = OpenAICompatibleLLM(model=model)

    def score_answer(self, question: str, prediction: str, reference: str) -> dict:
        prompt = (
            "你是严格的中文问答评估器。判断模型答案是否正确回答问题。"
            "允许同义表达；明显错误、答非所问、缺少关键事实判 0。\n\n"
            f"问题：{question}\n标准答案：{reference}\n模型答案：{prediction}\n\n"
            '只输出 JSON：{"score":0或1,"reason":"一句话原因"}'
        )
        return _json_result(self.llm.generate(question, [], prompt), "score")

    def score_support(self, claim: str, evidence: str) -> dict:
        prompt = (
            "你是严格的事实支撑判断器。只根据证据判断陈述是否被直接支持。"
            "证据没有明确支持则判 0。\n\n"
            f"陈述：{claim}\n证据：{evidence}\n\n"
            '只输出 JSON：{"supported":0或1,"reason":"一句话原因"}'
        )
        return _json_result(self.llm.generate(claim, [], prompt), "supported")


def make_judge(kind: str, model: str | None = None) -> Judge:
    return LLMJudge(model=model) if kind == "llm" else KeywordJudge()


def _json_result(text: str, key: str) -> dict:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return {key: 0, "reason": f"non-json judge output: {text[:80]}"}
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {key: 0, "reason": f"bad json judge output: {text[:80]}"}
    data[key] = int(data.get(key, 0))
    data.setdefault("reason", "")
    return data

