from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from rag_tutor.data.schema import Evidence
from rag_tutor.generation.llm import LLM


class OpenAICompatibleLLM(LLM):
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        chat_url: str | None = None,
        timeout: int = 60,
    ) -> None:
        self.model = model or os.getenv("OPENAI_COMPAT_CHAT_MODEL", "gpt-4o-mini")
        self.api_key = api_key or os.getenv("OPENAI_COMPAT_API_KEY")
        self.url = chat_url or os.getenv("OPENAI_COMPAT_CHAT_URL") or _join_url(
            base_url or os.getenv("OPENAI_COMPAT_BASE_URL", "https://api.openai.com/v1"),
            "chat/completions",
        )
        self.timeout = timeout
        if not self.api_key:
            raise RuntimeError("OPENAI_COMPAT_API_KEY is required for OpenAICompatibleLLM")

    def generate(self, question: str, evidences: list[Evidence], prompt: str) -> str:
        del question, evidences
        data = _post_json(
            self.url,
            self.api_key,
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
            },
            timeout=self.timeout,
        )
        return data["choices"][0]["message"]["content"].strip()


def _post_json(url: str, api_key: str, payload: dict, timeout: int = 60) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI-compatible request failed: HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"OpenAI-compatible request failed: {exc.reason}") from exc


def _join_url(base_url: str, path: str) -> str:
    return base_url.rstrip("/") + "/" + path.lstrip("/")

