"""LLM backend abstraction.

Two providers are supported:

* ``anthropic``          - the official Anthropic SDK (default, ``claude-opus-5``).
* ``openai_compatible``  - any OpenAI-compatible endpoint: Qwen3.5-VL served by vLLM,
                           Alibaba DashScope, etc. This matches the paper's Qwen setup.

Both expose the same surface:

    client.complete(system, user) -> str
    client.complete_vision(system, user, image_paths) -> str

Use :func:`extract_json` to robustly parse a JSON value out of a model reply.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import time
import re
from pathlib import Path
from typing import Any

from .config import Config


def _b64_image(path: str | os.PathLike) -> tuple[str, str]:
    """Return ``(media_type, base64_data)`` for an image file."""
    path = Path(path)
    media_type = mimetypes.guess_type(path.name)[0] or "image/png"
    data = base64.standard_b64encode(path.read_bytes()).decode("utf-8")
    return media_type, data



def _with_retry(fn, tries: int = 3, wait: float = 2.0):
    """Retry a transient server failure, the same way for every caller.

    Without this the retry a method happens to contain becomes part of its score.
    WikiWalk swallows an exception at each of its ~19 steps and still answers; a baseline
    makes one call and is marked wrong when that call returns a 500. Measured over the
    finished runs, the baselines lost 0.3-0.8% of their items that way and WikiWalk lost
    none -- a gap in our favour that came from where the try/except sat, not from the
    retrieval being compared. Putting it in the client gives every method the same
    tolerance. A 500 here is the server dropping a request under load; it is not the
    model failing to answer, and it should not be scored as one.
    """
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 -- re-raised below if it never succeeds
            last = exc
            if i + 1 < tries:
                time.sleep(wait * (i + 1))
    raise last

class LLMClient:
    """Thin wrapper over the configured provider."""

    def __init__(self, config: Config):
        self.provider = config.get("llm", "provider", default="anthropic")
        self.model = config.get("llm", "model", default="claude-opus-5")
        self.max_tokens = int(config.get("llm", "max_tokens", default=8000))
        self.temperature = float(config.get("llm", "temperature", default=0.0))
        # Qwen3.5 is a thinking model and reasons by default. On the pipeline's
        # short structured-JSON tasks that chain of thought is the entire runtime
        # cost (thousands of tokens to answer "which entity is this image"), so it
        # is off unless a task asks for it.
        self.enable_thinking = bool(config.get("llm", "enable_thinking", default=False))
        self._config = config

        if self.provider == "anthropic":
            import anthropic  # imported lazily so the other backend need not be installed

            self._client = anthropic.Anthropic()
        elif self.provider == "openai_compatible":
            from openai import OpenAI

            base_url = config.get("llm", "base_url", default=None)
            key_env = config.get("llm", "api_key_env", default="OPENAI_API_KEY")
            api_key = os.environ.get(key_env, "EMPTY")  # vLLM accepts any non-empty key
            self._client = OpenAI(base_url=base_url, api_key=api_key)
        else:
            raise ValueError(f"Unknown llm.provider: {self.provider!r}")

    def _extra_body(self) -> dict:
        """Server-side knobs for OpenAI-compatible backends (vLLM passes these on).

        `chat_template_kwargs` is a vLLM extension. Servers that do not know it reject the
        whole request with "Unknown parameter", which surfaces as every item failing at once and
        being scored wrong -- a row of zeros that looks like a model result rather than a
        transport error. It is therefore sent only to servers that understand it.
        """
        if self.enable_thinking or not self._is_vllm():
            return {}
        return {"chat_template_kwargs": {"enable_thinking": False}}

    def _token_kwargs(self, n: int | None) -> dict:
        """The cap on generated tokens, under whichever name the server accepts.

        vLLM takes `max_tokens`; some other OpenAI-compatible servers require
        `max_completion_tokens` instead. Sending the wrong one fails the request outright,
        so every item errors and the row reads as a score of zero.
        """
        n = n or self.max_tokens
        if self._is_vllm():
            return {"max_tokens": n, "temperature": self.temperature}
        # Such servers may reject an explicit temperature, so sampling is left to them.
        # Where the budget also covers internal reasoning, a cap sized for the answer
        # alone can be spent entirely before any visible text and return an empty reply
        # rather than an error, so the reasoning gets its own room on top of it.
        return {"max_completion_tokens": max(n * 4, n + 600)}

    def _is_vllm(self) -> bool:
        """True for a self-hosted vLLM server."""
        base = str(getattr(self._client, "base_url", "") or "")
        return not re.search(r"(^|//)(api\.openai\.com|api\.anthropic\.com)", base)

    # ------------------------------------------------------------------ text
    def complete(self, system: str, user: str, max_tokens: int | None = None) -> str:
        if self.provider == "anthropic":
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens or self.max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return "".join(b.text for b in resp.content if b.type == "text")

        # openai_compatible
        resp = _with_retry(lambda: self._client.chat.completions.create(
            model=self.model,
            **self._token_kwargs(max_tokens),
                        messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            extra_body=self._extra_body(),
        ))
        return resp.choices[0].message.content or ""

    # ---------------------------------------------------------------- vision
    def complete_vision(
        self, system: str, user: str, image_paths: list[str], max_tokens: int | None = None
    ) -> str:
        if self.provider == "anthropic":
            content: list[dict[str, Any]] = []
            for p in image_paths:
                media_type, data = _b64_image(p)
                content.append(
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": media_type, "data": data},
                    }
                )
            content.append({"type": "text", "text": user})
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens or self.max_tokens,
                system=system,
                messages=[{"role": "user", "content": content}],
            )
            return "".join(b.text for b in resp.content if b.type == "text")

        # openai_compatible (Qwen-VL style image_url with a data URI)
        #
        # Images dominate the prompt: three of them plus the evidence can exceed a 32k
        # context, and the server answers 400 rather than truncating. Dropping the last
        # image and retrying degrades the request instead of losing the item -- a run of
        # 1,600 must not die because one of them was too long, and scoring it wrong would
        # blame the model for a context limit.
        paths = list(image_paths)
        while True:
            content = [{"type": "text", "text": user}]
            for p in paths:
                media_type, data = _b64_image(p)
                content.insert(
                    0,
                    {"type": "image_url",
                     "image_url": {"url": f"data:{media_type};base64,{data}"}},
                )
            try:
                # retried like the text path; shedding an image is a separate remedy for
                # a separate failure, and is handled below
                resp = _with_retry(lambda: self._client.chat.completions.create(
                    model=self.model,
                    **self._token_kwargs(max_tokens),
                                        messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": content},
                    ],
                    extra_body=self._extra_body(),
                ))
            except Exception as exc:  # noqa: BLE001 - provider-specific error types
                if paths and "maximum context length" in str(exc):
                    paths = paths[:-1]
                    self.context_drops = getattr(self, "context_drops", 0) + 1
                    continue
                raise
            return resp.choices[0].message.content or ""


# --------------------------------------------------------------------------- json
_JSON_BLOCK = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def extract_json(text: str, default: Any = None) -> Any:
    """Best-effort extraction of a JSON value from a model reply.

    Handles fenced code blocks and leading/trailing prose. Returns ``default`` if
    nothing parseable is found.
    """
    if not text:
        return default
    # 1) fenced ```json ... ``` block
    m = _JSON_BLOCK.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # 2) whole string
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    # 3) first {...} or [...] span
    for opener, closer in (("[", "]"), ("{", "}")):
        start = text.find(opener)
        end = text.rfind(closer)
        if 0 <= start < end:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    return default
