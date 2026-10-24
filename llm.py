from __future__ import annotations

import json
import os
import re
import threading
import time

THINK = re.compile(r"<think>.*?</think>", re.S)


class BudgetExceeded(RuntimeError):
    pass


class ChatClient:
    """OpenAI-compatible chat client; works with any endpoint exposing /chat/completions."""

    def __init__(self, base_url: str | None = None, api_key: str | None = None, max_usd: float | None = None,
                 usd_per_m_in: float = 0.30, usd_per_m_out: float = 1.20):
        from openai import OpenAI

        self._c = OpenAI(
            base_url=base_url or os.environ["LLM_BASE_URL"],
            api_key=api_key or os.environ["LLM_API_KEY"],
        )
        # e.g. LLM_EXTRA_BODY='{"chat_template_kwargs": {"thinking": false}}' to disable thinking on DeepSeek V4.1
        self._extra = json.loads(os.environ["LLM_EXTRA_BODY"]) if os.environ.get("LLM_EXTRA_BODY") else None
        self.max_usd, self.rates = max_usd, (usd_per_m_in / 1e6, usd_per_m_out / 1e6)
        self.spent_usd, self._lock = 0.0, threading.Lock()  # list-price estimate: ignores any cached-input discount

    def models(self):
        try:
            return [m.id for m in self._c.models.list().data]
        except Exception:
            return None

    def chat(self, model, system, user, temperature=0.7, max_tokens=2048):
        if self.max_usd is not None and self.spent_usd >= self.max_usd:
            raise BudgetExceeded(f"spend cap ${self.max_usd:.2f} reached (spent ${self.spent_usd:.2f})")
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        for attempt in range(4):
            t0 = time.time()
            try:
                r = self._c.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_body=self._extra,
                )
                msg = r.choices[0].message
                raw = msg.content or ""
                text = THINK.sub("", raw).strip()
                u = r.usage
                with self._lock:
                    self.spent_usd += (u.prompt_tokens * self.rates[0] + u.completion_tokens * self.rates[1]) if u else 0.0
                details = getattr(u, "prompt_tokens_details", None)
                return text, {
                    "in": u.prompt_tokens if u else 0,
                    "out": u.completion_tokens if u else 0,
                    # extra keys are for offline logging only; callers sum just in/out
                    "cached_in": getattr(details, "cached_tokens", None),
                    "latency": round(time.time() - t0, 3),
                    "http_attempts": attempt + 1,
                    "finish": r.choices[0].finish_reason,
                    "raw": raw,
                    "reasoning": getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None),
                    "resp_model": getattr(r, "model", None),
                    "resp_id": getattr(r, "id", None),
                }
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2**attempt)
