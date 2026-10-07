"""Plain-language summary of the evidence, with an optional LLM rewrite behind a small interface.

Rules that the code enforces:
* The LLM receives measured facts only: calibrated probability, operating threshold, neighbour
  label counts and the strongest saliency zone. It never receives the predicted label as a fact.
* The image is not sent unless PNEUMOVIT_LLM_SEND_IMAGE=true (privacy).
* The LLM text is accepted only if every number in it occurs in the facts and it uses no
  certainty words. Otherwise the deterministic template is used.
* Every summary ends with the not-a-diagnosis banner.
"""

from __future__ import annotations

import base64
import io
import json
import re
import urllib.request
from typing import Callable, Protocol

import numpy as np

from .. import BANNER

CERTAINTY = re.compile(r"\b(definitely|certainly|confirms?|confirmed|diagnos(?:is|ed|e)|proves?|clearly has)\b", re.I)
NUMBER = re.compile(r"\d+(?:\.\d+)?")


class LLMClient(Protocol):
    name: str

    def generate(self, prompt: str, image_png: bytes | None = None) -> str: ...


class FakeLLM:
    """Offline, deterministic: returns the template summary that it finds in the prompt."""

    name = "fake"

    def generate(self, prompt: str, image_png: bytes | None = None) -> str:
        return prompt.split("TEMPLATE:\n", 1)[-1].strip()


Transport = Callable[[str, dict, bytes], bytes]


def _post(url: str, headers: dict, body: bytes) -> bytes:  # pragma: no cover - network
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


class GeminiClient:
    name = "gemini"

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash", transport: Transport = _post):
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is not set")
        self._key, self.model, self._post = api_key, model, transport

    def generate(self, prompt: str, image_png: bytes | None = None) -> str:
        parts = [{"text": prompt}]
        if image_png is not None:
            parts.append({"inline_data": {"mime_type": "image/png", "data": base64.b64encode(image_png).decode()}})
        body = json.dumps({"contents": [{"parts": parts}], "generationConfig": {"temperature": 0.2}}).encode()
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        raw = self._post(url, {"Content-Type": "application/json", "x-goog-api-key": self._key}, body)
        data = json.loads(raw)
        return data["candidates"][0]["content"]["parts"][0]["text"]


class OpenAICompatibleClient:
    name = "openai"

    def __init__(self, api_key: str, model: str = "gpt-4.1-mini", base_url: str = "https://api.openai.com/v1",
                 transport: Transport = _post):
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set")
        self._key, self.model, self.base_url, self._post = api_key, model, base_url.rstrip("/"), transport

    def generate(self, prompt: str, image_png: bytes | None = None) -> str:
        content: list[dict] = [{"type": "text", "text": prompt}]
        if image_png is not None:
            content.append({"type": "image_url", "image_url": {
                "url": "data:image/png;base64," + base64.b64encode(image_png).decode()}})
        body = json.dumps({"model": self.model, "temperature": 0.2,
                           "messages": [{"role": "user", "content": content}]}).encode()
        raw = self._post(f"{self.base_url}/chat/completions",
                         {"Content-Type": "application/json", "Authorization": f"Bearer {self._key}"}, body)
        return json.loads(raw)["choices"][0]["message"]["content"]


def make_client(settings) -> LLMClient | None:
    if settings.llm_provider == "none":
        return None
    if settings.llm_provider == "fake":
        return FakeLLM()
    if settings.llm_provider == "gemini":
        return GeminiClient(settings.api_keys.get("GOOGLE_API_KEY", ""), settings.llm_model or "gemini-2.5-flash")
    return OpenAICompatibleClient(settings.api_keys.get("OPENAI_API_KEY", ""), settings.llm_model or "gpt-4.1-mini",
                                  settings.llm_base_url or "https://api.openai.com/v1")


def facts_from(result: dict) -> dict:
    """The only facts that a summary (template or LLM) can use. No predicted label is included."""
    nb = result["neighbours_summary"]
    return {
        "pneumonia_probability": round(result["probability"], 2),
        "operating_threshold": round(result["threshold"], 2),
        "neighbours_total": nb["k"],
        "neighbours_pneumonia": nb["pneumonia"],
        "neighbours_normal": nb["normal"],
        "strongest_zone": result["saliency_zone"],
        "zone_share": round(result["saliency_zone_share"], 2),
    }


def template(f: dict) -> str:
    side = "at or above" if f["pneumonia_probability"] >= f["operating_threshold"] else "below"
    agree = (f["neighbours_pneumonia"] > f["neighbours_normal"]) == (side == "at or above")
    lines = [
        f"The model gives a pneumonia probability of {f['pneumonia_probability']:.2f}. "
        f"This is {side} the operating threshold of {f['operating_threshold']:.2f}.",
        f"Of the {f['neighbours_total']} most similar training images, {f['neighbours_pneumonia']} have the label "
        f"PNEUMONIA and {f['neighbours_normal']} have the label NORMAL.",
        f"The strongest image evidence is in the {f['strongest_zone']} zone "
        f"({f['zone_share']:.2f} of the evidence in the six lung zones).",
    ]
    if not agree:
        lines.append("The similar training images do not agree with the model. Review this image with priority.")
    return " ".join(lines)


def check_text(text: str, f: dict) -> list[str]:
    """Problems that make an LLM text unusable."""
    problems = []
    if CERTAINTY.search(text):
        problems.append("uses a certainty word")
    allowed = {f"{v:.2f}" for v in f.values() if isinstance(v, float)} | {str(v) for v in f.values()}
    allowed |= {str(int(v)) for v in f.values() if isinstance(v, (int, float)) and float(v).is_integer()}
    for num in NUMBER.findall(text):
        if num not in allowed and f"{float(num):.2f}" not in allowed:
            problems.append(f"number {num} is not in the facts")
    return problems


def build_prompt(f: dict) -> str:
    return ("Rewrite the summary below for a clinician in at most four short sentences. Use only these facts. "
            "Do not state a diagnosis. Do not add numbers. Say that the result needs review by a clinician.\n"
            f"FACTS: {json.dumps(f)}\nTEMPLATE:\n{template(f)}")


def summarise(result: dict, client: LLMClient | None = None, image: np.ndarray | None = None,
              send_image: bool = False) -> dict:
    f = facts_from(result)
    text, source, problems = template(f), "template", []
    if client is not None:
        png = None
        if send_image and image is not None:
            from PIL import Image
            buf = io.BytesIO()
            Image.fromarray((np.clip(image, 0, 1) * 255).astype(np.uint8)).save(buf, format="PNG")
            png = buf.getvalue()
        try:
            candidate = client.generate(build_prompt(f), png).strip()
            problems = check_text(candidate, f)
            if not problems and candidate:
                text, source = candidate, client.name
        except Exception as exc:  # network or API error: keep the template
            problems = [f"LLM call failed: {type(exc).__name__}"]
    return {"text": f"{text} {BANNER}", "source": source, "rejected_because": problems, "facts": f}
