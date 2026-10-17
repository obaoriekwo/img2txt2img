"""
Model providers for the two halves of the pipeline.

Vision (image -> text), all through the OpenAI-compatible Chat Completions API:
    gemini   Google Gemini (default, free API key)        GEMINI_API_KEY
    openai   OpenAI, e.g. gpt-5.6-luna                     OPENAI_API_KEY
    local    LM Studio / Ollama / vLLM, e.g. Qwen3.6       LOCAL_BASE_URL (default http://localhost:1234/v1)

Image generation (text -> image):
    pollinations  Pollinations.ai (default, free, no key; optional POLLINATIONS_TOKEN)
    huggingface   Hugging Face Inference, FLUX.1-schnell   HF_TOKEN (free account)
    gemini        Google "Nano Banana" image models        GEMINI_API_KEY (needs billing)
    openai        OpenAI gpt-image-2                       OPENAI_API_KEY (paid)

Every generator returns the image as bytes, so the rest of the program does not care which one ran.
"""

import base64
import io
import os
import random
from urllib.parse import quote

import requests


class ProviderError(Exception):
    pass


# --------------------------------------------------------------------------- vision

VISION = {
    "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai/", "key_env": "GEMINI_API_KEY",
               "model": "gemini-3.5-flash-lite"},
    "openai": {"base_url": None, "key_env": "OPENAI_API_KEY", "model": "gpt-5.6-luna"},
    "local":  {"base_url": os.getenv("LOCAL_BASE_URL", "http://localhost:1234/v1"), "key_env": None,
               "model": "Qwen/Qwen3.6-35B-A3B"},
}


def describe_image(provider: str, model: str | None, image_bytes: bytes, mime: str,
                   system: str, instruction: str, max_tokens: int = 1200) -> tuple[str, str]:
    """Send the image to a vision model. Returns (description, model used)."""
    try:
        from openai import (APIConnectionError, APIStatusError, AuthenticationError, BadRequestError,
                            NotFoundError, OpenAI, RateLimitError)
    except ImportError:
        raise ProviderError("the 'openai' package is not installed (pip install -r requirements.txt)")
    cfg = VISION[provider]
    key = "not-needed"
    if cfg["key_env"]:
        key = os.getenv(cfg["key_env"])
        if not key:
            raise ProviderError(f"{cfg['key_env']} is not set. In PowerShell: $env:{cfg['key_env']}=\"your-key\"")
    model = model or cfg["model"]
    client = OpenAI(api_key=key, base_url=cfg["base_url"], timeout=120, max_retries=2)
    data_url = f"data:{mime};base64,{base64.b64encode(image_bytes).decode()}"
    params = {"model": model, "messages": [
        {"role": "system", "content": system},
        {"role": "user", "content": [{"type": "text", "text": instruction},
                                     {"type": "image_url", "image_url": {"url": data_url}}]}]}
    params["max_tokens" if cfg["base_url"] else "max_completion_tokens"] = max_tokens
    try:
        response = client.chat.completions.create(**params)
    except AuthenticationError:
        raise ProviderError("vision: authentication failed, check the API key")
    except NotFoundError:
        raise ProviderError(f"vision: model '{model}' not found (use --vision-model)")
    except RateLimitError as e:
        raise ProviderError(f"vision: rate limit or quota exceeded: {e}")
    except APIConnectionError:
        raise ProviderError("vision: could not connect to the provider (is the local server running?)")
    except (BadRequestError, APIStatusError) as e:
        raise ProviderError(f"vision: the provider rejected the request: {e}")
    text = (response.choices[0].message.content or "").strip()
    if not text:
        raise ProviderError("vision: the model returned an empty description")
    return text, model


# --------------------------------------------------------------------------- image generation

IMAGE_DEFAULT_MODELS = {
    "pollinations": "flux",
    "huggingface": "black-forest-labs/FLUX.1-schnell",
    "gemini": "gemini-3-pro-image",          # "Nano Banana Pro"; override with --image-model
    "openai": "gpt-image-2",
}
POLLINATIONS_URL = os.getenv("POLLINATIONS_URL", "https://image.pollinations.ai")
HF_URL = os.getenv("HF_URL", "https://router.huggingface.co/hf-inference/models")
GEMINI_URL = os.getenv("GEMINI_URL", "https://generativelanguage.googleapis.com/v1beta")
MAX_URL_PROMPT = 1800   # characters; the Pollinations prompt goes inside the URL


def generate_image(provider: str, model: str | None, prompt: str, width: int, height: int,
                   seed: int | None) -> tuple[bytes, str]:
    """Generate an image from text. Returns (image bytes, model used)."""
    model = model or IMAGE_DEFAULT_MODELS[provider]
    try:
        if provider == "pollinations":
            return _pollinations(model, prompt, width, height, seed), model
        if provider == "huggingface":
            return _huggingface(model, prompt, width, height, seed), model
        if provider == "gemini":
            return _gemini(model, prompt, width, height), model
        if provider == "openai":
            return _openai(model, prompt, width, height), model
    except requests.Timeout:
        raise ProviderError(f"{provider}: the image request timed out")
    except requests.ConnectionError:
        raise ProviderError(f"{provider}: could not connect")
    raise ProviderError(f"unknown image provider '{provider}'")


def _check(r: requests.Response, provider: str) -> None:
    if r.status_code in (401, 403):
        raise ProviderError(f"{provider}: access denied (HTTP {r.status_code}) - check the API key or token")
    if r.status_code == 402:
        raise ProviderError(f"{provider}: payment or billing required (HTTP 402)")
    if r.status_code == 429:
        raise ProviderError(f"{provider}: rate limit or quota exceeded (HTTP 429) - wait a moment and try again")
    if r.status_code >= 400:
        raise ProviderError(f"{provider}: HTTP {r.status_code}: {r.text[:300]}")


def _ensure_image(data: bytes, provider: str) -> bytes:
    from PIL import Image
    try:
        Image.open(io.BytesIO(data)).verify()
    except Exception:
        raise ProviderError(f"{provider}: the response was not an image: {data[:200]!r}")
    return data


def _pollinations(model: str, prompt: str, width: int, height: int, seed: int | None) -> bytes:
    if len(prompt) > MAX_URL_PROMPT:
        prompt = prompt[:MAX_URL_PROMPT].rsplit(" ", 1)[0]
    params = {"width": width, "height": height, "model": model, "nologo": "true",
              "seed": seed if seed is not None else random.randint(1, 10**9)}
    headers = {}
    if os.getenv("POLLINATIONS_TOKEN"):
        headers["Authorization"] = f"Bearer {os.getenv('POLLINATIONS_TOKEN')}"
    r = requests.get(f"{POLLINATIONS_URL}/prompt/{quote(prompt, safe='')}", params=params,
                     headers=headers, timeout=180)
    _check(r, "pollinations")
    return _ensure_image(r.content, "pollinations")


def _huggingface(model: str, prompt: str, width: int, height: int, seed: int | None) -> bytes:
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ProviderError("HF_TOKEN is not set (create a free token at huggingface.co/settings/tokens)")
    payload = {"inputs": prompt, "parameters": {"width": width, "height": height}}
    if seed is not None:
        payload["parameters"]["seed"] = seed
    r = requests.post(f"{HF_URL}/{model}", json=payload, timeout=180,
                      headers={"Authorization": f"Bearer {token}", "Accept": "image/png"})
    _check(r, "huggingface")
    return _ensure_image(r.content, "huggingface")


def _gemini(model: str, prompt: str, width: int, height: int) -> bytes:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise ProviderError("GEMINI_API_KEY is not set")
    ratio = _closest_ratio(width, height)
    body = {"contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"responseModalities": ["IMAGE"], "imageConfig": {"aspectRatio": ratio}}}
    r = requests.post(f"{GEMINI_URL}/models/{model}:generateContent", json=body, timeout=180,
                      headers={"x-goog-api-key": key})
    if r.status_code == 429:
        raise ProviderError("gemini: no image quota. Gemini image models are not on the free tier; "
                            "link billing to the project, or use --image-provider pollinations")
    _check(r, "gemini")
    for part in r.json().get("candidates", [{}])[0].get("content", {}).get("parts", []):
        inline = part.get("inlineData") or part.get("inline_data")
        if inline and inline.get("data"):
            return _ensure_image(base64.b64decode(inline["data"]), "gemini")
    raise ProviderError("gemini: the response contained no image (it may have been blocked by a safety filter)")


def _openai(model: str, prompt: str, width: int, height: int) -> bytes:
    try:
        from openai import OpenAI, OpenAIError
    except ImportError:
        raise ProviderError("the 'openai' package is not installed")
    if not os.getenv("OPENAI_API_KEY"):
        raise ProviderError("OPENAI_API_KEY is not set")
    size = "1024x1024" if width == height else ("1536x1024" if width > height else "1024x1536")
    try:
        result = OpenAI(timeout=180).images.generate(model=model, prompt=prompt, size=size, n=1)
    except OpenAIError as e:
        raise ProviderError(f"openai: {e}")
    item = result.data[0]
    if getattr(item, "b64_json", None):
        return _ensure_image(base64.b64decode(item.b64_json), "openai")
    if getattr(item, "url", None):
        r = requests.get(item.url, timeout=120)
        _check(r, "openai")
        return _ensure_image(r.content, "openai")
    raise ProviderError("openai: no image in the response")


def _closest_ratio(width: int, height: int) -> str:
    options = {"1:1": 1, "3:4": 0.75, "4:3": 4 / 3, "9:16": 9 / 16, "16:9": 16 / 9, "2:3": 2 / 3, "3:2": 1.5}
    target = width / height
    return min(options, key=lambda k: abs(options[k] - target))
