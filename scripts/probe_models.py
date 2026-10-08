"""One-off probe: which chat model IDs actually work with our API keys.

Reads GEMINI_API_KEY / GROQ_API_KEY from .env, lists Groq's active models,
then sends a 1-token completion to every candidate. Prints OK/FAIL per model.
Delete this file after the model list is fixed.
"""
from __future__ import annotations

import re
import sys
import urllib.request
import json


def read_env(path: str = ".env") -> dict[str, str]:
    env: dict[str, str] = {}
    for line in open(path, encoding="utf-8-sig"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            env[key.strip()] = value.strip()
    return env


def list_groq_models(groq_key: str) -> list[str]:
    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/models",
        headers={"Authorization": f"Bearer {groq_key}"},
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = json.load(resp)
    return sorted(m["id"] for m in data.get("data", []))


def probe(model: str, api_key: str) -> str:
    import litellm

    try:
        resp = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": "Reply with the word OK."}],
            api_key=api_key,
            max_tokens=8,
            timeout=45,
        )
        text = (resp.choices[0].message.content or "").strip()[:40]
        return f"OK -> {text!r}"
    except Exception as exc:  # noqa: BLE001 - probe reports, never raises
        name = type(exc).__name__
        msg = str(exc).replace("\n", " ")[:150]
        return f"FAIL [{name}] {msg}"


def main() -> int:
    env = read_env()
    gemini_key = env.get("GEMINI_API_KEY", "")
    groq_key = env.get("GROQ_API_KEY", "")
    print(f"GEMINI key: {gemini_key[:6]}...len={len(gemini_key)}")
    print(f"GROQ key:   {groq_key[:6]}...len={len(groq_key)}")

    if groq_key:
        try:
            active = list_groq_models(groq_key)
            print("\n-- Groq active models --")
            for mid in active:
                print("  ", mid)
        except Exception as exc:  # noqa: BLE001
            print(f"\nGroq /models list failed: {type(exc).__name__}: {exc}")
            active = []
    else:
        active = []

    gemini_candidates = [
        "gemini/gemini-2.5-flash",
        "gemini/gemini-2.5-flash-lite",
        "gemini/gemini-3.5-flash-lite",
        "gemini/gemini-3.1-flash-lite",
        "gemini/gemini-3.6-flash",
        "gemini/gemini-flash-latest",
        "gemini/gemini-2.0-flash",
    ]
    groq_candidates = [f"groq/{mid}" for mid in active if any(
        tag in mid for tag in ("gpt-oss", "llama", "qwen", "moonshot", "kimi", "mistral", "deepseek")
    )] or [
        "groq/openai/gpt-oss-120b",
        "groq/openai/gpt-oss-20b",
        "groq/llama-3.3-70b-versatile",
        "groq/llama-3.1-8b-instant",
    ]

    print("\n-- Gemini probes --")
    for model in gemini_candidates:
        print(f"{model}: {probe(model, gemini_key)}")
    print("\n-- Groq probes --")
    for model in groq_candidates[:10]:
        print(f"{model}: {probe(model, groq_key)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
