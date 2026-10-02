"""LLM backends for the Text2SQL pipeline.

Select with LLM_BACKEND:
  vertex  -> Gemini on Vertex AI (Cloud Run default; no torch needed)
  local   -> Qwen via Hugging Face transformers on CPU (offline/local dev)

All agents call ``generate(messages, max_new_tokens)`` with OpenAI-style
messages: [{"role": "system"|"user"|"assistant", "content": str}, ...].
"""

import logging
import os

logger = logging.getLogger("Text2SQL_RAG")

LLM_BACKEND = os.getenv("LLM_BACKEND", "local").lower()

# Vertex AI settings
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "als-user-office-software")
GOOGLE_CLOUD_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "global")
# Optional: MINIMAL / LOW / MEDIUM / HIGH. Unset = model default.
GEMINI_THINKING_LEVEL = os.getenv("GEMINI_THINKING_LEVEL")
# Headroom for thinking tokens, which count toward the output limit
GEMINI_MIN_OUTPUT_TOKENS = int(os.getenv("GEMINI_MIN_OUTPUT_TOKENS", "2048"))

# Local model settings
MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2.5-Coder-3B-Instruct")

_backend = None


class VertexBackend:
    """Gemini on Vertex AI via the google-genai SDK."""

    def __init__(self) -> None:
        from google import genai
        from google.genai import types

        self._types = types
        # Uses Application Default Credentials: the Cloud Run service account
        # in production, `gcloud auth application-default login` locally.
        self._client = genai.Client(
            vertexai=True, project=GOOGLE_CLOUD_PROJECT, location=GOOGLE_CLOUD_LOCATION
        )
        logger.info(
            f"[LLM] Vertex AI backend: {GEMINI_MODEL} "
            f"({GOOGLE_CLOUD_PROJECT}/{GOOGLE_CLOUD_LOCATION})"
        )

    def generate(self, messages: list[dict], max_new_tokens: int) -> str:
        types = self._types
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        contents = [
            types.Content(
                role="model" if m["role"] == "assistant" else "user",
                parts=[types.Part(text=m["content"])],
            )
            for m in messages
            if m["role"] != "system"
        ]
        config = types.GenerateContentConfig(
            system_instruction=system or None,
            temperature=0.0,
            max_output_tokens=max(max_new_tokens, GEMINI_MIN_OUTPUT_TOKENS),
        )
        if GEMINI_THINKING_LEVEL:
            config.thinking_config = types.ThinkingConfig(
                thinking_level=GEMINI_THINKING_LEVEL.upper()
            )
        response = self._client.models.generate_content(
            model=GEMINI_MODEL, contents=contents, config=config
        )
        usage = response.usage_metadata
        if usage:
            logger.info(
                f"[LLM] tokens in={usage.prompt_token_count} "
                f"out={usage.candidates_token_count} "
                f"thinking={getattr(usage, 'thoughts_token_count', None)}"
            )
        return (response.text or "").strip()


class LocalBackend:
    """Qwen on CPU via transformers. Imports torch only when selected."""

    def __init__(self) -> None:
        import warnings

        import torch
        from transformers import logging as hf_logging
        from transformers import pipeline

        hf_logging.set_verbosity_error()
        warnings.filterwarnings("ignore")
        # os.cpu_count() can report host cores in containers; allow pinning
        torch.set_num_threads(int(os.getenv("TORCH_THREADS", os.cpu_count() or 1)))
        logger.info(f"[LLM] Local backend: loading {MODEL_ID} on CPU...")
        self._pipe = pipeline(
            "text-generation",
            model=MODEL_ID,
            device=-1,
            dtype=torch.bfloat16,
            model_kwargs={"low_cpu_mem_usage": True, "use_cache": True},
        )

    def generate(self, messages: list[dict], max_new_tokens: int) -> str:
        prompt = self._pipe.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        out = self._pipe(
            prompt, max_new_tokens=max_new_tokens, max_length=None, do_sample=False
        )[0]["generated_text"]
        return out[len(prompt):].strip()


def init() -> None:
    """Create the configured backend once, at server startup."""
    global _backend
    if LLM_BACKEND == "vertex":
        _backend = VertexBackend()
    elif LLM_BACKEND == "local":
        _backend = LocalBackend()
    else:
        raise ValueError(f"Unknown LLM_BACKEND={LLM_BACKEND!r}; use 'vertex' or 'local'")


def generate(messages: list[dict], max_new_tokens: int) -> str:
    if _backend is None:
        raise RuntimeError("llm.init() has not been called")
    return _backend.generate(messages, max_new_tokens)
