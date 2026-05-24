"""
LLM Adapter — unified interface for LLM calls.

Call chain: Codex CLI → OpenAI API → Mock
Supports model selection (gpt-5.4-mini for cost efficiency) and parallel execution.
"""

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

logger = logging.getLogger("vpatent.llm")

# ─── nvm path injection (tomo0 server) ──────────────────
for _node_dir in [
    os.path.expanduser("~/.nvm/versions/node/v24.14.0/bin"),
    os.path.expanduser("~/.nvm/versions/node/v20.9.0/bin"),
]:
    if os.path.isdir(_node_dir) and _node_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _node_dir + ":" + os.environ["PATH"]
        break


# ─── JSON extraction ─────────────────────────────────────

def extract_json(text: str) -> Optional[dict]:
    """Extract JSON from LLM response that may contain markdown fences."""
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for pattern in [r"```json\s*(.*?)```", r"```\s*(.*?)```", r"(\{.*\})"]:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                continue
    return None


# ─── Codex CLI backend ───────────────────────────────────

def is_codex_available() -> bool:
    return shutil.which("codex") is not None


def _call_codex(prompt: str, model: str = "gpt-5.4-mini",
                timeout: int = 120) -> Optional[str]:
    """Call Codex CLI exec mode with specified model."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
        output_file = tmp.name
    try:
        cmd = [
            "codex", "exec",
            "-m", model,
            "--full-auto",
            "--skip-git-repo-check",
            "--ephemeral",
            "-o", output_file,
            prompt,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if os.path.exists(output_file):
            with open(output_file) as f:
                content = f.read().strip()
            if content:
                return content
        return result.stdout.strip() or result.stderr.strip() or None
    except subprocess.TimeoutExpired:
        logger.warning("Codex CLI timed out after %ds (model=%s)", timeout, model)
        return None
    except Exception as e:
        logger.warning("Codex CLI error: %s", e)
        return None
    finally:
        if os.path.exists(output_file):
            try:
                os.unlink(output_file)
            except OSError:
                pass


# ─── OpenAI API backend ──────────────────────────────────

def _call_openai(system_prompt: str, user_prompt: str,
                 model: str = "gpt-4.1-mini",
                 json_mode: bool = True) -> Optional[str]:
    """Call OpenAI API as fallback."""
    try:
        from openai import OpenAI
    except ImportError:
        logger.warning("openai package not installed")
        return None

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        for env_path in ["/data1/tomo-agent/.env", "/data1/paper-kb/.env"]:
            if os.path.exists(env_path):
                with open(env_path) as f:
                    for line in f:
                        if "OPENAI_API_KEY=" in line:
                            api_key = line.strip().split("=", 1)[1]
                            break
            if api_key:
                break
    if not api_key:
        logger.warning("No OpenAI API key found")
        return None

    client = OpenAI(api_key=api_key)
    try:
        kwargs = dict(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.4,
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content
    except Exception as e:
        logger.warning("OpenAI API error: %s", e)
        return None




# ─── Anthropic direct API backend ─────────────────────────

def _normalize_anthropic_model(model: str) -> str:
    """Convert OpenRouter-style model id to Anthropic SDK id.

    anthropic/claude-sonnet-4.5 -> claude-sonnet-4-5
    claude-sonnet-4-5           -> claude-sonnet-4-5 (passthrough)
    """
    m = model.split("/", 1)[1] if "/" in model else model
    return m.replace(".", "-")


def _call_anthropic_direct(system_prompt: str, user_prompt: str,
                           model: str, json_mode: bool = True,
                           max_retries: int = 5) -> Optional[str]:
    """Call Anthropic API directly with prompt caching on the system block.

    Caching: system prompt is marked ephemeral. Within a single examiner's
    8-step pipeline the same persona system prompt is reused, so steps 2-8
    hit the cache (cost ~10% of base input price for that portion).
    """
    try:
        import anthropic
    except ImportError:
        logger.warning("anthropic package not installed")
        return None

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        logger.warning("No ANTHROPIC_API_KEY found")
        return None

    if json_mode:
        user_prompt = user_prompt + "\n\nRespond with valid JSON only."

    sdk_model = _normalize_anthropic_model(model)
    client = anthropic.Anthropic(api_key=api_key)
    system_blocks = [{
        "type": "text",
        "text": system_prompt,
        "cache_control": {"type": "ephemeral"},
    }]

    for attempt in range(max_retries):
        try:
            resp = client.messages.create(
                model=sdk_model,
                max_tokens=16000,
                system=system_blocks,
                messages=[{"role": "user", "content": user_prompt}],
            )
            text = next((b.text for b in resp.content if b.type == "text"), None)
            if attempt == 0 and resp.usage.cache_read_input_tokens:
                logger.debug("[Anthropic/%s] cache read=%d write=%d input=%d output=%d",
                             sdk_model, resp.usage.cache_read_input_tokens,
                             resp.usage.cache_creation_input_tokens,
                             resp.usage.input_tokens, resp.usage.output_tokens)
            return text
        except anthropic.RateLimitError:
            wait = 10 * (attempt + 1)
            logger.info("[Anthropic] Rate limited, waiting %ds (attempt %d/%d)",
                        wait, attempt + 1, max_retries)
            time.sleep(wait)
        except anthropic.APIStatusError as e:
            if e.status_code >= 500:
                logger.info("[Anthropic] Server error %d, retrying", e.status_code)
                time.sleep(5 * (attempt + 1))
                continue
            logger.warning("Anthropic API error: %s", e)
            return None
        except Exception as e:
            logger.warning("Anthropic API error: %s", e)
            return None

    logger.error("Anthropic: max retries exceeded for model=%s", sdk_model)
    return None


# ─── OpenAI direct API backend ────────────────────────────

def _normalize_openai_model(model: str) -> str:
    """Convert OpenRouter-style model id to OpenAI SDK id.

    openai/gpt-4.1 -> gpt-4.1
    gpt-4.1        -> gpt-4.1 (passthrough)
    """
    return model.split("/", 1)[1] if "/" in model else model


def _call_openai_direct(system_prompt: str, user_prompt: str,
                        model: str, json_mode: bool = True,
                        max_retries: int = 5) -> Optional[str]:
    """Call OpenAI API directly. Caching is automatic for prompts >= 1024 tokens.

    OpenAI's prompt caching activates automatically when the prompt prefix
    exceeds 1024 tokens — no special headers or content-block annotations.
    Cached input tokens cost 25% of base price (75% discount).
    """
    try:
        from openai import OpenAI
    except ImportError:
        logger.warning("openai package not installed")
        return None

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        logger.warning("No OPENAI_API_KEY found")
        return None

    sdk_model = _normalize_openai_model(model)
    client = OpenAI(api_key=api_key)

    for attempt in range(max_retries):
        try:
            kwargs = dict(
                model=sdk_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.4,
                max_tokens=16000,
            )
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            resp = client.chat.completions.create(**kwargs)
            cached = getattr(getattr(resp.usage, "prompt_tokens_details", None),
                             "cached_tokens", 0) or 0
            if attempt == 0 and cached:
                logger.debug("[OpenAI/%s] cache hit=%d input=%d output=%d",
                             sdk_model, cached, resp.usage.prompt_tokens,
                             resp.usage.completion_tokens)
            return resp.choices[0].message.content
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "rate" in err_str.lower():
                wait = 10 * (attempt + 1)
                logger.info("[OpenAI] Rate limited, waiting %ds (attempt %d/%d)",
                            wait, attempt + 1, max_retries)
                time.sleep(wait)
                continue
            if any(s in err_str for s in ("500", "502", "503", "504")):
                logger.info("[OpenAI] Server error, retrying")
                time.sleep(5 * (attempt + 1))
                continue
            logger.warning("OpenAI API error: %s", e)
            return None

    logger.error("OpenAI: max retries exceeded for model=%s", sdk_model)
    return None


# ─── DeepSeek API backend ─────────────────────────────────

def _call_deepseek(system_prompt: str, user_prompt: str,
                   model: str = "qwen/qwen3-next-80b-a3b-instruct:free",
                   json_mode: bool = True,
                   max_retries: int = 5) -> Optional[str]:
    """Call LLM via OpenRouter with automatic retry/backoff for rate limits.

    Claude / GPT models MUST go through their direct APIs (cheaper + caching).
    Routing here forbids the OpenRouter path for those models even if an
    OpenRouter key is present — fail fast if the direct API key is missing.
    """
    if "claude" in model.lower() and not os.environ.get("LLM_BASE_URL"):
        if not os.environ.get("ANTHROPIC_API_KEY"):
            logger.error("Claude model %s requires ANTHROPIC_API_KEY (not set). "
                         "Sonnet/Claude no longer routes through OpenRouter.", model)
            return None
        return _call_anthropic_direct(system_prompt, user_prompt,
                                      model=model, json_mode=json_mode,
                                      max_retries=max_retries)

    is_openai_model = (model.startswith("openai/")
                       or model.startswith("gpt-")
                       or model.startswith("o1-")
                       or model.startswith("o3-")
                       or model.startswith("o4-"))
    if is_openai_model and not os.environ.get("LLM_BASE_URL"):
        if not os.environ.get("OPENAI_API_KEY"):
            logger.error("OpenAI model %s requires OPENAI_API_KEY (not set). "
                         "GPT no longer routes through OpenRouter.", model)
            return None
        return _call_openai_direct(system_prompt, user_prompt,
                                   model=model, json_mode=json_mode,
                                   max_retries=max_retries)

    try:
        from openai import OpenAI
    except ImportError:
        logger.warning("openai package not installed")
        return None

    api_base = os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1")
    is_openrouter = "openrouter" in api_base
    api_key = os.environ.get("OPENROUTER_API_KEY", "EMPTY")
    if is_openrouter and api_key == "EMPTY":
        logger.warning("No OPENROUTER_API_KEY found")
        return None

    client = OpenAI(api_key=api_key, base_url=api_base)

    for attempt in range(max_retries):
        try:
            kwargs = dict(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.4,
            )
            if json_mode and is_openrouter:
                kwargs["response_format"] = {"type": "json_object"}
            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "rate" in err_str.lower():
                wait = 10 * (attempt + 1)  # 10s, 20s, 30s...
                logger.info("[OpenRouter] Rate limited, waiting %ds (attempt %d/%d)",
                           wait, attempt + 1, max_retries)
                time.sleep(wait)
                continue
            logger.warning("OpenRouter API error: %s", e)
            return None

    logger.error("OpenRouter: max retries exceeded for model=%s", model)
    return None

# ─── Mock backend ─────────────────────────────────────────

_MOCK_RESPONSES: dict[str, dict] = {}


def register_mock(role_name: str, response: dict):
    _MOCK_RESPONSES[role_name] = response


def _call_mock(role_name: str) -> Optional[dict]:
    return _MOCK_RESPONSES.get(role_name)


# ─── Unified Interface ────────────────────────────────────

class LLMAdapter:
    """
    Unified LLM call interface.

    Modes:
      - "auto": Codex → OpenAI → error
      - "codex": Codex only
      - "openai": OpenAI only
      - "deepseek": DeepSeek API (deepseek-chat = V3)
      - "mock": Mock responses only
    """

    def __init__(self, mode: str = "auto", model: str = "gpt-5.4-mini",
                 timeout: int = 120, retries: int = 2,
                 max_parallel: int = 1):
        self.mode = mode
        self.model = model
        self.timeout = timeout
        self.retries = retries
        self.max_parallel = max_parallel
        self._call_count = 0
        self._total_elapsed = 0.0
        logger.info("LLMAdapter: mode=%s, model=%s, parallel=%d", mode, model, max_parallel)

    @property
    def stats(self) -> dict:
        return {"calls": self._call_count, "total_elapsed": self._total_elapsed}

    def call_json(self, system_prompt: str, user_prompt: str,
                  role_name: str = "") -> Optional[dict]:
        """Call LLM and return parsed JSON dict."""
        start = time.time()
        self._call_count += 1

        if self.mode == "mock":
            mock = _call_mock(role_name)
            if mock:
                self._total_elapsed += time.time() - start
                return mock
            logger.warning("No mock registered for role=%s", role_name)
            return None

        json_instruction = ("\n\nIMPORTANT: Respond ONLY with valid JSON. "
                            "No markdown fences, no explanation. Just the raw JSON object.")

        # Codex path
        if self.mode in ("auto", "codex"):
            combined = f"{system_prompt}\n\n---\n\n{user_prompt}{json_instruction}"
            for attempt in range(self.retries):
                raw = _call_codex(combined, model=self.model, timeout=self.timeout)
                if raw:
                    result = extract_json(raw)
                    if result:
                        self._total_elapsed += time.time() - start
                        return result
                    logger.info("[Codex/%s] JSON parse fail (attempt %d)", self.model, attempt + 1)
                else:
                    logger.info("[Codex/%s] Empty response (attempt %d)", self.model, attempt + 1)
                time.sleep(1)
            if self.mode == "codex":
                self._total_elapsed += time.time() - start
                return None

        # DeepSeek path
        if self.mode in ("auto", "deepseek"):
            ds_model = self.model if self.model != "gpt-5.4-mini" else "qwen/qwen3-next-80b-a3b-instruct:free"
            for attempt in range(self.retries):
                raw = _call_deepseek(system_prompt, user_prompt + json_instruction,
                                     model=ds_model, json_mode=True)
                if raw:
                    result = extract_json(raw)
                    if result:
                        self._total_elapsed += time.time() - start
                        return result
                    logger.info("[DeepSeek/%s] JSON parse fail (attempt %d) raw=%r", ds_model, attempt + 1, (raw or "")[:500])
                time.sleep(0.5)
            if self.mode == "deepseek":
                self._total_elapsed += time.time() - start
                return None

        # OpenAI path
        if self.mode in ("auto", "openai"):
            openai_model = os.environ.get("VPATENT_OPENAI_MODEL", "gpt-4.1-mini")
            for attempt in range(self.retries):
                raw = _call_openai(system_prompt, user_prompt + json_instruction,
                                   model=openai_model, json_mode=True)
                if raw:
                    result = extract_json(raw)
                    if result:
                        self._total_elapsed += time.time() - start
                        return result
                    logger.info("[OpenAI/%s] JSON parse fail (attempt %d)", openai_model, attempt + 1)
                time.sleep(1)

        self._total_elapsed += time.time() - start
        logger.error("All LLM backends failed for role=%s", role_name)
        return None

    def call_text(self, system_prompt: str, user_prompt: str,
                  role_name: str = "") -> Optional[str]:
        """Call LLM and return raw text."""
        start = time.time()
        self._call_count += 1

        if self.mode == "mock":
            mock = _call_mock(role_name)
            self._total_elapsed += time.time() - start
            if mock:
                return mock.get("text", json.dumps(mock, ensure_ascii=False))
            return None

        if self.mode in ("auto", "codex"):
            combined = f"{system_prompt}\n\n---\n\n{user_prompt}"
            raw = _call_codex(combined, model=self.model, timeout=self.timeout)
            if raw:
                self._total_elapsed += time.time() - start
                return raw
            if self.mode == "codex":
                self._total_elapsed += time.time() - start
                return None

        if self.mode in ("auto", "deepseek"):
            raw = _call_deepseek(system_prompt, user_prompt,
                                 model=self.model if self.model != "gpt-5.4-mini" else "qwen/qwen3-next-80b-a3b-instruct:free", json_mode=False)
            if raw:
                self._total_elapsed += time.time() - start
                return raw
            if self.mode == "deepseek":
                self._total_elapsed += time.time() - start
                return None

        if self.mode in ("auto", "openai"):
            openai_model = os.environ.get("VPATENT_OPENAI_MODEL", "gpt-4.1-mini")
            raw = _call_openai(system_prompt, user_prompt,
                               model=openai_model, json_mode=False)
            if raw:
                self._total_elapsed += time.time() - start
                return raw

        self._total_elapsed += time.time() - start
        return None
