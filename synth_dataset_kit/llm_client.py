"""Provider-aware LLM client with first-class Ollama and Anthropic support."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Semaphore
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from openai import OpenAI

from synth_dataset_kit.config import LLMConfig, LLMProvider

logger = logging.getLogger(__name__)


class LLMClient:
    """Unified client for Ollama, OpenAI, Anthropic, vLLM, and compatible APIs."""

    def __init__(self, config: LLMConfig):
        self.config = config
        api_key = self._resolve_api_key(config)
        self._client = OpenAI(
            base_url=config.api_base,
            api_key=api_key,
            timeout=config.timeout,
            max_retries=config.max_retries,
        )
        self._recommendation_cache_path = Path(".sdk_cache/recommendations/ollama_models.json")
        self._semaphore = Semaphore(max(1, config.concurrent_requests))
        self._max_workers = max(1, config.concurrent_requests)

    def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict | None = None,
    ) -> str:
        """Send a chat completion request and return the text response."""
        if self.config.provider == LLMProvider.ANTHROPIC:
            return self._complete_anthropic(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature or self.config.temperature,
            "top_p": self.config.top_p,
        }
        token_limit = max_tokens or self.config.max_tokens
        if self.config.model.startswith("gpt-5"):
            kwargs["max_completion_tokens"] = token_limit
        else:
            kwargs["max_tokens"] = token_limit
        if response_format and self.config.provider != LLMProvider.OLLAMA:
            kwargs["response_format"] = response_format

        for attempt in range(self.config.max_retries):
            try:
                response = self._client.chat.completions.create(**kwargs)
                return response.choices[0].message.content or ""
            except Exception as e:
                logger.warning(f"LLM call failed (attempt {attempt + 1}): {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(2**attempt)
                else:
                    raise

        return ""

    def _complete_anthropic(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Call Anthropic's native messages API without requiring an extra SDK."""
        api_key = self._resolve_api_key(self.config, allow_placeholder=False)
        if not api_key:
            raise ValueError("Anthropic provider requires `api_key` in config.")

        system_parts: list[str] = []
        anthropic_messages: list[dict[str, str]] = []
        for message in messages:
            role = str(message.get("role", "user")).strip().lower()
            content = str(message.get("content", ""))
            if role == "system":
                system_parts.append(content)
                continue
            if role not in {"user", "assistant"}:
                role = "user"
            anthropic_messages.append({"role": role, "content": content})

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": anthropic_messages or [{"role": "user", "content": ""}],
            "temperature": temperature if temperature is not None else self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
        }
        if system_parts:
            payload["system"] = "\n\n".join(part for part in system_parts if part.strip())

        request = Request(
            f"{self.config.api_base.rstrip('/')}/messages",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        with urlopen(request, timeout=self.config.timeout) as response:
            body = json.loads(response.read().decode("utf-8"))

        content = body.get("content", [])
        text_parts = [
            str(item.get("text", ""))
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        return "\n".join(part for part in text_parts if part).strip()

    def complete_json(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
    ) -> dict | list:
        """Send a request expecting JSON output, parse and return."""
        raw = self.complete(
            messages,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        parsed = self._parse_json_response(raw)
        if parsed is None:
            logger.error(f"Failed to parse JSON from LLM response: {raw[:200]}")
            return {}
        return parsed

    def batch_complete(
        self,
        message_batches: list[list[dict[str, str]]],
        temperature: float | None = None,
    ) -> list[str]:
        """Process multiple requests concurrently using a thread pool.

        Concurrency is governed by ``config.concurrent_requests`` (default 4).
        A semaphore ensures no more than that many in-flight requests hit the
        LLM provider at any time, acting as a simple rate limiter.
        """
        if not message_batches:
            return []

        # For a single request, skip the pool overhead.
        if len(message_batches) == 1:
            try:
                return [self.complete(message_batches[0], temperature=temperature)]
            except Exception as e:
                logger.error(f"Batch item failed: {e}")
                return [""]

        results: list[str | None] = [None] * len(message_batches)

        def _worker(index: int, messages: list[dict[str, str]]) -> tuple[int, str]:
            self._semaphore.acquire()
            try:
                return index, self.complete(messages, temperature=temperature)
            except Exception as e:
                logger.error(f"Batch item {index} failed: {e}")
                return index, ""
            finally:
                self._semaphore.release()

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = [pool.submit(_worker, i, msgs) for i, msgs in enumerate(message_batches)]
            for future in as_completed(futures):
                idx, text = future.result()
                results[idx] = text

        return [r or "" for r in results]

    def batch_complete_json(
        self,
        message_batches: list[list[dict[str, str]]],
        temperature: float | None = None,
    ) -> list[dict | list]:
        """Process multiple JSON requests concurrently.

        Same concurrency semantics as :meth:`batch_complete`, but each
        response is parsed as JSON via :meth:`complete_json`.
        """
        if not message_batches:
            return []

        if len(message_batches) == 1:
            return [self.complete_json(message_batches[0], temperature=temperature)]

        results: list[dict | list | None] = [None] * len(message_batches)

        def _worker(index: int, messages: list[dict[str, str]]) -> tuple[int, dict | list]:
            self._semaphore.acquire()
            try:
                return index, self.complete_json(messages, temperature=temperature)
            except Exception as e:
                logger.error(f"Batch JSON item {index} failed: {e}")
                return index, {}
            finally:
                self._semaphore.release()

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = [pool.submit(_worker, i, msgs) for i, msgs in enumerate(message_batches)]
            for future in as_completed(futures):
                idx, parsed = future.result()
                results[idx] = parsed

        return [r if r is not None else {} for r in results]

    def health_check(self) -> bool:
        """Verify the LLM endpoint is reachable."""
        try:
            response = self.complete(
                [{"role": "user", "content": "Say 'ok'"}],
                max_tokens=5,
                temperature=0,
            )
            return len(response) > 0
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def list_models(self) -> list[dict[str, Any]]:
        """List available models for supported providers when possible."""
        if self.config.provider == LLMProvider.OLLAMA:
            return self._list_ollama_models()

        try:
            models = self._client.models.list()
            return [{"name": model.id} for model in models.data]
        except Exception as e:
            logger.warning(f"Model listing failed: {e}")
            return []

    def recommend_model(self, domain: str = "") -> str | None:
        """Recommend the best available model for the current provider and domain."""
        models = self.list_models()
        if not models:
            return None

        names = [str(model.get("name", "")) for model in models if model.get("name")]
        if self.config.provider == LLMProvider.OLLAMA:
            benchmark_recommendation = self._load_benchmark_recommendation(domain)
            if benchmark_recommendation and benchmark_recommendation in names:
                return benchmark_recommendation

            domain_lower = domain.lower()
            preferred_groups = []
            is_code_domain = any(
                token in domain_lower for token in ["code", "python", "sql", "program"]
            )
            if is_code_domain:
                preferred_groups.extend(
                    ["qwen2.5-coder", "deepseek-coder", "codellama", "codegemma"]
                )
            if any(
                token in domain_lower
                for token in ["support", "customer", "chat", "multilingual", "russian"]
            ):
                preferred_groups.extend(["qwen", "llama3", "mistral", "gemma"])
            if is_code_domain:
                preferred_groups.extend(["qwen", "llama3", "mistral", "gemma"])
            else:
                preferred_groups.extend(["llama3", "qwen", "mistral", "gemma"])

            for preferred in preferred_groups:
                for name in names:
                    if (
                        not is_code_domain
                        and "coder" in name.lower()
                        and preferred in {"qwen", "llama3", "mistral", "gemma"}
                    ):
                        continue
                    if preferred in name.lower():
                        return name
        return names[0]

    def save_benchmark_recommendation(
        self,
        domain: str,
        recommendation: dict[str, Any],
    ) -> str | None:
        """Persist the best Ollama model for a given domain for future zero-shot runs."""
        if self.config.provider != LLMProvider.OLLAMA:
            return None
        model_name = str(recommendation.get("model", "")).strip()
        if not domain.strip() or not model_name:
            return None

        payload = self._read_recommendation_cache()
        key = self._normalize_domain_key(domain)
        payload[key] = {
            "domain": domain.strip(),
            "model": model_name,
            "benchmark_score": recommendation.get("benchmark_score"),
            "avg_quality_score": recommendation.get("avg_quality_score"),
            "pass_rate": recommendation.get("pass_rate"),
            "examples_per_second": recommendation.get("examples_per_second"),
            "updated_at": int(time.time()),
        }
        cache_path = self._recommendation_cache_file()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return str(cache_path)

    def _load_benchmark_recommendation(self, domain: str) -> str | None:
        """Return the best prior benchmark result for a domain when available."""
        if not domain.strip():
            return None
        payload = self._read_recommendation_cache()
        if not payload:
            return None

        exact = payload.get(self._normalize_domain_key(domain))
        if isinstance(exact, dict):
            return str(exact.get("model", "")).strip() or None

        target_tokens = set(self._normalize_domain_key(domain).split("_"))
        best_model = None
        best_overlap = 0
        for key, item in payload.items():
            if not isinstance(item, dict):
                continue
            candidate_tokens = set(str(key).split("_"))
            overlap = len(target_tokens & candidate_tokens)
            if overlap > best_overlap:
                candidate_model = str(item.get("model", "")).strip()
                if candidate_model:
                    best_overlap = overlap
                    best_model = candidate_model
        return best_model

    def _read_recommendation_cache(self) -> dict[str, Any]:
        cache_path = self._recommendation_cache_file()
        if not cache_path.exists():
            return {}
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}

    def _normalize_domain_key(self, domain: str) -> str:
        cleaned = "".join(char.lower() if char.isalnum() else "_" for char in domain)
        tokens = [token for token in cleaned.split("_") if token]
        return "_".join(tokens) or "general"

    def _recommendation_cache_file(self) -> Path:
        return Path(
            getattr(
                self,
                "_recommendation_cache_path",
                ".sdk_cache/recommendations/ollama_models.json",
            )
        )

    def _resolve_api_key(
        self,
        config: LLMConfig,
        allow_placeholder: bool = True,
    ) -> str:
        direct = str(config.api_key or "").strip()
        if direct:
            return direct

        env_map = {
            LLMProvider.OPENAI: "OPENAI_API_KEY",
            LLMProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
        }
        env_name = env_map.get(config.provider)
        if env_name:
            env_value = os.getenv(env_name, "").strip()
            if env_value:
                return env_value

        if config.provider == LLMProvider.OLLAMA:
            return "ollama"
        if config.provider == LLMProvider.VLLM:
            return "vllm"
        return "no-key" if allow_placeholder else ""

    def _parse_json_response(self, raw: str) -> dict | list | None:
        """Parse messy model output into JSON, with extra repair for local models."""
        candidates = self._json_candidates(raw)
        for candidate in candidates:
            parsed = self._try_json_parse(candidate)
            if parsed is not None:
                return parsed
        return None

    def _json_candidates(self, raw: str) -> list[str]:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned)
            cleaned = cleaned.strip()

        candidates: list[str] = []
        for candidate in [cleaned, *self._extract_json_spans(cleaned)]:
            candidate = candidate.strip()
            if candidate and candidate not in candidates:
                candidates.append(candidate)

        repaired_candidates: list[str] = []
        for candidate in candidates:
            repaired_candidates.append(candidate)
            repaired = self._repair_json_text(candidate)
            if repaired != candidate and repaired not in repaired_candidates:
                repaired_candidates.append(repaired)
        return repaired_candidates

    def _extract_json_spans(self, text: str) -> list[str]:
        spans: list[str] = []
        object_start = text.find("{")
        object_end = text.rfind("}")
        if object_start >= 0 and object_end > object_start:
            spans.append(text[object_start : object_end + 1])
        list_start = text.find("[")
        list_end = text.rfind("]")
        if list_start >= 0 and list_end > list_start:
            spans.append(text[list_start : list_end + 1])
        return spans

    def _try_json_parse(self, text: str) -> dict | list | None:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    def _repair_json_text(self, text: str) -> str:
        repaired = text
        repaired = repaired.replace("\ufeff", "").strip()
        repaired = self._escape_unquoted_control_chars(repaired)
        repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
        repaired = self._balance_json_brackets(repaired)
        return repaired.strip()

    def _escape_unquoted_control_chars(self, text: str) -> str:
        result: list[str] = []
        in_string = False
        escaped = False
        for char in text:
            if in_string:
                if escaped:
                    result.append(char)
                    escaped = False
                    continue
                if char == "\\":
                    result.append(char)
                    escaped = True
                    continue
                if char == '"':
                    result.append(char)
                    in_string = False
                    continue
                if char == "\n":
                    result.append("\\n")
                    continue
                if char == "\r":
                    result.append("\\r")
                    continue
                if char == "\t":
                    result.append("\\t")
                    continue
                if ord(char) < 32:
                    result.append(f"\\u{ord(char):04x}")
                    continue
                result.append(char)
                continue
            result.append(char)
            if char == '"':
                in_string = True
                escaped = False
        return "".join(result)

    def _balance_json_brackets(self, text: str) -> str:
        open_curly = text.count("{")
        close_curly = text.count("}")
        open_square = text.count("[")
        close_square = text.count("]")
        if close_square < open_square:
            text += "]" * (open_square - close_square)
        if close_curly < open_curly:
            text += "}" * (open_curly - close_curly)
        return text

    def _list_ollama_models(self) -> list[dict[str, Any]]:
        """Query the Ollama tags endpoint and return installed models."""
        tags_url = f"{self._ollama_http_base().rstrip('/')}/api/tags"
        try:
            request = Request(tags_url, headers={"Accept": "application/json"})
            with urlopen(request, timeout=self.config.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as e:
            logger.warning(f"Ollama model discovery failed: {e}")
            return []

        models = []
        for model in payload.get("models", []):
            models.append(
                {
                    "name": model.get("name"),
                    "size": model.get("size"),
                    "modified_at": model.get("modified_at"),
                    "digest": model.get("digest"),
                    "details": model.get("details", {}),
                }
            )
        return models

    def _ollama_http_base(self) -> str:
        """Convert OpenAI-compatible Ollama base URL to Ollama HTTP base."""
        parsed = urlparse(self.config.api_base)
        path = parsed.path
        if path.endswith("/v1"):
            path = path[: -len("/v1")]
        return f"{parsed.scheme}://{parsed.netloc}{path}"
