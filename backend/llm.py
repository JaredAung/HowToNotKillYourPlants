"""
LLM routing (Google Gemini only):

- **LangGraph / LangChain chat** (``llm`` / ``gemini_llm``): ``ChatGoogleGenerativeAI``
  (``GEMINI_API_KEY`` / ``GOOGLE_API_KEY``, ``GEMINI_MODEL``).

- **Recommendation NL explanations**, **search**, and other one-shot text: ``gemini_generate``
  via the Google GenAI client (same API key / model env vars).
"""
import os
from typing import Any, Iterator

from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig


def get_active_llm() -> str:
    """Provider label for observability; stack is always Gemini."""
    return "gemini"


def _gemini_model() -> str:
    return os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def _gemini_api_key() -> str | None:
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


def _get_gemini_llm():
    """Create LangChain ChatGoogleGenerativeAI instance."""
    key = _gemini_api_key()
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY or GOOGLE_API_KEY required for LangGraph/chat and gemini_generate."
        )
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=_gemini_model(),
        google_api_key=key,
        temperature=0,
    )


def _get_llm():
    """LangChain chat model for LangGraph: Gemini (cached via ``get_gemini_llm``)."""
    return get_gemini_llm()


def reset_llm_cache() -> None:
    """Drop cached Gemini LangChain instance (e.g. after tests)."""
    global _gemini_llm_instance, _gemini_llm_cache_key
    _gemini_llm_instance = None
    _gemini_llm_cache_key = None


class _LazyActiveLLM(Runnable):
    """
    Runnable facade so ``prompt | gemini_llm`` always uses the current Gemini LangChain client.

    Without this, a cached import would freeze the first model/key forever.
    """

    def invoke(self, input: Any, config: RunnableConfig | None = None, **kwargs: Any) -> Any:
        return _get_llm().invoke(input, config=config, **kwargs)

    def stream(self, input: Any, config: RunnableConfig | None = None, **kwargs: Any) -> Iterator[Any]:
        return _get_llm().stream(input, config=config, **kwargs)

    def bind_tools(self, tools, **kwargs):
        return _get_llm().bind_tools(tools, **kwargs)

    def __getattr__(self, name):
        return getattr(_get_llm(), name)


_lazy_active_llm = _LazyActiveLLM()


_gemini_llm_instance = None
_gemini_llm_cache_key: tuple | None = None


def get_gemini_llm():
    """Return LangChain ChatGoogleGenerativeAI. Cache keyed by model + API key."""
    global _gemini_llm_instance, _gemini_llm_cache_key
    sig = (_gemini_model(), _gemini_api_key() or "")
    if _gemini_llm_instance is None or sig != _gemini_llm_cache_key:
        _gemini_llm_cache_key = sig
        _gemini_llm_instance = _get_gemini_llm()
    return _gemini_llm_instance


def gemini_generate(system: str | None, user_message: str, model: str | None = None) -> str:
    """Single-shot text via Google GenAI (e.g. search, recommendation copy)."""
    from google import genai
    from google.genai import types

    key = _gemini_api_key()
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY or GOOGLE_API_KEY required for gemini_generate()."
        )
    client = genai.Client(api_key=key)
    config_kw = {"temperature": 0}
    if system:
        config_kw["system_instruction"] = system
    response = client.models.generate_content(
        model=model or _gemini_model(),
        contents=user_message,
        config=types.GenerateContentConfig(**config_kw),
    )
    return (response.text or "").strip()


def __getattr__(name):
    if name in ("llm", "gemini_llm"):
        return _lazy_active_llm
    if name == "use_gemini":
        return True
    if name == "GEMINI_MODEL":
        return _gemini_model()
    if name == "GEMINI_API_KEY":
        return _gemini_api_key()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def generate(prompt: str, model: str | None = None) -> str:
    """One-shot text generation. Returns the full response."""
    llm_instance = _get_llm()
    from langchain_core.messages import HumanMessage

    response = llm_instance.invoke([HumanMessage(content=prompt)])
    return getattr(response, "content", "") or ""


def chat(messages: list[dict], model: str | None = None, stream: bool = False):
    """
    Chat with the LLM.
    messages: [{"role": "user"|"system"|"assistant", "content": "..."}]
    Returns assistant reply string (stream=False) or stream generator (stream=True).
    """
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    def _to_lc(msg: dict):
        role = (msg.get("role") or "user").lower()
        content = msg.get("content", msg.get("text", ""))
        if role == "system":
            return SystemMessage(content=content)
        if role == "assistant":
            return AIMessage(content=content)
        return HumanMessage(content=content)

    lc_messages = [_to_lc(m) for m in messages]
    llm_instance = _get_llm()

    if stream:
        return llm_instance.stream(lc_messages)
    response = llm_instance.invoke(lc_messages)
    return getattr(response, "content", "") or ""


def chat_simple(user_message: str, system: str | None = None, model: str | None = None) -> str:
    """Convenience: send a user message, optionally with system prompt. Returns assistant reply."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_message})
    return chat(messages=messages, model=model, stream=False)
