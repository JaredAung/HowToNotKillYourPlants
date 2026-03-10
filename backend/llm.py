"""
LLM module: Ollama and Gemini interchangeable via USE_GEMINI env var.

Set USE_GEMINI=true (default) or USE_GEMINI=false for Ollama.

Ollama: Requires ollama serve. Uses OLLAMA_HOST, OLLAMA_MODEL.
Gemini: Requires GEMINI_API_KEY (or GOOGLE_API_KEY). Uses GEMINI_MODEL.
"""
import os

use_gemini = os.getenv("USE_GEMINI", "true").lower() in ("true", "1", "yes")

# Ollama config
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "120"))

# Gemini config
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

_llm_instance = None


def _get_ollama_llm():
    """Create LangChain ChatOllama instance."""
    from langchain_ollama import ChatOllama
    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_HOST,
        temperature=0,
    )


def _get_gemini_llm():
    """Create LangChain ChatGoogleGenerativeAI instance."""
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY or GOOGLE_API_KEY required for Gemini. "
            "Set in .env or use USE_GEMINI=false for Ollama."
        )
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GEMINI_API_KEY,
        temperature=0,
    )


def _get_llm():
    """Return the active LangChain Chat model (Ollama or Gemini). Lazy-init."""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = _get_gemini_llm() if use_gemini else _get_ollama_llm()
    return _llm_instance


# Primary export: the active LLM for LangChain chains
# Use: from llm import llm  or  from llm import ollama_llm (backward compat)


_gemini_llm_instance = None


def get_gemini_llm():
    """Return LangChain ChatGoogleGenerativeAI. For LangGraph only."""
    global _gemini_llm_instance
    if _gemini_llm_instance is None:
        _gemini_llm_instance = _get_gemini_llm()
    return _gemini_llm_instance


def gemini_generate(system: str | None, user_message: str, model: str | None = None) -> str:
    """
    Use native google-genai SDK for non-LangGraph parts (recommendation explanation, search extract).
    Returns assistant reply string.
    """
    from google import genai
    from google.genai import types

    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY required for gemini_generate.")
    client = genai.Client(api_key=GEMINI_API_KEY)
    config_kw = {"temperature": 0}
    if system:
        config_kw["system_instruction"] = system
    response = client.models.generate_content(
        model=model or GEMINI_MODEL,
        contents=user_message,
        config=types.GenerateContentConfig(**config_kw),
    )
    return (response.text or "").strip()


def __getattr__(name):
    if name == "llm":
        return _get_llm()
    if name == "ollama_llm":
        # Backward compat: same as llm (active provider)
        return _get_llm()
    if name == "gemini_llm":
        return get_gemini_llm()
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
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

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
