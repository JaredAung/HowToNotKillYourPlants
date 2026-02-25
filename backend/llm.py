"""
Call Ollama LLM for text generation and chat.
Requires: pip install ollama
Ensure Ollama is running: ollama serve
"""
import os

DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "120"))


def _client():
    """Get Ollama client with configured host and timeout."""
    from ollama import Client
    return Client(host=OLLAMA_HOST, timeout=OLLAMA_TIMEOUT)


def generate(prompt: str, model: str | None = None) -> str:
    """One-shot text generation. Returns the full response."""
    try:
        import ollama
    except ImportError:
        raise ImportError("Install ollama: pip install ollama")

    model = model or DEFAULT_MODEL
    response = ollama.generate(model=model, prompt=prompt)
    return response.get("response", "")


def chat(messages: list[dict], model: str | None = None, stream: bool = False):
    """
    Chat with the LLM.
    messages: [{"role": "user"|"system"|"assistant", "content": "..."}]
    Returns full response dict or stream generator.
    """
    try:
        from ollama import Client
    except ImportError:
        raise ImportError("Install ollama: pip install ollama")

    model = model or DEFAULT_MODEL
    client = _client()
    response = client.chat(model=model, messages=messages, stream=stream)
    return response


def chat_simple(user_message: str, system: str | None = None, model: str | None = None) -> str:
    """Convenience: send a user message, optionally with system prompt. Returns assistant reply."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_message})

    response = chat(messages=messages, model=model)
    msg = getattr(response, "message", None)
    if msg is None and isinstance(response, dict):
        msg = response.get("message")
    if msg is not None:
        content = getattr(msg, "content", None)
        if content is None and isinstance(msg, dict):
            content = msg.get("content", "")
        if content:
            return content
    return ""
