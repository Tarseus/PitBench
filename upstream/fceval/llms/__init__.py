__all__ = ["Chat", "LiteLLM", "PortkeyLiteLLM"]


def __getattr__(name: str):
    if name == "Chat":
        from fceval.llms.chat import Chat

        return Chat
    if name == "LiteLLM":
        from fceval.llms.lite_llm import LiteLLM

        return LiteLLM
    if name == "PortkeyLiteLLM":
        from fceval.llms.portkey_llm import PortkeyLiteLLM

        return PortkeyLiteLLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
