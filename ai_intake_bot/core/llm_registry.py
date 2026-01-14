from typing import Dict, Type
from .llm import BaseLLM

_LLM_REGISTRY: Dict[str, Type[BaseLLM]] = {}


def register_llm(name: str, llm_cls: Type[BaseLLM]):
    if not issubclass(llm_cls, BaseLLM):
        raise TypeError("LLM must extend BaseLLM")
    _LLM_REGISTRY[name] = llm_cls


def create_llm(name: str, **kwargs) -> BaseLLM:
    if name not in _LLM_REGISTRY:
        raise ValueError(f"Unknown LLM provider: {name}")
    return _LLM_REGISTRY[name](**kwargs)
