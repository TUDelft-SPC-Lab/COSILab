"""Looking up a backend by name.

Backend modules are resolved lazily, by import path, and only when one is
actually requested. That is what keeps a task's non-inference entrypoints --
runnable with the standard library alone: importing this module must never drag
in torch.
"""

from __future__ import annotations

import importlib
from typing import Any

from models.base import BaseMultimodalModel

__all__ = [
    "BACKEND_MODULES",
    "available_backends",
    "backend_capabilities",
    "default_model_id",
    "load_model",
]

# name -> "module path:class name". Add a backend by implementing
# BaseMultimodalModel and adding one line here.
#
# The Qwen generation and size are part of the backend name rather than separate
# flags: the name is what a run records and what the job scripts turn into a
# results directory, so "which weights ran" stays answerable from the backend
# alone. Another size of a generation already listed is one more line here plus a
# backends.<name> section in the task's model_config.json, both pointing at the
# same class -- qwen7b and a hypothetical qwen3b would share QwenOmniModel.
# Another *generation* needs its own class, as qwen3omni30b does, because the
# transformers model classes and the config tree differ.
BACKEND_MODULES: dict[str, str] = {
    "gemma": "models.gemma.engine:GemmaModel",
    "qwen7b": "models.qwen.engine:QwenOmniModel",
    "qwen3omni30b": "models.qwen3omni.engine:Qwen3OmniModel",
}

# No default backend on purpose. Which model produced a result is the first thing
# anyone asks of it, and here it is also a path component, so letting it be
# decided by omission is how a run ends up filed under a model nobody chose.
# Callers make --backend required.


def available_backends() -> list[str]:
    return sorted(BACKEND_MODULES)


def _resolve_backend_class(backend: str) -> type[BaseMultimodalModel]:
    try:
        target = BACKEND_MODULES[backend]
    except KeyError:
        raise ValueError(
            f"Unknown model backend: {backend!r}. "
            f"Available: {', '.join(available_backends())}"
        ) from None
    module_name, _, class_name = target.partition(":")
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def default_model_id(backend: str) -> str:
    """The weights path or model id a backend uses when none is given."""
    backend_class = _resolve_backend_class(backend)
    module = importlib.import_module(backend_class.__module__)
    model_id = getattr(module, "DEFAULT_MODEL_PATH", None)
    if not isinstance(model_id, str) or not model_id:
        raise ValueError(f"Backend {backend!r} declares no DEFAULT_MODEL_PATH.")
    return model_id


def backend_capabilities(backend: str) -> frozenset[str]:
    """What a backend can do, without loading its weights.

    Resolving the class imports the backend module (and its ML stack), but not
    the model itself, so a task can reject an unsupported flag before spending
    minutes on weights.
    """
    return _resolve_backend_class(backend).capabilities


def load_model(
    backend: str,
    model_id: str | None = None,
    **backend_kwargs: Any,
) -> BaseMultimodalModel:
    """Load one backend's weights and return it ready to generate."""
    backend_class = _resolve_backend_class(backend)
    if model_id is None:
        model_id = default_model_id(backend)
    return backend_class(model_id, **backend_kwargs)
