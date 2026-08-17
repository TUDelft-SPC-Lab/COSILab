"""Qwen2.5-Omni backend package.

Nothing is imported eagerly here: ``models.qwen.engine`` pulls in torch and
transformers, so it is left to ``models.registry`` to import it only when a Qwen
model is actually requested. ``qwen-omni-utils`` is not vendored -- it comes from
the container image, or from ``QWEN_OMNI_UTILS_PATH`` -- and the engine imports
it on first use rather than at module import.
"""
