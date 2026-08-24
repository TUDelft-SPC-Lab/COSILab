"""Qwen3-Omni backend package.

Nothing is imported eagerly here: ``models.qwen3omni.engine`` pulls in torch and
transformers, so it is left to ``models.registry`` to import it only when a
Qwen3-Omni model is actually requested. The plumbing it shares with the
Qwen2.5-Omni backend lives in ``models.qwen.shared``; ``qwen-omni-utils`` serves
both generations and is not vendored -- it comes from the container image, or
from ``QWEN_OMNI_UTILS_PATH`` -- and the engine imports it on first use rather
than at module import.
"""
