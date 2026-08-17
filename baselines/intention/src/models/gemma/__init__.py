"""Gemma backend package.

Nothing is imported eagerly here: ``models.gemma.engine`` pulls in torch and
transformers, so it is left to ``models.registry`` to import it only when a
Gemma model is actually requested.
"""
