"""Model backends.

``models.base`` defines the interface every backend implements and is safe to
import anywhere (standard library only). ``models.registry`` resolves a backend
name to a class, importing the heavy module lazily.
"""
