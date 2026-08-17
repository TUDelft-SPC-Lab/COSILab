"""Task modes: the variants of the intention task.

A mode owns what is asked and how it is grounded -- which visual reference a
record needs, how the prompt is rendered, and in what order the parts of the turn
are laid out. It does not own which model answers: that is ``--backend`` and the
``models/`` package, and the two axes are independent.

``modes.base`` defines the interface every mode implements; ``modes.registry``
resolves a mode name to a class. Neither imports torch, and neither should: a
mode builds ``models.base.MediaPart`` values out of paths and hands them on.
"""
