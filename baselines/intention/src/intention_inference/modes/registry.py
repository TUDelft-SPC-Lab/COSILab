"""Looking up a task mode by name.

Deliberately the same shape as ``models/registry.py``, down to the lazy import,
so the two axes of a run read alike. The laziness is not load-bearing here the
way it is there -- a mode is standard library plus ``models.base``, and importing
one costs nothing -- it is kept because a mode that later needs a heavy import
(drawing id overlays, say) should not become a special case.
"""

from __future__ import annotations

import importlib

from .base import BaseTaskMode

__all__ = [
    "MODE_MODULES",
    "available_modes",
    "load_mode",
]

# name -> "module path:class name". Add a mode by implementing BaseTaskMode and
# adding one line here.
#
# The three named here are one axis: how much of the group's audio is stacked
# into a single soundtrack, and therefore how much of it can be attributed to a
# person.
#
#   sa   1 audio part    everyone stacked, person of interest included
#   pa   2 audio parts   person of interest, and everyone else stacked
#   fa   N audio parts   one per person, each labelled with their id
#
# A placeholder belongs in this table too, not outside it: an unfinished mode
# that is merely absent looks to the caller exactly like a typo, whereas one that
# is present and raises can say what remains to be written.
MODE_MODULES: dict[str, str] = {
    "pa": "intention_inference.modes.pa:PartiallyStackedAudioMode",
    "fa": "intention_inference.modes.fa:FlattenedAudioMode",
    "sa": "intention_inference.modes.sa:StackedAudioMode",
}

# No default mode, for the reason --backend has none: which variant of the
# question produced a result is the first thing anyone asks of it, and here it is
# also a path component. Callers make --mode required.


def available_modes() -> list[str]:
    return sorted(MODE_MODULES)


def _resolve_mode_class(mode: str) -> type[BaseTaskMode]:
    try:
        target = MODE_MODULES[mode]
    except KeyError:
        raise ValueError(
            f"Unknown task mode: {mode!r}. Available: {', '.join(available_modes())}"
        ) from None
    module_name, _, class_name = target.partition(":")
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def load_mode(mode: str) -> BaseTaskMode:
    """Instantiate one mode. A placeholder mode raises here, by design."""
    return _resolve_mode_class(mode)()
