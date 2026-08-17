"""The interface every model backend implements.

A task builds turns out of :class:`MediaPart` values -- a type plus a path or a
string, never a decoded array -- and hands them to :meth:`BaseMultimodalModel.generate`.
Decoding, chat templating, tokenisation and sampling all live behind that call,
so task code needs no torch, no transformers, and no knowledge of which model is
running.

This module is deliberately dependency-free (standard library only). Importing it
must never pull in a machine-learning stack: see ``models/registry.py``, which
imports backend modules lazily for exactly that reason.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

PartType = Literal["text", "image", "audio", "video"]
Role = Literal["system", "user", "assistant"]

# Capabilities a backend can advertise through supports(). "thinking" is Gemma's
# reasoning mode; a task that needs it must ask before enabling it, so that an
# unsupported flag fails loudly instead of being dropped on the floor.
Capability = Literal["text", "image", "audio", "video", "thinking", "multi_turn"]

DEFAULT_SYSTEM_PROMPT = "You are a helpful multimodal assistant."


@dataclass(frozen=True)
class MediaPart:
    """One typed piece of a turn: a path plus the intent to use it as media.

    Media is carried as a path rather than as decoded pixels or samples so that
    a local backend can decode it the way its processor wants, while a hosted
    backend can upload the file instead of ever decoding it.
    """

    type: PartType
    text: str | None = None
    path: str | Path | None = None
    # Concrete frame count for a video part. Set by the backend through
    # prepare_video_part(), never guessed by the task.
    num_frames: int | None = None

    @classmethod
    def text_part(cls, text: str) -> "MediaPart":
        return cls(type="text", text=text)

    @classmethod
    def image(cls, path: str | Path) -> "MediaPart":
        return cls(type="image", path=path)

    @classmethod
    def audio(cls, path: str | Path) -> "MediaPart":
        return cls(type="audio", path=path)

    @classmethod
    def video(cls, path: str | Path, num_frames: int | None = None) -> "MediaPart":
        return cls(type="video", path=path, num_frames=num_frames)


@dataclass
class ChatMessage:
    role: Role
    content: list[MediaPart]


@dataclass
class GenerationConfig:
    """Decoding parameters shared by every backend.

    Backend-only knobs go in ``extra`` rather than growing this class: a field
    here is a promise that every backend honours it, and ``extra`` carries no
    such promise. Callers check supports() before setting one.
    """

    max_new_tokens: int = 512
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 64
    extra: dict[str, Any] = field(default_factory=dict)


class BaseMultimodalModel(ABC):
    """One loaded model.

    Constructing a subclass loads the weights; ``generate`` is then a pure call.
    Backends are expected to raise on failure -- the caller owns how errors are
    recorded, so that error strings stay in one place per task rather than being
    invented differently by every backend.
    """

    name: str = "base"
    # Declared on the class, not the instance, so a caller can check a backend's
    # capabilities before paying to load its weights (see
    # models.registry.backend_capabilities).
    capabilities: frozenset[str] = frozenset({"text"})

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id

    @abstractmethod
    def generate(
        self,
        messages: Sequence[ChatMessage],
        config: GenerationConfig,
    ) -> str:
        """Return the assistant text for one turn."""

    def prepare_video_part(
        self,
        video_path: str | Path,
        max_frames: int,
        *,
        fps: float = 2.0,
        min_frames: int = 4,
    ) -> MediaPart:
        """Turn the run's frame-sampling policy into a concrete video part.

        The policy -- a target rate with a floor and a ceiling -- is a property
        of the run, not of any one backend, because frame count is the largest
        controllable confound when two models are compared. Every
        frame-sampling backend resolves it through
        ``models.media_io.frames_at_fps`` so all of them see the same frames of
        the same clip; upload-based backends keep this default and leave
        ``num_frames`` unset, since the service does its own sampling.
        """
        return MediaPart.video(video_path, num_frames=None)

    def supports(self, capability: Capability) -> bool:
        return capability in self.capabilities

    def close(self) -> None:
        """Release weights or sessions. Default: nothing to release."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, model_id={self.model_id!r})"


def combine_system_and_user_prompt(system_prompt: str, user_prompt: str) -> str:
    """Fold a system prompt into the user text.

    Kept here rather than in a task module because both the intention-narrative
    and context tasks need it, and because backends that have no system role
    still get the instruction this way.
    """
    system_prompt = system_prompt.strip()
    user_prompt = user_prompt.strip()
    if not system_prompt:
        return user_prompt
    return f"{system_prompt}\n\n{user_prompt}"
