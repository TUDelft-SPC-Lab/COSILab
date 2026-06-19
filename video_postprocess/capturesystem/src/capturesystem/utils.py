import os
from contextlib import AbstractContextManager
from enum import Enum
from pathlib import Path
from typing import TypeVar

_EnumType = TypeVar("_EnumType", bound=Enum)


def get_camera_to_process() -> str | None:
    camera_to_process = os.getenv("CAMERA_TO_PROCESS")
    if camera_to_process == "":
        camera_to_process = None
    return camera_to_process


def get_num_threads(max_threads: int | None = None) -> int:
    if max_threads is not None:
        return max_threads

    # See https://doc.daic.tudelft.nl/support/faqs/job-resources/#how-do-i-request-cpus-for-a-multithreaded-program
    # for why os.sched_getaffinity(0) is used and not os.cpu_count()
    # allocated_threads = len(os.sched_getaffinity(0)) - 1
    allocated_threads = 0  # on windows, os.sched_getaffinity(0) is not supported
    print(f"num allocated_threads is {allocated_threads}")
    return allocated_threads


# A simple context manager that gets initialised with a Path and returns it on entering the context
class SimplePath(AbstractContextManager):
    def __init__(self, file_path: Path) -> None:
        super().__init__()
        self.file_path = file_path

    def __enter__(self) -> Path:
        return self.file_path

    def __exit__(self, exc_type, exc_value, exc_tb):
        return True
