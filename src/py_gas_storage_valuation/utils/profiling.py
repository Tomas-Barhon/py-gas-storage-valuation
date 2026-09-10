from __future__ import annotations

import time
from collections.abc import Callable
from functools import wraps
from typing import Any


class Timer:
    """Times a function (decorator) or a code block (context manager).

    Usage as decorator::

        @timer
        def step(action):
            ...

        @timer(label="custom label")
        def reset(seed=None):
            ...

    Usage as context manager::

        with Timer("data load"):
            x = expensive_call()
    """

    def __init__(self, label: str | None = None) -> None:
        self.label = label
        self._start: float = 0.0

    # -- context manager ---------------------------------------------------

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: Any) -> None:
        elapsed = time.perf_counter() - self._start
        print(f"[Timer] {self.label or 'block'}: {elapsed:.6f}s")

    # -- decorator ---------------------------------------------------------

    def __call__(self, func: Callable) -> Callable:
        label = self.label or getattr(func, "__name__", "unknown")

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            print(f"[Timer] {label}: {elapsed:.6f}s")
            return result

        return wrapper


def timer(
    func: Callable | None = None, *, label: str | None = None
) -> Callable:
    """Shortcut so you can write ``@timer`` without parentheses."""
    if func is not None:
        return Timer()(func)
    return Timer(label=label)
