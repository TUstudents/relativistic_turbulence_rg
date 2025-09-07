import warnings


def _deprecated(old: str, new: str, remove: str = "v0.2") -> None:
    warnings.warn(
        f"{old} is deprecated and will be removed in {remove}; use {new} instead.",
        DeprecationWarning,
        stacklevel=2,
    )

