"""Rank-blend study: a component pool and a weight search over it.

`weights` is pure numpy and has no optional dependencies; `components` needs lightgbm and
torch. Importing them eagerly here made `blend.weights` -- and every test of it -- unimportable
on a machine without lightgbm, even though the weight search never touches it. The names stay
available at package level; they are just resolved on first use.
"""

__all__ = ["available", "fit_predict"]


def __getattr__(name: str):
    if name in __all__:
        from . import components
        return getattr(components, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
