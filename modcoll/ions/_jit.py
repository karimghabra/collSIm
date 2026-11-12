"""Optional numba acceleration.

Every kernel in this package is plain Python that numba can compile. With
numba installed the kernels run compiled; without it they run interpreted and
give the same numbers, roughly a hundred times slower. Nothing else changes.
"""
try:
    from numba import njit as _njit

    HAVE_NUMBA = True

    def njit(*args, **kwargs):
        kwargs.setdefault("cache", True)
        return _njit(*args, **kwargs)

except ImportError:  # pragma: no cover - exercised only without numba
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def wrap(fn):
            return fn

        return wrap
