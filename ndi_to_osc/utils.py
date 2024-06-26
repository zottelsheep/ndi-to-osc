from itertools import islice
import typing as t
import weakref
import functools
import numpy as np
import colorsys

T = t.TypeVar('T')
def batched(iterable:t.Iterable[T], n:int) -> t.Iterator[t.Sequence[T]]:
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


def lru_method_cache(maxsize=128, typed=False):
    """Creates a lru_cache_wrapper that binds to instances of classes"""
    def decorator(func):
        @functools.wraps(func)
        def wrapped_func(self, *args, **kwargs):
            self_weak = weakref.ref(self)
            @functools.wraps(func)
            @functools.lru_cache(maxsize=maxsize, typed=typed)
            def cached_method(*args, **kwargs):
                return func(self_weak(), *args, **kwargs)
            setattr(self, func.__name__, cached_method)
            return cached_method(*args, **kwargs)
        return wrapped_func

    if callable(maxsize) and isinstance(typed, bool):
        # The user_function was passed in directly via the maxsize argument
        func, maxsize = maxsize, 128
        return decorator(func)

    return decorator

P = t.ParamSpec('P')
T = t.TypeVar('T')
def cache_method(func: t.Callable[P,T]) -> t.Callable[P,T]:
    return lru_method_cache(maxsize=None)(func)
hsl_to_rgb_vec = np.vectorize(lambda rgb: np.array(colorsys.hls_to_rgb(*rgb)),
                              signature='(3)->(3)')

def fade_rgb(from_rgb: tuple[int,int,int],
             to_rgb: tuple[int,int,int],
             steps: int):
    from_hsl = colorsys.rgb_to_hls(*(np.array(from_rgb)))
    to_hsl = colorsys.rgb_to_hls(*(np.array(to_rgb)))

    fades = np.linspace(from_hsl,to_hsl,steps)
    fades_rgb = hsl_to_rgb_vec(fades).round().astype(int)
    return fades_rgb


