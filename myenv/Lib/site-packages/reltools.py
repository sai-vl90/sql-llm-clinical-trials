"""Relation tools for Python."""

from abc import ABCMeta, abstractmethod
from itertools import groupby
from operator import itemgetter
from typing import Callable, Generic, Iterable, Iterator, Tuple, TypeVar

__all__ = [
    'OneToManyChainer',
    'relate_one_to_many',
    'left_join',
    'outer_join',
    'inner_join',
]

T = TypeVar('T')


class _Peekable(Generic[T], Iterator[T]):
    """
    An iterator where the current element can be fetched.

    When given an empty iterator, then only stops iteration.
    >>> peekable = _Peekable(iter([]))
    >>> bool(peekable)
    False
    >>> peekable.peek()
    Traceback (most recent call last):
        ...
    StopIteration
    >>> next(peekable)
    Traceback (most recent call last):
        ...
    StopIteration
    >>> for item in _Peekable(iter([])):
    ...     item

    When given a filled iterator, then peeks and iterates it.
    >>> peekable = _Peekable(iter([1, 2]))
    >>> bool(peekable)
    True
    >>> peekable.peek()
    1
    >>> next(peekable)
    1
    >>> bool(peekable)
    True
    >>> peekable.peek()
    2
    >>> next(peekable)
    2
    >>> peekable.peek()
    Traceback (most recent call last):
        ...
    StopIteration
    >>> next(peekable)
    Traceback (most recent call last):
        ...
    StopIteration
    >>> for item in _Peekable(iter([1, 2])):
    ...     item
    1
    2

    Peeks values as lazyly as possible.
    >>> iterator = iter([1])
    >>> _ = _Peekable(iterator)
    >>> for item in iterator:
    ...     item
    1
    >>> iterator = iter([1, 2])
    >>> peekable = _Peekable(iterator)
    >>> peekable.peek()
    1
    >>> next(peekable)
    1
    >>> for item in iterator:
    ...     item
    2
    """

    __NO_VALUE = object()

    def __init__(self, iterable: Iterable[T]):
        self._iterator = iter(iterable)
        self._current: object = self.__NO_VALUE  # T or __NO_VALUE

    def peek(self) -> T:
        if self._current is self.__NO_VALUE:
            self._current = next(self._iterator)
        return self._current  # type: ignore

    def __iter__(self) -> Iterator[T]:
        return self

    def __next__(self) -> T:
        current = self.peek()
        self._current = self.__NO_VALUE
        return current  # type: ignore

    def __bool__(self) -> bool:
        try:
            self.peek()
        except StopIteration:
            return False
        return True


# HACK: implemented for Python 3.6, may be replaced to use typing.Protocol.
class _Comparable(metaclass=ABCMeta):
    """Protocol for annotating comparable types."""

    @abstractmethod
    def __lt__(self, other) -> bool: ...

    @abstractmethod
    def __le__(self, other) -> bool: ...

    @abstractmethod
    def __gt__(self, other) -> bool: ...

    @abstractmethod
    def __ge__(self, other) -> bool: ...

    @abstractmethod
    def __eq__(self, other) -> bool: ...

    @abstractmethod
    def __ne__(self, other) -> bool: ...


Key = TypeVar('Key', bound=_Comparable)
Value = TypeVar('Value')
Left = TypeVar('Left')
Right = TypeVar('Right')

_EMPTY_ITERABLE: Iterable = tuple()
FIRST_ITEM_KEY = itemgetter(0)
DEFAULT_KEY = FIRST_ITEM_KEY


class _UnidirectionalFinder(Generic[Value, Key], Iterator[Iterator[Value]]):
    """
    This class finds items in `iterable` unidirectionally
    and groups them by the given `key`.

    Note that the `Key` must be 'comparable'
    (supports `__lt__()` and  `__gt__()`)
    and `iterable` must be sorted by `key`.

    Here are some normal cases.
    Collections are sorted by the first items.
    >>> iterable = [(0, 'a'), (1, 'b'), (1, 'c'), (2, 'd')]
    >>> finder = _UnidirectionalFinder(iterable, itemgetter(0))
    >>> finder.has_items
    True
    >>> finder.current_key()
    0

    When given a waiting key, then finds items and groups them by the key.
    >>> list(finder.find(1))
    [(1, 'b'), (1, 'c')]
    >>> finder.current_key()
    2

    When given passed keys, then cannot find items.
    >>> list(finder.find(1))
    []
    >>> list(finder.find(0))
    []

    When given too large keys, then cannot find items.
    Once given a too large key, the finder is exhausted.
    >>> list(finder.find(3))
    []
    >>> finder.has_items
    False
    >>> finder.current_key()
    Traceback (most recent call last):
        ...
    StopIteration

    When exhausted and given an existing key, then cannot find items.
    >>> list(finder.find(2))
    []

    Sequencial usage is also supported.
    >>> iterable = [(0, 'a'), (1, 'b'), (1, 'c'), (2, 'd'), (3, 'e')]
    >>> finder = _UnidirectionalFinder(iterable, itemgetter(0))
    >>> finder.current_key()
    0
    >>> list(next(finder))
    [(0, 'a')]
    >>> finder.current_key()
    1
    >>> list(next(finder))
    [(1, 'b'), (1, 'c')]
    >>> finder.current_key()
    2
    >>> list(finder.find(3))
    [(3, 'e')]
    >>> finder.current_key()
    Traceback (most recent call last):
        ...
    StopIteration
    >>> next(finder)
    Traceback (most recent call last):
        ...
    StopIteration

    Here are some seminormal cases.
    When given an empty collection, then cannot find any items.
    >>> finder = _UnidirectionalFinder([], itemgetter(0))
    >>> finder.has_items
    False
    >>> list(finder.find(0))
    []
    >>> finder.current_key()
    Traceback (most recent call last):
        ...
    StopIteration

    When given a not sorted `iterable`,
    then stops finding at reverse-ordering segments.
    >>> iterable = [(0, 'a'), (2, 'b'), (1, 'c'), (3, 'd')]
    >>> finder = _UnidirectionalFinder(iterable, itemgetter(0))
    >>> list(finder.find(1))
    []
    >>> finder.current_key()
    2
    >>> list(finder.find(2))
    [(2, 'b')]
    >>> list(finder.find(3))
    [(3, 'd')]
    """

    def __init__(
        self,
        iterable: Iterable[Value],
        key: Callable[[Value], Key],
    ) -> None:
        """Initialize"""
        self._groups = _Peekable(groupby(iterable, key))

    def find(self, key: Key) -> Iterator[Value]:
        """Find items that have the given key."""
        self.seek_to(key)

        if not self.has_items:
            return iter(_EMPTY_ITERABLE)

        group_key, group_items = self._groups.peek()
        if group_key > key:
            return iter(_EMPTY_ITERABLE)

        next(self)
        return group_items

    def seek_to(self, key: Key) -> None:
        """Seek to the given key."""
        try:
            while self._groups.peek()[0] < key:
                next(self._groups)
        except StopIteration:
            pass

    @property
    def has_items(self) -> bool:
        """Check if the iterator has items."""
        return bool(self._groups)

    def current_key(self) -> Key:
        """
        Returns the current key.
        When exhausted, then throws StopIteration.
        """
        return self._groups.peek()[0]

    def __next__(self) -> Iterator[Value]:
        """
        Returns the current value and move to the next.
        When exhausted, then throws StopIteration.
        """
        return next(self._groups)[1]


class OneToManyChainer(Generic[Left]):
    """
    Relate `lhs` to one or more `rhs`.

    When given no `rhs`, then iterates the tuple of `lhs`.
    >>> lhs = [(0, 'a'), (1, 'b'), (2, 'c')]
    >>> chainer = OneToManyChainer(lhs)
    >>> for left in chainer.chain():
    ...     left
    ((0, 'a'),)
    ((1, 'b'),)
    ((2, 'c'),)

    When given `lhs`, `rhs1` and `rhs2`,
    then iterates the tuple of (`lhs`, `rhs1`, `rhs2`)
    >>> rhs1 = [(1, 's'), (2, 't'), (2, 'u'), (3, 'v')]
    >>> rhs2 = [('a', 'x'), ('a', 'y'), ('b', 'z')]
    >>> chainer = OneToManyChainer(lhs)
    >>> chainer.append(rhs1)
    >>> chainer.append(rhs2, lhs_key=itemgetter(1), rhs_key=itemgetter(0))
    >>> for left, right1, right2 in chainer.chain():
    ...     left, list(right1), list(right2)
    ((0, 'a'), [], [('a', 'x'), ('a', 'y')])
    ((1, 'b'), [(1, 's')], [('b', 'z')])
    ((2, 'c'), [(2, 't'), (2, 'u')], [])
    """

    def __init__(self, lhs: Iterable[Left]):
        self._lhs = lhs
        self._chain: list = []

    def append(
        self,
        rhs: Iterable[Right],
        lhs_key: Callable[[Left], Key] = DEFAULT_KEY,
        rhs_key: Callable[[Right], Key] = DEFAULT_KEY,
    ) -> None:
        item = (lhs_key, _UnidirectionalFinder(rhs, rhs_key))
        self._chain.append(item)

    def chain(self) -> Iterator[Tuple[Left, ...]]:
        return (self._next(item) for item in self._lhs)

    def _next(self, item: Left) -> Tuple[Left, ...]:
        rs = (
            r_finder.find(lhs_key(item))
            for lhs_key, r_finder in self._chain
        )
        return (item, *rs)


def relate_one_to_many(
    lhs: Iterable[Left],
    rhs: Iterable[Right],
    lhs_key: Callable[[Left], Key] = DEFAULT_KEY,
    rhs_key: Callable[[Right], Key] = DEFAULT_KEY,
) -> Iterator[Tuple[Left, Iterator[Right]]]:
    """
    Relates `rhs` items to each `lhs` items.

    Note that:
    - `Key` must be 'comparable' (supports `__lt__()` and  `__gt__()`).
    - `lhs` must be sorted by keys that `lhs_key` provides.
    - `rhs` must be sorted by keys that `rhs_key` provides.

    `lhs_key` and `rhs_key` are optional.
    When not given, then relates `rhs` to `lhs`
    by their first items (`left[0]` and `right[0]`).

    Here are some normal cases.
    These collections are sorted by the first items.
    >>> lhs = [(0, 'a'), (1, 'b'), (2, 'c')]
    >>> rhs = [(1, 's'), (2, 't'), (2, 'u'), (3, 'v')]

    When not given any keys,
    then relates `rhs` to `lhs` by their first items.
    >>> relations = relate_one_to_many(lhs, rhs)
    >>> for left, right in relations:
    ...     left, list(right)
    ((0, 'a'), [])
    ((1, 'b'), [(1, 's')])
    ((2, 'c'), [(2, 't'), (2, 'u')])

    When given custom keys, then relates `rhs` to `lhs` by that keys.
    Note that the custom keys *must not* break the key ordering.
    >>> relations = relate_one_to_many(
    ...     lhs, rhs,
    ...     lhs_key=lambda l: l[0] * 2,
    ...     rhs_key=lambda r: r[0] - 1)
    >>> for left, right in relations:
    ...     left, list(right)
    ((0, 'a'), [(1, 's')])
    ((1, 'b'), [(3, 'v')])
    ((2, 'c'), [])

    Here are some seminormal cases.
    When given an empty `lhs`, then returns an empty iterator.
    >>> relations = relate_one_to_many([], [(1, 's')])
    >>> list(relations)
    []

    When given an empty `rhs`, then returns an iterator that relates nothing.
    >>> relations = relate_one_to_many([(1, 'a')], [])
    >>> for left, right in relations:
    ...     left, list(right)
    ((1, 'a'), [])

    When given unordered `lhs`, then stops relationing
    at reverse-ordering segments.
    >>> lhs = [(0, 'a'), (2, 'b'), (1, 'c'), (4, 'd'), (3, 'e')]
    >>> rhs = [(1, 's'), (2, 't'), (2, 'u'), (3, 'v'), (4, 'w')]
    >>> relations = relate_one_to_many(lhs, rhs)
    >>> for left, right in relations:
    ...     left, list(right)
    ((0, 'a'), [])
    ((2, 'b'), [(2, 't'), (2, 'u')])
    ((1, 'c'), [])
    ((4, 'd'), [(4, 'w')])
    ((3, 'e'), [])

    When given unordered `rhs`, then stops relationing
    at reverse-ordering segments.
    >>> lhs = [(0, 'a'), (1, 'b'), (2, 'c'), (3, 'd'), (4, 'e')]
    >>> rhs = [(1, 's'), (3, 't'), (3, 'u'), (2, 'v'), (4, 'w')]
    >>> relations = relate_one_to_many(lhs, rhs)
    >>> for left, right in relations:
    ...     left, list(right)
    ((0, 'a'), [])
    ((1, 'b'), [(1, 's')])
    ((2, 'c'), [])
    ((3, 'd'), [(3, 't'), (3, 'u')])
    ((4, 'e'), [(4, 'w')])
    """

    chainer = OneToManyChainer(lhs)
    chainer.append(rhs, lhs_key, rhs_key)
    return chainer.chain()  # type: ignore


def left_join(
    lhs: Iterable[Left],
    rhs: Iterable[Right],
    lhs_key: Callable[[Left], Key] = DEFAULT_KEY,
    rhs_key: Callable[[Right], Key] = DEFAULT_KEY,
) -> Iterator[Tuple[Iterator[Left], Iterator[Right]]]:
    """
    Join two iterables like SQL left joining.
    While SQL left joining returns all the combinations,
    this returns the pair of items.

    The arguments are very similar to `relate_one_to_many`.
    See `relate_one_to_many` doc for more information.

    This function is equivalent to below:
    - Groups `lhs` by `lhs_key`.
    - Run `relate_one_to_many` with that group as `lhs` and `rhs`.

    Here are some normal cases.
    Note that the `right` can empty, like SQL left joining.
    >>> lhs = [(1, 'a'), (1, 'b'), (2, 'c'), (4, 'd')]
    >>> rhs = [(1, 's'), (1, 't'), (3, 'u'), (4, 'v')]
    >>> relations = left_join(lhs, rhs)
    >>> for left, right in relations:
    ...     list(left), list(right)
    ([(1, 'a'), (1, 'b')], [(1, 's'), (1, 't')])
    ([(2, 'c')], [])
    ([(4, 'd')], [(4, 'v')])

    Custom keys are acceptable.
    >>> relations = left_join(
    ...     lhs, rhs,
    ...     lhs_key=lambda l: l[0] * 2,
    ...     rhs_key=lambda r: r[0] + 1)
    >>> for left, right in relations:
    ...     list(left), list(right)
    ([(1, 'a'), (1, 'b')], [(1, 's'), (1, 't')])
    ([(2, 'c')], [(3, 'u')])
    ([(4, 'd')], [])

    Here is a seminormal cases.
    When given empty `lhs`, returns the empty iterator.
    >>> relations = left_join([], [(1, 's')])
    >>> list(relations)
    []
    """

    lhs_groups = groupby(lhs, lhs_key)
    relations = relate_one_to_many(lhs_groups, rhs, FIRST_ITEM_KEY, rhs_key)
    return ((left, right) for (_, left), right in relations)


def outer_join(
    lhs: Iterable[Left],
    rhs: Iterable[Right],
    lhs_key: Callable[[Left], Key] = DEFAULT_KEY,
    rhs_key: Callable[[Right], Key] = DEFAULT_KEY,
) -> Iterator[Tuple[Iterator[Left], Iterator[Right]]]:
    """
    Join two iterables preserving all existing keys.
    In contrast to `left_join`, this preserve keys that are only in `rhs`.

    The arguments are very similar to `relate_one_to_many`.
    See `relate_one_to_many` doc for more information.

    Here are some normal cases.
    Note that all existing keys are covered and
    both `left` and `right` can be empty.
    >>> lhs = [(1, 'a'), (1, 'b'), (2, 'c'), (4, 'd')]
    >>> rhs = [(1, 's'), (1, 't'), (3, 'u'), (4, 'v')]
    >>> relations = outer_join(lhs, rhs)
    >>> for left, right in relations:
    ...     list(left), list(right)
    ([(1, 'a'), (1, 'b')], [(1, 's'), (1, 't')])
    ([(2, 'c')], [])
    ([], [(3, 'u')])
    ([(4, 'd')], [(4, 'v')])

    When given custom keys, then joins by them.
    >>> relations = relate_one_to_many(
    ...     lhs, rhs,
    ...     lhs_key=lambda l: l[0] * 2,
    ...     rhs_key=lambda r: r[0] + 1)
    >>> for left, right in relations:
    ...     left, list(right)
    ((1, 'a'), [(1, 's'), (1, 't')])
    ((1, 'b'), [])
    ((2, 'c'), [(3, 'u')])
    ((4, 'd'), [])

    When given long tail `lhs`, then returns the empty tail for the right.
    >>> relations = outer_join([(1, 'a'), (2, 'b')], [(1, 's')])
    >>> for left, right in relations:
    ...     list(left), list(right)
    ([(1, 'a')], [(1, 's')])
    ([(2, 'b')], [])

    When given long tail `rhs`, then returns the empty tail for the left.
    >>> relations = outer_join([(1, 'a')], [(1, 's'), (2, 't')])
    >>> for left, right in relations:
    ...     list(left), list(right)
    ([(1, 'a')], [(1, 's')])
    ([], [(2, 't')])

    Here are some seminormal cases.
    When given empty `lhs`,
    then returns an iterator that all left items are empty.
    >>> relations = outer_join([], [(1, 's')])
    >>> for left, right in relations:
    ...     list(left), list(right)
    ([], [(1, 's')])

    When given empty `rhs`,
    then returns an iterator that all right items are empty
    >>> relations = outer_join([(1, 'a')], [])
    >>> for left, right in relations:
    ...     list(left), list(right)
    ([(1, 'a')], [])
    """

    lhs_finder = _UnidirectionalFinder(lhs, lhs_key)
    rhs_finder = _UnidirectionalFinder(rhs, rhs_key)

    while lhs_finder.has_items:
        if not rhs_finder.has_items:
            yield next(lhs_finder), iter(_EMPTY_ITERABLE)
            continue

        key_curr = min(lhs_finder.current_key(), rhs_finder.current_key())
        yield lhs_finder.find(key_curr), rhs_finder.find(key_curr)

    while rhs_finder.has_items:
        yield iter(_EMPTY_ITERABLE), next(rhs_finder)


def inner_join(
    lhs: Iterable[Left],
    rhs: Iterable[Right],
    lhs_key: Callable[[Left], Key] = DEFAULT_KEY,
    rhs_key: Callable[[Right], Key] = DEFAULT_KEY,
) -> Iterator[Tuple[Iterator[Left], Iterator[Right]]]:
    """
    Join two iterables like SQL inner joining.

    This function's behavior is very similar to `left_join`.
    See `left_join` doc first.

    In contrast to `left_join`, `right` cannot be empty,
    like SQL inner joining.

    Here are some normal cases.
    >>> lhs = [(1, 'a'), (1, 'b'), (2, 'c'), (4, 'd')]
    >>> rhs = [(1, 's'), (1, 't'), (3, 'u'), (4, 'v')]
    >>> relations = inner_join(lhs, rhs)
    >>> for left, right in relations:
    ...     list(left), list(right)
    ([(1, 'a'), (1, 'b')], [(1, 's'), (1, 't')])
    ([(4, 'd')], [(4, 'v')])

    Custom keys are acceptable.
    >>> relations = inner_join(
    ...     lhs, rhs,
    ...     lhs_key=lambda l: l[0] * 2,
    ...     rhs_key=lambda r: r[0] + 1)
    >>> for left, right in relations:
    ...     list(left), list(right)
    ([(1, 'a'), (1, 'b')], [(1, 's'), (1, 't')])
    ([(2, 'c')], [(3, 'u')])

    Here is a seminormal cases.
    When given empty `lhs`, returns the empty iterator.
    >>> relations = inner_join([], [(1, 's')])
    >>> list(relations)
    []
    """

    left_joined = left_join(lhs, rhs, lhs_key, rhs_key)
    relations = ((left, _Peekable(right)) for left, right in left_joined)
    return ((left, right) for left, right in relations if right)
