from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

from polars._utils.wrap import wrap_df
from polars.series import Series
from polars.testing.asserts.utils import raise_assertion_error

if TYPE_CHECKING:
    from polars._plr import compare_series_py
    from polars.dataframe import DataFrame

with contextlib.suppress(ImportError):  # Module not available when building docs
    from polars._plr import compare_series_py


def _assert_correct_input_type(left: Any, right: Any) -> bool:
    __tracebackhide__ = True

    if not (isinstance(left, Series) and isinstance(right, Series)):
        raise_assertion_error(
            "inputs",
            "unexpected input types",
            type(left).__name__,
            type(right).__name__,
        )
    return True


def compare_series(
    left: Series,
    right: Series,
    *,
    sort: bool = False,
    keep_shape: bool = False,
    keep_equal: bool = False,
) -> DataFrame:
    """
    Compare two Series and return a DataFrame showing the differences.

    Parameters
    ----------
    left
        The first Series to compare.
    right
        The second Series to compare.
    sort
        Sort both Series before comparing. Useful if the Series are not sorted already.
    keep_shape
        If `True`, keep all rows all rows in the output,
        nulling out positions based on `keep_equal`.
        If `False`, only return rows where values differ or match,
        if `keep_equal=True`).
    keep_equal
        If `True`, keep rows where values are equal and null out differences.
        If `False`, keep rows where values differ and null out equal values.

    Returns
    -------
    DataFrame
        A DataFrame with two columns, `"left"` and `"right"`,
        containing the compared values.

    Examples
    --------
    >>> s1 = pl.Series("a", [1, 2, 3, 4])
    >>> s2 = pl.Series("a", [1, 2, 5, 4])
    >>> pl.testing.compare_series(s1, s2)
    shape: (1, 2)
    ┌──────┬───────┐
    │ left ┆ right │
    │ ---  ┆ ---   │
    │ i64  ┆ i64   │
    ╞══════╪═══════╡
    │ 3    ┆ 5     │
    └──────┴───────┘
    """
    _assert_correct_input_type(left, right)

    return wrap_df(
        compare_series_py(
            left._s, right._s, sort=sort, keep_shape=keep_shape, keep_equal=keep_equal
        )
    )
