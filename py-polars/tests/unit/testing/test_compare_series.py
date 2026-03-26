import pytest

import polars as pl
from polars.testing import compare_series


def test_compare_series_different_lengths() -> None:
    s1 = pl.Series("a", [1, 2, 3])
    s2 = pl.Series("a", [1, 2])
    with pytest.raises(Exception, match="same length"):
        compare_series(s1, s2)


def test_compare_series_different_names() -> None:
    s1 = pl.Series("a", [1, 2, 3])
    s2 = pl.Series("b", [1, 2, 3])
    with pytest.raises(Exception, match="different names"):
        compare_series(s1, s2)


def test_compare_series_different_dtypes() -> None:
    s1 = pl.Series("a", [1, 2, 3])
    s2 = pl.Series("a", [1.0, 2.0, 3.0])
    with pytest.raises(Exception, match="different dtypes"):
        compare_series(s1, s2)


def test_compare_series_both_empty() -> None:
    s1 = pl.Series("a", [], dtype=pl.Int64)
    s2 = pl.Series("a", [], dtype=pl.Int64)
    result = compare_series(s1, s2)
    assert result.is_empty()
    assert result.columns == ["left", "right"]


def test_compare_series_both_all_null() -> None:
    s1 = pl.Series("a", [None, None, None], dtype=pl.Int64)
    s2 = pl.Series("a", [None, None, None], dtype=pl.Int64)
    result = compare_series(s1, s2)
    assert result.is_empty()
    assert result.columns == ["left", "right"]


def test_compare_series_no_differences() -> None:
    s1 = pl.Series("a", [1, 2, 3])
    s2 = pl.Series("a", [1, 2, 3])
    result = compare_series(s1, s2)
    assert result.is_empty()


def test_compare_series_default() -> None:
    s1 = pl.Series("a", [1, 2, 3, 4])
    s2 = pl.Series("a", [1, 5, 3, 6])
    result = compare_series(s1, s2)
    assert result["left"].to_list() == [2, 4]
    assert result["right"].to_list() == [5, 6]


def test_compare_series_keep_shape_no_keep_equal() -> None:
    s1 = pl.Series("a", [1, 2, 3, 4])
    s2 = pl.Series("a", [1, 5, 3, 6])
    result = compare_series(s1, s2, keep_shape=True)
    assert result["left"].to_list() == [None, 2, None, 4]
    assert result["right"].to_list() == [None, 5, None, 6]


def test_compare_series_keep_shape_keep_equal() -> None:
    s1 = pl.Series("a", [1, 2, 3, 4])
    s2 = pl.Series("a", [1, 5, 3, 6])
    result = compare_series(s1, s2, keep_shape=True, keep_equal=True)
    assert result["left"].to_list() == [1, None, 3, None]
    assert result["right"].to_list() == [1, None, 3, None]


def test_compare_series_no_keep_shape_keep_equal() -> None:
    s1 = pl.Series("a", [1, 2, 3, 4])
    s2 = pl.Series("a", [1, 5, 3, 6])
    result = compare_series(s1, s2, keep_equal=True)
    assert result["left"].to_list() == [1, 3]
    assert result["right"].to_list() == [1, 3]


def test_compare_series_sort() -> None:
    s1 = pl.Series("a", [3, 1, 2])
    s2 = pl.Series("a", [2, 1, 3])
    result = compare_series(s1, s2, sort=True)
    assert result.is_empty()


def test_compare_series_no_sort_produces_differences() -> None:
    s1 = pl.Series("a", [3, 1, 2])
    s2 = pl.Series("a", [2, 1, 3])
    result = compare_series(s1, s2, sort=False)
    assert not result.is_empty()


def test_compare_series_with_nulls() -> None:
    s1 = pl.Series("a", [1, None, 3])
    s2 = pl.Series("a", [1, None, 4])
    result = compare_series(s1, s2)
    assert result["left"].to_list() == [3]
    assert result["right"].to_list() == [4]


def test_compare_series_null_vs_value() -> None:
    s1 = pl.Series("a", [1, None, 3])
    s2 = pl.Series("a", [1, 2, 3])
    result = compare_series(s1, s2)
    assert result["left"].to_list() == [None]
    assert result["right"].to_list() == [2]
