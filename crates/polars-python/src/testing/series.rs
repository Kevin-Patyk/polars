use polars_testing::asserts::{SeriesEqualOptions, assert_series_equal};
use polars_testing::compare::{CompareSeriesOptions, compare_series};
use pyo3::prelude::*;

use crate::PySeries;
use crate::error::PyPolarsErr;

#[pyfunction]
#[pyo3(signature = (left, right, *, check_dtypes, check_names, check_order, check_exact, rel_tol, abs_tol, categorical_as_str))]
pub fn assert_series_equal_py(
    left: &PySeries,
    right: &PySeries,
    check_dtypes: bool,
    check_names: bool,
    check_order: bool,
    check_exact: bool,
    rel_tol: f64,
    abs_tol: f64,
    categorical_as_str: bool,
) -> PyResult<()> {
    let left_series = &left.series.read();
    let right_series = &right.series.read();

    let options = SeriesEqualOptions {
        check_dtypes,
        check_names,
        check_order,
        check_exact,
        rel_tol,
        abs_tol,
        categorical_as_str,
    };

    assert_series_equal(left_series, right_series, options).map_err(|e| PyPolarsErr::from(e).into())
}

#[pyfunction]
#[pyo3(signature = (left, right, *, sort, keep_shape, keep_equal))]
pub fn compare_series_py(
    left: &PySeries,
    right: &PySeries,
    sort: bool,
    keep_shape: bool,
    keep_equal: bool,
) -> PyResult<crate::PyDataFrame> {
    let left_series = &left.series.read();
    let right_series = &right.series.read();

    let options = CompareSeriesOptions {
        sort,
        keep_shape,
        keep_equal,
    };

    compare_series(left_series, right_series, options)
        .map(|df| df.into())
        .map_err(|e| PyPolarsErr::from(e).into())
}
