use std::borrow::Cow;

use polars_core::prelude::*;

/// Options for comparing two Series.
#[derive(Default)]
pub struct CompareSeriesOptions {
    /// Sort both Series before comparing. Default is `false`.
    pub sort: bool,
    /// Keep all rows in the output, nulling out positions based on `keep_equal`. Default is `false`.
    pub keep_shape: bool,
    /// If `true`, keep equal values and null out differences. If `false`, keep differences and null out equal values. Default is `false`.
    pub keep_equal: bool,
}

/// Compare two Series and return a DataFrame showing the differences.
///
/// # Arguments
/// * `left` - The left Series to compare.
/// * `right` - The right Series to compare.
/// * `options` - [`CompareSeriesOptions`] controlling the comparison behavior.
///
/// # Returns
/// A [`DataFrame`] with two columns, `"left"` and `"right"`, containing the compared values.
/// By default, only rows where values differ are returned.
///
/// # Errors
/// Returns an error if:
/// * The Series have different lengths.
/// * The Series have different names.
/// * The Series have different dtypes.
///
/// # Example
/// ```rust
/// use polars_core::prelude::*;
/// use polars_testing::compare::{compare_series, CompareSeriesOptions};
///
/// let left = Series::new("a".into(), &[1, 2, 3]);
/// let right = Series::new("a".into(), &[1, 4, 3]);
///
/// let result = compare_series(&left, &right, CompareSeriesOptions::default()).unwrap();
/// // Returns a DataFrame with only row 1 where values differ: left=2, right=4
/// ```
pub fn compare_series(
    left: &Series,
    right: &Series,
    options: CompareSeriesOptions,
) -> PolarsResult<DataFrame> {
    if left.len() != right.len() {
        polars_bail!(
            ShapeMismatch: "Series must have the same length: {} and {}",
            left.len(),
            right.len(),
        )
    }

    if left.name() != right.name() {
        polars_bail!(
            InvalidOperation: "Series have different names: '{}' and '{}'",
            left.name(),
            right.name(),
        )
    }

    if left.dtype() != right.dtype() {
        polars_bail!(
            InvalidOperation: "Series have different dtypes: {} and {}",
            left.dtype(),
            right.dtype(),
        )
    }

    // Short-circuit to return an empty DataFrame (no differences in elements) if
    // both Series are all null or both series are completely empty
    if left.null_count() == left.len() && right.null_count() == right.len()
        || left.is_empty() && right.is_empty()
    {
        return DataFrame::new(
            0,
            vec![
                Series::new_empty("left".into(), left.dtype()).into(),
                Series::new_empty("right".into(), right.dtype()).into(),
            ],
        );
    }

    // Sort both Series using default options if the user decides to
    let (left, right) = if options.sort {
        (
            Cow::Owned(left.sort(SortOptions::default())?),
            Cow::Owned(right.sort(SortOptions::default())?),
        )
    } else {
        (Cow::Borrowed(left), Cow::Borrowed(right))
    };

    let null_series = Series::full_null("".into(), left.len(), left.dtype());

    // Build a mask that is true for positions we want to keep
    let mask = if options.keep_equal {
        left.as_ref().equal_missing(right.as_ref())?
    } else {
        !(left.as_ref().equal_missing(right.as_ref())?)
    };

    // Apply mask to either filter rows or null out positions based on keep_shape
    let (left_result, right_result) = if options.keep_shape {
        (
            left.as_ref()
                .zip_with_same_type(&mask, &null_series)?
                .with_name("left".into()),
            right
                .as_ref()
                .zip_with_same_type(&mask, &null_series)?
                .with_name("right".into()),
        )
    } else {
        (
            left.as_ref().filter(&mask)?.with_name("left".into()),
            right.as_ref().filter(&mask)?.with_name("right".into()),
        )
    };

    DataFrame::new(
        left_result.len(),
        vec![left_result.into(), right_result.into()],
    )
}

/// Compare two Series and return a [`polars_core::frame::DataFrame`] showing the differences.
///
/// See [`compare_series`] for full documentation and options.
///
/// # Example
///
/// ```
/// use polars_core::prelude::*;
/// use polars_testing::compare_series;
/// use polars_testing::compare::CompareSeriesOptions;
///
/// let left = Series::new("a".into(), &[1, 2, 3]);
/// let right = Series::new("a".into(), &[1, 4, 3]);
///
/// let result = compare_series!(&left, &right).unwrap();
/// ```
#[macro_export]
macro_rules! compare_series {
    ($left:expr, $right:expr $(, $options:expr)?) => {
        {
            #[allow(unused_assignments)]
            #[allow(unused_mut)]
            let mut options = $crate::compare::CompareSeriesOptions::default();
            $(options = $options;)?

            $crate::compare::compare_series($left, $right, options)
        }
    };
}
