#![allow(dead_code)]
use crate::datatypes::unpack_dtypes;
use crate::prelude::*;

pub struct SeriesEqualOptions {
    /// Requires data types to match.
    pub check_dtypes: bool,

    /// Requires names to match.
    pub check_names: bool,

    /// Requires element to appear in the same order.
    pub check_order: bool,

    /// Requires float values to match exactly.
    /// If set to `false`, values are considered equal within tolerance
    /// of each other (see `rtol` and `atol`).
    /// Only affects columns with a Float data type
    pub check_exact: bool,

    /// Relative tolerance for inexact checking, given as a fraction of the values.
    pub rtol: f64,

    // Absolute tolerance for inexact checking.
    pub atol: f64,

    // Cast categorical columns to string before comparing.
    // Enabling this helps compare columns that do not share the same string cache.
    pub categorical_as_str: bool,
}

/// Default implementation for SeriesEqualOptions
impl Default for SeriesEqualOptions {
    fn default() -> Self {
        Self {
            check_dtypes: true,
            check_names: true,
            check_order: true,
            check_exact: true,
            rtol: 1e-5,
            atol: 1e-8,
            categorical_as_str: false,
        }
    }
}

impl SeriesEqualOptions {
    /// Create a new instance of SeriesEqualOptions with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder method to set `check_dtypes`
    pub fn with_check_dtypes(mut self, value: bool) -> Self {
        self.check_dtypes = value;
        self
    }

    /// Builder method to set `check_names`
    pub fn with_check_names(mut self, value: bool) -> Self {
        self.check_names = value;
        self
    }

    /// Builder method to set `check_order`
    pub fn with_check_order(mut self, value: bool) -> Self {
        self.check_order = value;
        self
    }

    /// Builder method to set `check_exact`
    pub fn with_check_exact(mut self, value: bool) -> Self {
        self.check_exact = value;
        self
    }

    /// Builder method  to set `rtol`
    pub fn with_rtol(mut self, value: f64) -> Self {
        self.rtol = value;
        self
    }

    /// Builder method  to set `atol`
    pub fn with_atol(mut self, value: f64) -> Self {
        self.atol = value;
        self
    }

    /// Builder method to set `categorical_as_str`
    pub fn with_categorical_as_str(mut self, value: bool) -> Self {
        self.categorical_as_str = value;
        self
    }
}

/// Change a (possibly nested) Categorical data type to a String data type.
fn categorical_dtype_to_string_dtype(dtype: &DataType) -> DataType {
    match dtype {
        DataType::Categorical(..) => DataType::String,
        DataType::List(inner) => {
            let inner_cast = categorical_dtype_to_string_dtype(inner);
            DataType::List(Box::new(inner_cast))
        },
        DataType::Array(inner, size) => {
            let inner_cast = categorical_dtype_to_string_dtype(inner);
            DataType::Array(Box::new(inner_cast), *size)
        },
        DataType::Struct(fields) => {
            let transformed_fields = fields
                .iter()
                .map(|field| {
                    Field::new(
                        field.name().clone(),
                        categorical_dtype_to_string_dtype(&field.dtype()),
                    )
                })
                .collect::<Vec<Field>>();

            DataType::Struct(transformed_fields)
        },
        _ => dtype.clone(),
    }
}

/// Change a (possibly nested) Categorical Series into a String Series.
fn categorical_series_to_string(s: &Series) -> Series {
    let dtype = s.dtype();
    let noncat_dtype = categorical_dtype_to_string_dtype(dtype);

    if *dtype != noncat_dtype {
        s.cast(&noncat_dtype).unwrap()
    } else {
        s.clone()
    }
}

/// Checks if two data types are float.
fn comparing_floats(left: &DataType, right: &DataType) -> bool {
    left.is_float() && right.is_float()
}

/// Checks if two data types are either Lists or Arrays.
fn comparing_lists(left: &DataType, right: &DataType) -> bool {
    matches!(left, DataType::List(_) | DataType::Array(_, _))
        && matches!(right, DataType::List(_) | DataType::Array(_, _))
}

/// Checks if two data types are Structs.
fn comparing_structs(left: &DataType, right: &DataType) -> bool {
    left.is_struct() && right.is_struct()
}

/// Checks if compound DataTypes contain any float values.
fn comparing_nested_floats(left: &DataType, right: &DataType) -> bool {
    if !comparing_lists(left, right) && !comparing_structs(left, right) {
        return false;
    }

    let left_dtypes = unpack_dtypes(left, false);
    let right_dtypes = unpack_dtypes(right, false);

    let left_has_floats = left_dtypes.iter().any(|dt| dt.is_float());
    let right_has_floats = right_dtypes.iter().any(|dt| dt.is_float());

    left_has_floats && right_has_floats
}

/// Sorts the left and right Series in ascending order.
fn sort_series(left: &Series, right: &Series) -> PolarsResult<(Series, Series)> {
    let sorted_left = match left.sort(SortOptions::default()) {
        Ok(sorted) => sorted,
        Err(_) => return Err(polars_err!(op = "sort", left.dtype())),
    };
    let sorted_right = match right.sort(SortOptions::default()) {
        Ok(sorted) => sorted,
        Err(_) => return Err(polars_err!(op = "sort", right.dtype())),
    };

    Ok((sorted_left, sorted_right))
}

/// Checks if two Series have the same amount of null values.
fn assert_series_null_values_match(left: &Series, right: &Series) -> PolarsResult<()> {
    let null_value_mismatch = left.is_null().not_equal(&right.is_null());
    if null_value_mismatch.any() {
        return Err(polars_err!(
            assertion_error = "Series",
            "null value mismatch",
            left.null_count(),
            right.null_count()
        ));
    }

    Ok(())
}

/// Checks if two Series have the same amount of NaN values.
fn assert_series_nan_values_match(left: &Series, right: &Series) -> PolarsResult<()> {
    if !comparing_floats(&left.dtype(), &right.dtype()) {
        return Ok(());
    }
    let left_nan = left.is_nan()?;
    let right_nan = right.is_nan()?;

    let nan_value_mismatch = left_nan.not_equal(&right_nan);

    let left_nan_count = left_nan.sum().unwrap_or(0);
    let right_nan_count = right_nan.sum().unwrap_or(0);

    if nan_value_mismatch.any() {
        return Err(polars_err!(
            assertion_error = "Series",
            "nan value mismatch",
            left_nan_count,
            right_nan_count
        ));
    }

    Ok(())
}

#[allow(unused_variables)]
fn assert_series_values_within_tolerance(
    left: &Series,
    right: &Series,
    unequal: &Series,
    rtol: f64,
    atol: f64,
) -> PolarsResult<()> {

    // let left_unequal = left.filter(unequal)?;
    // let right_unequal = right.filter(unequal)?;

    // let difference = (&left_unequal - &right_unequal)?;
    // let abs_difference = abs(&difference)?;

    // let right_abs = abs(&right_unequal)?;
    // let rtol_series = Series::new("rtol".into(), &[rtol]);
    // let atol_series = Series::new("atol".into(), &[atol]);
    // let tolerance = (&right_abs * &rtol_series)?.add(&atol_series)?;

    todo!()
}
