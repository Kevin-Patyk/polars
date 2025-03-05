#![allow(dead_code)]
use std::ops::Not;

use polars_core::datatypes::unpack_dtypes;
use polars_core::prelude::*;
use polars_ops::series::abs;

/// Change a (possibly nested) Categorical data type to a String data type.
pub fn categorical_dtype_to_string_dtype(dtype: &DataType) -> DataType {
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
pub fn categorical_series_to_string(s: &Series) -> Series {
    let dtype = s.dtype();
    let noncat_dtype = categorical_dtype_to_string_dtype(dtype);

    if *dtype != noncat_dtype {
        s.cast(&noncat_dtype).unwrap()
    } else {
        s.clone()
    }
}

/// Checks if two data types are float.
pub fn comparing_floats(left: &DataType, right: &DataType) -> bool {
    left.is_float() && right.is_float()
}

/// Checks if two data types are either Lists or Arrays.
pub fn comparing_lists(left: &DataType, right: &DataType) -> bool {
    matches!(left, DataType::List(_) | DataType::Array(_, _))
        && matches!(right, DataType::List(_) | DataType::Array(_, _))
}

/// Checks if two data types are Structs.
pub fn comparing_structs(left: &DataType, right: &DataType) -> bool {
    left.is_struct() && right.is_struct()
}

/// Checks if compound DataTypes contain any float values.
pub fn comparing_nested_floats(left: &DataType, right: &DataType) -> bool {
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
pub fn sort_series(left: &Series, right: &Series) -> PolarsResult<(Series, Series)> {
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
pub fn assert_series_null_values_match(left: &Series, right: &Series) -> PolarsResult<()> {
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
pub fn assert_series_nan_values_match(left: &Series, right: &Series) -> PolarsResult<()> {
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

/// Checks whether values in two Series are within a given tolerance.
pub fn assert_series_values_within_tolerance(
    left: &Series,
    right: &Series,
    unequal: &Series,
    rtol: f64,
    atol: f64,
) -> PolarsResult<()> {
    // Converts the `unequal` Series into a boolean mask
    // `true` where values are unequal and `false` if they are equal
    let unequal_bool = unequal.bool()?;

    // Keeps only elements in each Series where `unequal_bool` is `true`
    let left_unequal = left.filter(unequal_bool)?;
    let right_unequal = right.filter(unequal_bool)?;

    // Computes the element-wise absolute difference between  the two Series
    let difference = (&left_unequal - &right_unequal)?;
    let abs_difference = abs(&difference)?;

    // Computes the absolute values for `right_unequal` Series
    let right_abs = abs(&right_unequal)?;

    // Creating Series objects for `rtol` and `atol` to be used in calculations
    let rtol_series = Series::new("rtol".into(), &[rtol]);
    let atol_series = Series::new("atol".into(), &[atol]);

    // Computes the tolerance
    let rtol_part = (&right_abs * &rtol_series)?;
    let tolerance = (&rtol_part + &atol_series)?;

    // Check if the differences are within tolerance
    let finite_mask = right_unequal.is_finite()?;
    let diff_within_tol = abs_difference.lt_eq(&tolerance)?;
    let equal_values = left_unequal.equal(&right_unequal)?;

    // Combines the three previous conditions and creates a final mask
    let within_tolerance = (diff_within_tol & finite_mask) | equal_values;

    if within_tolerance.all() {
        return Ok(());
    } else {
        // Finds indices where the tolerance is exceeded
        let exceeded_indices = within_tolerance.not();
        let problematic_left = left_unequal.filter(&exceeded_indices)?;
        let problematic_right = right_unequal.filter(&exceeded_indices)?;

        return Err(polars_err!(
            assertion_error = "Series",
            "values not within tolerance",
            problematic_left,
            problematic_right
        ));
    }
}

#[allow(unused_variables)]
/// Checks whether the nested values in two Series are equal.
pub fn assert_series_nested_values_equal(
    left: &Series,
    right: &Series,
    check_exact: bool,
    rtol: f64,
    atol: f64,
    categorical_as_str: bool,
) -> PolarsResult<()> {
    todo!()
}
