#![allow(dead_code)]
use std::ops::{Not, BitAnd, BitOr};

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
pub fn sort_series(s: &Series) -> PolarsResult<Series> {
    let sorted_series = match s.sort(SortOptions::default()) {
        Ok(sorted) => sorted,
        Err(_) => return Err(polars_err!(op = "sort", s.dtype())),
    };

    Ok(sorted_series)
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
    unequal: &ChunkedArray<BooleanType>,
    rtol: f64,
    atol: f64,
) -> PolarsResult<()> {
    // Keeps only elements in each Series where `unequal_bool` is `true`
    let left_unequal = left.filter(unequal)?;
    let right_unequal = right.filter(unequal)?;

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

/// Assert that the values in both series are equal.
pub fn assert_series_values_equal(
    left: &Series,
    right: &Series,
    check_order: bool,
    check_exact: bool,
    rtol: f64,
    atol: f64,
    categorical_as_str: bool,
) -> PolarsResult<()> {

    // Handle categoricals
    let (left, right) = if categorical_as_str {
        (categorical_series_to_string(left), categorical_series_to_string(right))
    } else {
        (left.clone(), right.clone())
    };

    // Sort the series 
    let (left, right) = if !check_order {
        (sort_series(&left)?, sort_series(&right)?)
    } else {
        (left.clone(), right.clone())
    };

    // Determine unequal elements while also considering missing values
    // There is no direct `ne_missing()` function in Rust like there is in Python for Series
    let (left_null, right_null) = (left.is_null(), right.is_null());
    let both_null = left_null.bitand(right_null);

    let values_equal = left.equal(&right)?;
    let combined_equal = values_equal.bitor(both_null);
    let unequal = combined_equal.not();

    // Checked nested dypes in separate function
    if comparing_nested_floats(left.dtype(), right.dtype()) {
        let filtered_left = left.filter(&unequal)?;
        let filtered_right = right.filter(&unequal)?;

        match assert_series_nested_values_equal(
            &filtered_left, 
            &filtered_right, 
            check_exact, 
            rtol, 
            atol, 
            categorical_as_str,
        ) {
            Ok(_) => {
                return Ok(());
            },
            Err(_) => {
                return Err(polars_err!(
                    assertion_error = "Series",
                    "nested value mismatch",
                    left, 
                    right
                ));
            }
        }
    }

    // If no differences are found during exact checking, we are finished
    if !unequal.any() {
        return Ok(())
    }

    // Only do inexact checking for float types
    if check_exact || !left.dtype().is_float() || !right.dtype().is_float() {
        return Err(polars_err!(
            assertion_error = "Series",
            "exact value mismatch",
            left,
            right
        ))
    }

    assert_series_null_values_match(&left, &right)?;
    assert_series_nan_values_match(&left, &right)?;
    assert_series_values_within_tolerance(&left, &right, &unequal, rtol, atol)?;

    Ok(())
}

/// Checks whether the nested values in two Series are equal.
pub fn assert_series_nested_values_equal(
    left: &Series,
    right: &Series,
    check_exact: bool,
    rtol: f64,
    atol: f64,
    categorical_as_str: bool,
) -> PolarsResult<()> {
    
    // Compare nested lists element-wise
    if comparing_lists(left.dtype(), right.dtype()) {
        let zipped = left.iter().zip(right.iter());

        for (s1, s2) in zipped {
            if s1.is_null() || s2.is_null() {
                return Err(polars_err!(
                    assertion_error = "Series",
                    "values do not match",
                    s1,
                    s2
                ));
            } else {
                // Convert `AnyValue` to Series
                let s1_series = Series::new("s1".into(), vec![s1.clone()]);
                let s2_series = Series::new("s2".into(), vec![s2.clone()]);

                // Check the result and only return on error
                match assert_series_values_equal(
                    &s1_series,
                    &s2_series,
                    true,
                    check_exact,
                    rtol,
                    atol,
                    categorical_as_str,
                ) {
                    Ok(_) => continue,
                    Err(e) => return Err(e),
                }
            }
        }
    } else {
        // Unnest Structs as Series for comparison
        let ls = left.struct_()?.clone().unnest();
        let rs = right.struct_()?.clone().unnest();

        // Get the columns from each DataFrame
        let ls_cols = ls.get_columns();
        let rs_cols = rs.get_columns();

        // Run `for` loop over paired-columns while converting them into Series
        for (s1, s2) in ls_cols.iter().zip(rs_cols.iter()) {
            // Check the result and only return on error
            match assert_series_values_equal(
                &s1.as_series().unwrap(),
                &s2.as_series().unwrap(),
                true,
                check_exact,
                rtol,
                atol,
                categorical_as_str
            ) {
                Ok(_) => continue,
                Err(e) => return Err(e),
            }
        }
    }

    Ok(())
}
