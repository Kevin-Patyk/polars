#[macro_export]
macro_rules! assert_series_equal {
    // **Note:** Allows the user to use the macro with custom options.
    (custom = $left:expr, $right:expr, $options:expr) => {
        match $crate::asserts::assert_series_equal($left, $right, $options) {
            Ok(_) => {},
            Err(e) => panic!("{}", e),
        }
    };
    // **Note:** Allows the user to use the macro with default options enabled.
    (default = $left:expr, $right:expr) => {
        match $crate::asserts::assert_series_equal(
            $left,
            $right,
            $crate::asserts::SeriesEqualOptions::default(),
        ) {
            Ok(_) => {},
            Err(e) => panic!("{}", e),
        }
    };
}

#[cfg(test)]
mod tests {
    use polars_core::prelude::*;

    // Testing default struct implementation
    #[test]
    fn test_series_equal_options() {
        let options = crate::asserts::SeriesEqualOptions::default();

        assert!(options.check_dtypes);
        assert!(options.check_names);
        assert!(options.check_order);
        assert!(options.check_exact);
        assert_eq!(options.rtol, 1e-5);
        assert_eq!(options.atol, 1e-8);
        assert!(!options.categorical_as_str);
    }

    // Testing basic operations
    #[test]
    #[should_panic(expected = "length mismatch")]
    fn test_series_length_mismatch() {
        let s1 = Series::new("".into(), &[1, 2]);
        let s2 = Series::new("".into(), &[1, 2, 3]);

        assert_series_equal!(default = &s1, &s2)
    }

    #[test]
    #[should_panic(expected = "name mismatch")]
    fn test_series_names_mismatch() {
        let s1 = Series::new("s1".into(), &[1, 2, 3]);
        let s2 = Series::new("s2".into(), &[1, 2, 3]);

        assert_series_equal!(default = &s1, &s2)
    }

    #[test]
    fn test_series_check_names_false() {
        let s1 = Series::new("s1".into(), &[1, 2, 3]);
        let s2 = Series::new("s2".into(), &[1, 2, 3]);

        let options = crate::asserts::SeriesEqualOptions::default().with_check_names(false);

        assert_series_equal!(custom = &s1, &s2, options);
    }

    #[test]
    #[should_panic(expected = "data type mismatch")]
    fn test_series_dtype_mismatch() {
        let s1 = Series::new("".into(), &[1, 2, 3]);
        let s2 = Series::new("".into(), &["1", "2", "3"]);

        assert_series_equal!(default = &s1, &s2)
    }

    #[test]
    fn test_series_check_dtypes_false() {
        let s1 = Series::new("s1".into(), &[1, 2, 3]);
        let s2 = Series::new("s1".into(), &[1.0, 2.0, 3.0]);

        let options = crate::asserts::SeriesEqualOptions::default().with_check_dtypes(false);

        assert_series_equal!(custom = &s1, &s2, options);
    }

    #[test]
    #[should_panic(expected = "exact value mismatch")]
    fn test_assert_series_value_mismatch() {
        let s1 = Series::new("".into(), &[1, 2, 3]);
        let s2 = Series::new("".into(), &[2, 3, 4]);

        assert_series_equal!(default = &s1, &s2);
    }

    #[test]
    fn test_assert_series_values_match_int() {
        let s1 = Series::new("".into(), &[1, 2, 3]);
        let s2 = Series::new("".into(), &[1, 2, 3]);

        assert_series_equal!(default = &s1, &s2);
    }

    #[test]
    fn test_assert_series_values_match_str() {
        let s1 = Series::new("".into(), &["foo", "bar"]);
        let s2 = Series::new("".into(), &["foo", "bar"]);

        assert_series_equal!(default = &s1, &s2);
    }

    #[test]
    fn test_assert_series_empty_equal() {
        let s1 = Series::default();
        let s2 = Series::default();

        assert_series_equal!(default = &s1, &s2);
    }

    // Testing with nulls and nans
    #[test]
    fn test_assert_series_nan_equal() {
        let s1 = Series::new("".into(), &[f64::NAN, f64::NAN, f64::NAN]);
        let s2 = Series::new("".into(), &[f64::NAN, f64::NAN, f64::NAN]);

        assert_series_equal!(default = &s1, &s2);
    }

    #[test]
    fn test_assert_series_null_equal() {
        let s1 = Series::new("".into(), &[None::<f64>, None::<f64>, None::<f64>]);
        let s2 = Series::new("".into(), &[None::<f64>, None::<f64>, None::<f64>]);

        assert_series_equal!(default = &s1, &s2);
    }

    // Testing sorting operations
    #[test]
    #[should_panic(expected = "exact value mismatch")]
    fn test_assert_series_sorting_unequal() {
        let s1 = Series::new("".into(), &[Some(1), Some(2), Some(3), None::<i32>]);
        let s2 = Series::new("".into(), &[Some(2), None::<i32>, Some(3), Some(1)]);

        let options = crate::asserts::SeriesEqualOptions::default();

        assert_series_equal!(custom = &s1, &s2, options);
    }

    #[test]
    fn test_assert_series_sorting_equal() {
        let s1 = Series::new("".into(), &[Some(1), Some(2), Some(3), None::<i32>]);
        let s2 = Series::new("".into(), &[Some(2), None::<i32>, Some(3), Some(1)]);

        let options = crate::asserts::SeriesEqualOptions::default().with_check_order(false);

        assert_series_equal!(custom = &s1, &s2, options);
    }
}
