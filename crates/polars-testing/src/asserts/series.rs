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
    use polars_core::{disable_string_cache, enable_string_cache, prelude::*};

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

    // Testing with basic parameters
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

    // Testing basic equality
    #[test]
    #[should_panic(expected = "exact value mismatch")]
    fn test_series_value_mismatch_int() {
        let s1 = Series::new("".into(), &[1, 2, 3]);
        let s2 = Series::new("".into(), &[2, 3, 4]);

        assert_series_equal!(default = &s1, &s2);
    }

    #[test]
    fn test_series_values_match_int() {
        let s1 = Series::new("".into(), &[1, 2, 3]);
        let s2 = Series::new("".into(), &[1, 2, 3]);

        assert_series_equal!(default = &s1, &s2);
    }

    #[test]
    #[should_panic(expected = "exact value mismatch")]
    fn test_series_value_mismatch_str() {
        let s1 = Series::new("".into(), &["foo", "bar"]);
        let s2 = Series::new("".into(), &["moo", "car"]);

        assert_series_equal!(default = &s1, &s2);
    }

    #[test]
    fn test_series_values_match_str() {
        let s1 = Series::new("".into(), &["foo", "bar"]);
        let s2 = Series::new("".into(), &["foo", "bar"]);

        assert_series_equal!(default = &s1, &s2);
    }

    #[test]
    #[should_panic(expected = "exact value mismatch")]
    fn test_series_values_mismatch_float() {
        let s1 = Series::new("".into(), &[1.1, 2.2, 3.3]);
        let s2 = Series::new("".into(), &[2.2, 3.3, 4.4]);

        assert_series_equal!(default = &s1, &s2);
    }

    #[test]
    fn test_series_values_match_float() {
        let s1 = Series::new("".into(), &[1.1, 2.2, 3.3]);
        let s2 = Series::new("".into(), &[1.1, 2.2, 3.3]);

        assert_series_equal!(default = &s1, &s2);
    }

    // Testing float value precision equality
    #[test]
    #[should_panic(expected = "values not within tolerance")]
    fn test_series_float_exceeded_tol() {
        let s1 = Series::new("".into(), &[1.0, 2.2, 3.3]);
        let s2 = Series::new("".into(), &[1.00012, 2.200025, 3.300035]);

        let options = crate::asserts::SeriesEqualOptions::default().with_check_exact(false);

        assert_series_equal!(custom = &s1, &s2, options);
    }

    #[test]
    fn test_series_float_within_tol() {   
        let s1 = Series::new("".into(), &[1.0, 2.0, 3.0]);
        let s2 = Series::new("".into(), &[1.000005, 2.000015, 3.000025]);

        let options = crate::asserts::SeriesEqualOptions::default().with_check_exact(false);

        assert_series_equal!(custom = &s1, &s2, options);
    }

    // Testing equality with special values
    #[test]
    fn test_series_empty_equal() {
        let s1 = Series::default();
        let s2 = Series::default();

        assert_series_equal!(default = &s1, &s2);
    }

    #[test]
    fn test_series_nan_equal() {
        let s1 = Series::new("".into(), &[f64::NAN, f64::NAN, f64::NAN]);
        let s2 = Series::new("".into(), &[f64::NAN, f64::NAN, f64::NAN]);

        assert_series_equal!(default = &s1, &s2);
    }

    #[test]
    fn test_series_null_equal() {
        let s1 = Series::new("".into(), &[None::<i32>, None::<i32>, None::<i32>]);
        let s2 = Series::new("".into(), &[None::<i32>, None::<i32>, None::<i32>]);

        assert_series_equal!(default = &s1, &s2);
    }

    #[test]
    fn test_series_infinite_equal() {
        let s1 = Series::new("".into(), &[f32::INFINITY, f32::INFINITY, f32::INFINITY]);
        let s2 = Series::new("".into(), &[f32::INFINITY, f32::INFINITY, f32::INFINITY]);

        assert_series_equal!(default = &s1, &s2);
    }

    // Testing null and nan counts for floats
    #[test]
    #[should_panic(expected = "null value mismatch")]
    fn test_series_check_exact_false_null() {
        let s1 = Series::new("".into(), &[Some(1.0), None::<f64>, Some(3.0)]);
        let s2 = Series::new("".into(), &[Some(1.0), Some(2.0), Some(3.0)]);

        let options = crate::asserts::SeriesEqualOptions::default().with_check_exact(false);
        
        assert_series_equal!(custom = &s1, &s2, options);
    }

    #[test]
    #[should_panic(expected = "nan value mismatch")]
    fn test_series_check_exact_false_nan() {
    let s1 = Series::new("".into(), &[1.0, f64::NAN, 3.0]);
    let s2 = Series::new("".into(), &[1.0, 2.0, 3.0]);

        let options = crate::asserts::SeriesEqualOptions::default().with_check_exact(false);
        
        assert_series_equal!(custom = &s1, &s2, options);
    }

    // Testing sorting operations
    #[test]
    #[should_panic(expected = "exact value mismatch")]
    fn test_series_sorting_unequal() {
        let s1 = Series::new("".into(), &[Some(1), Some(2), Some(3), None::<i32>]);
        let s2 = Series::new("".into(), &[Some(2), None::<i32>, Some(3), Some(1)]);

        let options = crate::asserts::SeriesEqualOptions::default();

        assert_series_equal!(custom = &s1, &s2, options);
    }

    #[test]
    fn test_series_sorting_equal() {
        let s1 = Series::new("".into(), &[Some(1), Some(2), Some(3), None::<i32>]);
        let s2 = Series::new("".into(), &[Some(2), None::<i32>, Some(3), Some(1)]);

        let options = crate::asserts::SeriesEqualOptions::default().with_check_order(false);

        assert_series_equal!(custom = &s1, &s2, options);
    }

    // Testing categorical operations
    #[test]
    #[should_panic(expected = "exact value mismatch")]
    fn test_series_categorical_mismatch() {

        enable_string_cache();

        let s1 = Series::new("".into(), &["apple", "banana", "cherry"])
        .cast(&DataType::Categorical(None, Default::default()))
        .unwrap();
        let s2 = Series::new("".into(), &["apple", "orange", "cherry"])
        .cast(&DataType::Categorical(None, Default::default()))
        .unwrap();

        assert_series_equal!(default = &s1, &s2);

        disable_string_cache();
    }

    #[test]
    fn test_series_categorical_match() {
        let s1 = Series::new("".into(), &["apple", "banana", "cherry"])
        .cast(&DataType::Categorical(None, Default::default()))
        .unwrap();
        let s2 = Series::new("".into(), &["apple", "banana", "cherry"])
        .cast(&DataType::Categorical(None, Default::default()))
        .unwrap();
        
        assert_series_equal!(default = &s1, &s2);
    }

    #[test]
    #[should_panic(expected = "exact value mismatch")]
    fn test_series_categorical_str_mismatch() {
        let s1 = Series::new("".into(), &["apple", "banana", "cherry"])
        .cast(&DataType::Categorical(None, Default::default()))
        .unwrap();
        let s2 = Series::new("".into(), &["apple", "orange", "cherry"])
        .cast(&DataType::Categorical(None, Default::default()))
        .unwrap();

        let options = crate::asserts::SeriesEqualOptions::default().with_categorical_as_str(true);

        assert_series_equal!(custom = &s1, &s2, options);
    }

    #[test]
    fn test_series_categorical_str_match() {
        let s1 = Series::new("".into(), &["apple", "banana", "cherry"])
        .cast(&DataType::Categorical(None, Default::default()))
        .unwrap();
        let s2 = Series::new("".into(), &["apple", "banana", "cherry"])
        .cast(&DataType::Categorical(None, Default::default()))
        .unwrap();
        
        let options = crate::asserts::SeriesEqualOptions::default().with_categorical_as_str(true);

        assert_series_equal!(custom = &s1, &s2, options);
    }

    // Testing equality of nested values
    #[test]
    #[should_panic(expected = "exact value mismatch")]
    fn test_series_nested_values_int_mismatch() {
        let s1 = Series::new(
            "".into(),
            &[
                [1, 2, 3].iter().collect::<Series>(),
                [4, 5, 6].iter().collect::<Series>(),
                [7, 8, 9].iter().collect::<Series>()
            ]
        );
        
        let s2 = Series::new(
            "".into(),
            &[
                [0, 2, 3].iter().collect::<Series>(),
                [4, 7, 6].iter().collect::<Series>(),
                [7, 8, 10].iter().collect::<Series>()
            ]
        );
    
        assert_series_equal!(default = &s1, &s2);
    }

    #[test]
    fn test_series_nested_values_int_match() {
        let s1 = Series::new(
            "".into(),
            &[
                [1, 2, 3].iter().collect::<Series>(),
                [4, 5, 6].iter().collect::<Series>(),
                [7, 8, 9].iter().collect::<Series>()
            ]
        );
        
        let s2 = Series::new(
            "".into(),
            &[
                [1, 2, 3].iter().collect::<Series>(),
                [4, 5, 6].iter().collect::<Series>(),
                [7, 8, 9].iter().collect::<Series>()
            ]
        );
    
        assert_series_equal!(default = &s1, &s2);
    }

    #[test]
    #[should_panic(expected = "nested value mismatch")]
    fn test_series_nested_values_float_mismatch() {
        let s1 = Series::new(
            "".into(),
            &[
                [1.1, 2.0, 3.0].iter().collect::<Series>(),
                [4.0, 5.0, 6.0].iter().collect::<Series>(),
                [7.0, 8.0, 9.0].iter().collect::<Series>()
            ]
        );
        
        let s2 = Series::new(
            "".into(),
            &[
                [0.5, 2.0, 3.0].iter().collect::<Series>(),
                [4.0, 7.5, 6.0].iter().collect::<Series>(),
                [7.0, 8.0, 10.2].iter().collect::<Series>()
            ]
        );
    
        assert_series_equal!(default = &s1, &s2);
    }
}
