#[macro_export]
macro_rules! assert_dataframe_equal {
    ($left:expr, $right:expr $(, $options:expr)?) => {
        #[allow(unused_assignments)]
        #[allow(unused_mut)]
        let mut options = $crate::asserts::DataFrameEqualOptions::default();
        $(options = $options;)?

        match $crate::asserts::assert_dataframe_equal($left, $right, options) {
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
        let options = crate::asserts::DataFrameEqualOptions::default();

        assert!(options.check_row_order);
        assert!(options.check_column_order);
        assert!(options.check_dtypes);
        assert!(!options.check_exact);
        assert_eq!(options.rtol, 1e-5);
        assert_eq!(options.atol, 1e-8);
        assert!(!options.categorical_as_str);
    }
}
