#[allow(unused_imports)]
use super::utils;

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
    /// Create a new instance of SeriesEqualOptions with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder method to set `check_dtypes`.
    pub fn with_check_dtypes(mut self, value: bool) -> Self {
        self.check_dtypes = value;
        self
    }

    /// Builder method to set `check_names`.
    pub fn with_check_names(mut self, value: bool) -> Self {
        self.check_names = value;
        self
    }

    /// Builder method to set `check_order`.
    pub fn with_check_order(mut self, value: bool) -> Self {
        self.check_order = value;
        self
    }

    /// Builder method to set `check_exact`.
    pub fn with_check_exact(mut self, value: bool) -> Self {
        self.check_exact = value;
        self
    }

    /// Builder method  to set `rtol`.
    pub fn with_rtol(mut self, value: f64) -> Self {
        self.rtol = value;
        self
    }

    /// Builder method  to set `atol`.
    pub fn with_atol(mut self, value: f64) -> Self {
        self.atol = value;
        self
    }

    /// Builder method to set `categorical_as_str`.
    pub fn with_categorical_as_str(mut self, value: bool) -> Self {
        self.categorical_as_str = value;
        self
    }
}

pub fn assert_series_equal() {
    todo!()
}