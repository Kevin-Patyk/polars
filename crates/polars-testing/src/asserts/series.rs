#[allow(unused_imports)]
use super::utils::*;

pub struct SeriesEqualOptions {
    pub check_dtypes: bool,
    pub check_names: bool,
    pub check_order: bool,
    pub check_exact: bool,
    pub rtol: f64,
    pub atol: f64,
    pub categorical_as_str: bool,
}

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
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_check_dtypes(mut self, value: bool) -> Self {
        self.check_dtypes = value;
        self
    }

    pub fn with_check_names(mut self, value: bool) -> Self {
        self.check_names = value;
        self
    }

    pub fn with_check_order(mut self, value: bool) -> Self {
        self.check_order = value;
        self
    }

    pub fn with_check_exact(mut self, value: bool) -> Self {
        self.check_exact = value;
        self
    }

    pub fn with_rtol(mut self, value: f64) -> Self {
        self.rtol = value;
        self
    }

    pub fn with_atol(mut self, value: f64) -> Self {
        self.atol = value;
        self
    }

    pub fn with_categorical_as_str(mut self, value: bool) -> Self {
        self.categorical_as_str = value;
        self
    }
}
