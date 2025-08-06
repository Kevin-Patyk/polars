use std::ops::RangeInclusive;
use std::rc::Rc;

use arrow::bitmap::bitmask::nth_set_bit_u32;
use polars_utils::pl_str::PlSmallStr;
use proptest::prelude::*;

use super::{any_value, TimeUnit};

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct AnyValueArbitrarySelection: u32 {
        const NULL = 1;
        const BOOLEAN = 1 << 1;
        const STRING = 1 << 2;
        const UINT8 = 1 << 3;
        const UINT16 = 1 << 4;
        const UINT32 = 1 << 5;
        const UINT64 = 1 << 6;
        const INT8 = 1 << 7;
        const INT16 = 1 << 8;
        const INT32 = 1 << 9;
        const INT64 = 1 << 10;
        const INT128 = 1 << 11;
        const FLOAT32 = 1 << 12;
        const FLOAT64 = 1 << 13;
        const DATE = 1 << 14;
        const TIME = 1 << 15;
        const STRING_OWNED = 1 << 16;
        const BINARY = 1 << 17;
        const OBJECT = 1 << 18;

        const DATETIME = 1 << 19;
        const DATETIME_OWNED = 1 << 20;
        const DURATION = 1 << 21;
        const DECIMAL = 1 << 22;
        const BINARY_OWNED = 1 << 23;
        const CATEGORICAL = 1 << 24;
        const CATEGORICAL_OWNED = 1 << 25;
        const ENUM = 1 << 26;
        const ENUM_OWNED = 1 << 27;
        const OBJECTOWNED = 1 << 28;

        const LIST = 1 << 29;
        const ARRAY = 1 << 30;
        const STRUCT = 1 << 31;
        const STRUCT_OWNED = << 32; 
    }
}

impl AnyValueArbitrarySelection {
    pub fn nested() -> Self {
        Self::LIST | Self::ARRAY | Self::STRUCT | Self::STRUCT_OWNED
    }
}

#[derive(Clone)]
pub struct AnyValueArbitraryOptions {
    // TODO: Add fields later
}

impl Default for AnyValueArbitraryOptions {
    fn default() -> Self {
        Self {
            // TODO: Add fields later
        }
    }
}

