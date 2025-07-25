#![allow(unsafe_op_in_unsafe_fn)]
use arrow::bitmap::{Bitmap, BitmapBuilder};
use arrow::buffer::Buffer;
use arrow::datatypes::ArrowDataType;
use arrow::offset::OffsetsBuffer;
use arrow::types::NativeType;
use polars_dtype::categorical::CatNative;

use self::encode::fixed_size;
use self::row::{RowEncodingCategoricalContext, RowEncodingOptions};
use self::variable::utf8::decode_str;
use super::*;
use crate::fixed::numeric::{FixedLengthEncoding, FromSlice};
use crate::fixed::{boolean, decimal, numeric};
use crate::variable::{binary, no_order, utf8};

/// Decode `rows` into a arrow format
/// # Safety
/// This will not do any bound checks. Caller must ensure the `rows` are valid
/// encodings.
pub unsafe fn decode_rows_from_binary<'a>(
    arr: &'a BinaryArray<i64>,
    opts: &[RowEncodingOptions],
    dicts: &[Option<RowEncodingContext>],
    dtypes: &[ArrowDataType],
    rows: &mut Vec<&'a [u8]>,
) -> Vec<ArrayRef> {
    assert_eq!(arr.null_count(), 0);
    rows.clear();
    rows.extend(arr.values_iter());
    decode_rows(rows, opts, dicts, dtypes)
}

/// Decode `rows` into a arrow format
/// # Safety
/// This will not do any bound checks. Caller must ensure the `rows` are valid
/// encodings.
pub unsafe fn decode_rows(
    // the rows will be updated while the data is decoded
    rows: &mut [&[u8]],
    opts: &[RowEncodingOptions],
    dicts: &[Option<RowEncodingContext>],
    dtypes: &[ArrowDataType],
) -> Vec<ArrayRef> {
    assert_eq!(opts.len(), dtypes.len());
    assert_eq!(dicts.len(), dtypes.len());

    dtypes
        .iter()
        .zip(opts)
        .zip(dicts)
        .map(|((dtype, opt), dict)| decode(rows, *opt, dict.as_ref(), dtype))
        .collect()
}

unsafe fn decode_validity(rows: &mut [&[u8]], opt: RowEncodingOptions) -> Option<Bitmap> {
    // 2 loop system to avoid the overhead of allocating the bitmap if all the elements are valid.

    let null_sentinel = opt.null_sentinel();
    let first_null = (0..rows.len()).find(|&i| {
        let v;
        (v, rows[i]) = rows[i].split_at_unchecked(1);
        v[0] == null_sentinel
    });

    // No nulls just return None
    let first_null = first_null?;

    let mut bm = BitmapBuilder::new();
    bm.reserve(rows.len());
    bm.extend_constant(first_null, true);
    bm.push(false);
    bm.extend_trusted_len_iter(rows[first_null + 1..].iter_mut().map(|row| {
        let v;
        (v, *row) = row.split_at_unchecked(1);
        v[0] != null_sentinel
    }));
    bm.into_opt_validity()
}

// We inline this in an attempt to avoid the dispatch cost.
#[inline(always)]
fn dtype_and_data_to_encoded_item_len(
    dtype: &ArrowDataType,
    data: &[u8],
    opt: RowEncodingOptions,
    dict: Option<&RowEncodingContext>,
) -> usize {
    // Fast path: if the size is fixed, we can just divide.
    if let Some(size) = fixed_size(dtype, opt, dict) {
        return size;
    }

    use ArrowDataType as D;
    match dtype {
        D::Binary | D::LargeBinary | D::BinaryView | D::Utf8 | D::LargeUtf8 | D::Utf8View
            if opt.contains(RowEncodingOptions::NO_ORDER) =>
        unsafe { no_order::len_from_buffer(data, opt) },
        D::Binary | D::LargeBinary | D::BinaryView => unsafe {
            binary::encoded_item_len(data, opt)
        },
        D::Utf8 | D::LargeUtf8 | D::Utf8View => unsafe { utf8::len_from_buffer(data, opt) },

        D::List(list_field) | D::LargeList(list_field) => {
            let mut data = data;
            let mut item_len = 0;

            let list_continuation_token = opt.list_continuation_token();

            while data[0] == list_continuation_token {
                data = &data[1..];
                let len = dtype_and_data_to_encoded_item_len(list_field.dtype(), data, opt, dict);
                data = &data[len..];
                item_len += 1 + len;
            }
            1 + item_len
        },

        D::FixedSizeBinary(_) => todo!(),
        D::FixedSizeList(fsl_field, width) => {
            let mut data = &data[1..];
            let mut item_len = 1; // validity byte

            for _ in 0..*width {
                let len = dtype_and_data_to_encoded_item_len(
                    fsl_field.dtype(),
                    data,
                    opt.into_nested(),
                    dict,
                );
                data = &data[len..];
                item_len += len;
            }
            item_len
        },
        D::Struct(struct_fields) => {
            let mut data = &data[1..];
            let mut item_len = 1; // validity byte

            for struct_field in struct_fields {
                let len = dtype_and_data_to_encoded_item_len(
                    struct_field.dtype(),
                    data,
                    opt.into_nested(),
                    dict,
                );
                data = &data[len..];
                item_len += len;
            }
            item_len
        },

        D::Union(_) => todo!(),
        D::Map(_, _) => todo!(),
        D::Decimal32(_, _) => todo!(),
        D::Decimal64(_, _) => todo!(),
        D::Decimal256(_, _) => todo!(),
        D::Extension(_) => todo!(),
        D::Unknown => todo!(),

        _ => unreachable!(),
    }
}

fn rows_for_fixed_size_list<'a>(
    dtype: &ArrowDataType,
    opt: RowEncodingOptions,
    dict: Option<&RowEncodingContext>,
    width: usize,
    rows: &mut [&'a [u8]],
    nested_rows: &mut Vec<&'a [u8]>,
) {
    nested_rows.clear();
    nested_rows.reserve(rows.len() * width);

    // Fast path: if the size is fixed, we can just divide.
    if let Some(size) = fixed_size(dtype, opt, dict) {
        for row in rows.iter_mut() {
            for i in 0..width {
                nested_rows.push(&row[(i * size)..][..size]);
            }
            *row = &row[size * width..];
        }
        return;
    }

    // @TODO: This is quite slow since we need to dispatch for possibly every nested type
    for row in rows.iter_mut() {
        for _ in 0..width {
            let length = dtype_and_data_to_encoded_item_len(dtype, row, opt.into_nested(), dict);
            let v;
            (v, *row) = row.split_at(length);
            nested_rows.push(v);
        }
    }
}

unsafe fn decode_cat<T: NativeType + FixedLengthEncoding + CatNative>(
    rows: &mut [&[u8]],
    opt: RowEncodingOptions,
    ctx: &RowEncodingCategoricalContext,
) -> PrimitiveArray<T>
where
    T::Encoded: FromSlice,
{
    if ctx.is_enum || !opt.is_ordered() {
        numeric::decode_primitive::<T>(rows, opt)
    } else {
        variable::utf8::decode_str_as_cat::<T>(rows, opt, &ctx.mapping)
    }
}

unsafe fn decode(
    rows: &mut [&[u8]],
    opt: RowEncodingOptions,
    dict: Option<&RowEncodingContext>,
    dtype: &ArrowDataType,
) -> ArrayRef {
    use ArrowDataType as D;

    if let Some(RowEncodingContext::Categorical(ctx)) = dict {
        return match dtype {
            D::UInt8 => decode_cat::<u8>(rows, opt, ctx).to_boxed(),
            D::UInt16 => decode_cat::<u16>(rows, opt, ctx).to_boxed(),
            D::UInt32 => decode_cat::<u32>(rows, opt, ctx).to_boxed(),
            _ => unreachable!(),
        };
    }

    match dtype {
        D::Null => NullArray::new(D::Null, rows.len()).to_boxed(),
        D::Boolean => boolean::decode_bool(rows, opt).to_boxed(),
        D::Binary | D::LargeBinary | D::BinaryView | D::Utf8 | D::LargeUtf8 | D::Utf8View
            if opt.contains(RowEncodingOptions::NO_ORDER) =>
        {
            let array = no_order::decode_variable_no_order(rows, opt);

            if matches!(dtype, D::Utf8 | D::LargeUtf8 | D::Utf8View) {
                unsafe { array.to_utf8view_unchecked() }.to_boxed()
            } else {
                array.to_boxed()
            }
        },
        D::Binary | D::LargeBinary | D::BinaryView => binary::decode_binview(rows, opt).to_boxed(),
        D::Utf8 | D::LargeUtf8 | D::Utf8View => decode_str(rows, opt).boxed(),

        D::Struct(fields) => {
            let validity = decode_validity(rows, opt);

            let values = match dict {
                None => fields
                    .iter()
                    .map(|struct_fld| decode(rows, opt.into_nested(), None, struct_fld.dtype()))
                    .collect(),
                Some(RowEncodingContext::Struct(dicts)) => fields
                    .iter()
                    .zip(dicts)
                    .map(|(struct_fld, dict)| {
                        decode(rows, opt.into_nested(), dict.as_ref(), struct_fld.dtype())
                    })
                    .collect(),
                _ => unreachable!(),
            };
            StructArray::new(dtype.clone(), rows.len(), values, validity).to_boxed()
        },
        D::FixedSizeList(fsl_field, width) => {
            let validity = decode_validity(rows, opt);

            // @TODO: we could consider making this into a scratchpad
            let mut nested_rows = Vec::new();
            rows_for_fixed_size_list(
                fsl_field.dtype(),
                opt.into_nested(),
                dict,
                *width,
                rows,
                &mut nested_rows,
            );

            let values = decode(&mut nested_rows, opt.into_nested(), dict, fsl_field.dtype());

            FixedSizeListArray::new(dtype.clone(), rows.len(), values, validity).to_boxed()
        },
        D::List(list_field) | D::LargeList(list_field) => {
            let mut validity = BitmapBuilder::new();

            // @TODO: we could consider making this into a scratchpad
            let num_rows = rows.len();
            let mut nested_rows = Vec::new();
            let mut offsets = Vec::with_capacity(rows.len() + 1);
            offsets.push(0);

            let list_null_sentinel = opt.list_null_sentinel();
            let list_continuation_token = opt.list_continuation_token();
            let list_termination_token = opt.list_termination_token();

            // @TODO: make a specialized loop for fixed size list_field.dtype()
            for (i, row) in rows.iter_mut().enumerate() {
                while row[0] == list_continuation_token {
                    *row = &row[1..];
                    let len = dtype_and_data_to_encoded_item_len(
                        list_field.dtype(),
                        row,
                        opt.into_nested(),
                        dict,
                    );
                    nested_rows.push(&row[..len]);
                    *row = &row[len..];
                }

                offsets.push(nested_rows.len() as i64);

                // @TODO: Might be better to make this a 2-loop system.
                if row[0] == list_null_sentinel {
                    *row = &row[1..];
                    validity.reserve(num_rows);
                    validity.extend_constant(i - validity.len(), true);
                    validity.push(false);
                    continue;
                }

                assert_eq!(row[0], list_termination_token);
                *row = &row[1..];
            }

            let validity = if validity.is_empty() {
                None
            } else {
                validity.extend_constant(num_rows - validity.len(), true);
                validity.into_opt_validity()
            };
            assert_eq!(offsets.len(), rows.len() + 1);

            let values = decode(
                &mut nested_rows,
                opt.into_nested(),
                dict,
                list_field.dtype(),
            );

            ListArray::<i64>::new(
                dtype.clone(),
                unsafe { OffsetsBuffer::new_unchecked(Buffer::from(offsets)) },
                values,
                validity,
            )
            .to_boxed()
        },

        dt => {
            if matches!(dt, D::Int128) {
                if let Some(dict) = dict {
                    return match dict {
                        RowEncodingContext::Decimal(precision) => {
                            decimal::decode(rows, opt, *precision).to_boxed()
                        },
                        _ => unreachable!(),
                    };
                }
            }

            with_match_arrow_primitive_type!(dt, |$T| {
                numeric::decode_primitive::<$T>(rows, opt).to_boxed()
            })
        },
    }
}
