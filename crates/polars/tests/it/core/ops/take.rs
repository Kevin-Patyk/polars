use super::*;

#[test]
fn test_list_gather_nulls_and_empty() {
    let a: &[i32] = &[];
    let a = Series::new("".into(), a);
    let b = Series::try_build("".into(), &[None, Some(a.clone())]).unwrap();
    let indices = [Some(0 as IdxSize), Some(1), None]
        .into_iter()
        .collect_ca("".into());
    let out = b.take(&indices).unwrap();
    let expected = Series::try_build("".into(), &[None, Some(a), None]).unwrap();
    assert!(out.equals_missing(&expected))
}
