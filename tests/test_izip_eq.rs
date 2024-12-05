use _lowlevel::izip_eq;

#[test]
fn test_izip_eq_unary() {
    let result: Vec<_> = izip_eq!(0..3).collect();
    assert_eq!(result, vec![0, 1, 2]);
}

#[test]
fn test_izip_eq_binary() {
    let result: Vec<_> = izip_eq!(0..3, 2..5).collect();
    assert_eq!(result, vec![(0, 2), (1, 3), (2, 4)]);
}

#[test]
#[should_panic]
fn test_izip_eq_binary_panics() {
    let _: Vec<_> = izip_eq!(0..3, 2..4).collect();
}

#[test]
fn test_izip_eq_nary() {
    let result: Vec<_> = izip_eq!(0..3, 2..5, 4..7, 5..8).collect();
    assert_eq!(result, vec![(0, 2, 4, 5), (1, 3, 5, 6), (2, 4, 6, 7)]);
}

#[test]
#[should_panic]
fn test_izip_eq_nary_panics() {
    let _: Vec<_> = izip_eq!(0..3, 2..5, 4..8, 5..8).collect();
}
