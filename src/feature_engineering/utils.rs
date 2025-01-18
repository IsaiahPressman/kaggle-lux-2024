pub fn one_hot_bool_encode_param_range<T>(val: T, range: &[T]) -> Vec<bool>
where
    T: Copy + Eq,
{
    let mut encoded = vec![false; range.len()];
    encoded[range
        .iter()
        .position(|&v| v == val)
        .expect("Got invalid value")] = true;
    encoded
}

pub fn one_hot_float_encode_param_range<T>(val: T, range: &[T]) -> Vec<f32>
where
    T: Copy + Eq,
{
    let mut encoded = vec![0.0; range.len()];
    encoded[range
        .iter()
        .position(|&v| v == val)
        .expect("Got invalid value")] = 1.0;
    encoded
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_one_hot_encode_param_range() {
        let range = vec![1, 3, 10];
        assert_eq!(
            one_hot_bool_encode_param_range(1, &range),
            vec![true, false, false]
        );
        assert_eq!(
            one_hot_bool_encode_param_range(3, &range),
            vec![false, true, false]
        );
        assert_eq!(
            one_hot_bool_encode_param_range(10, &range),
            vec![false, false, true]
        );
    }

    #[test]
    #[should_panic(expected = "Got invalid value")]
    fn test_one_hot_encode_param_range_panics() {
        let range = vec![1, 2, 3];
        one_hot_bool_encode_param_range(0, &range);
    }
}
