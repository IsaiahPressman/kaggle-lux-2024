use itertools::Itertools;

#[derive(Debug, Clone, Default)]
pub struct MaskedPossibilities<T> {
    options: Vec<T>,
    mask: Vec<bool>,
}

impl<T> MaskedPossibilities<T> {
    pub fn new(options: Vec<T>, mask: Vec<bool>) -> Self {
        assert_eq!(options.len(), mask.len());
        Self { options, mask }
    }

    pub fn from_options(options: Vec<T>) -> Self {
        let mask = vec![true; options.len()];
        Self { options, mask }
    }

    pub fn get_weighted_possibilities(&self) -> Vec<f32> {
        let sum = self.mask.iter().filter(|mask| **mask).count();
        assert!(sum > 0, "self.mask is all false");

        let weight = 1.0 / sum as f32;
        self.mask
            .iter()
            .map(|mask| if *mask { weight } else { 0.0 })
            .collect()
    }

    pub fn still_unsolved(&self) -> bool {
        self.mask.iter().filter(|mask| **mask).count() > 1
    }

    pub fn iter_options_mut_mask(
        &mut self,
    ) -> impl Iterator<Item = (&T, &mut bool)> {
        self.options.iter().zip_eq(self.mask.iter_mut())
    }

    pub fn all_masked(&self) -> bool {
        self.mask.iter().all(|mask| !mask)
    }

    pub fn get_options(&self) -> &[T] {
        &self.options
    }

    pub fn get_mask(&self) -> &[bool] {
        &self.mask
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn test_get_weighted_possibilities() {
        let mut possibilities: MaskedPossibilities<usize> =
            MaskedPossibilities::default();
        possibilities.mask = vec![true; 3];
        let result = possibilities.get_weighted_possibilities();
        assert_eq!(result, vec![1.0 / 3.0; 3]);

        possibilities.mask = vec![false, true, false];
        let result = possibilities.get_weighted_possibilities();
        assert_eq!(result, vec![0.0, 1.0, 0.0]);

        possibilities.mask = vec![true; 2];
        let result = possibilities.get_weighted_possibilities();
        assert_eq!(result, vec![0.5; 2]);

        possibilities.mask = vec![true, false];
        let result = possibilities.get_weighted_possibilities();
        assert_eq!(result, vec![1.0, 0.0]);
    }

    #[test]
    #[should_panic(expected = "self.mask is all false")]
    fn test_get_weighted_possibilities_panics() {
        let mut possibilities: MaskedPossibilities<usize> =
            MaskedPossibilities::default();
        possibilities.mask = vec![false; 2];
        possibilities.get_weighted_possibilities();
    }
}
