use itertools::Itertools;

#[derive(Debug, Clone, Default)]
pub struct MaskedPossibilities<T> {
    options: Vec<T>,
    pub mask: Vec<bool>,
}

#[allow(dead_code)]
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

    #[inline(always)]
    pub fn still_unsolved(&self) -> bool {
        self.mask.iter().filter(|mask| **mask).count() > 1
    }

    #[inline(always)]
    pub fn iter_unmasked_options_mut_mask(
        &mut self,
    ) -> impl Iterator<Item = (&T, &mut bool)> {
        self.options
            .iter()
            .zip_eq(self.mask.iter_mut())
            .filter(|(opt, mask)| **mask)
    }

    #[inline(always)]
    pub fn iter_unmasked_options(&self) -> impl Iterator<Item = &T> {
        self.options
            .iter()
            .zip_eq(self.mask.iter())
            .filter_map(|(opt, mask)| mask.then_some(opt))
    }

    #[inline(always)]
    pub fn all_masked(&self) -> bool {
        self.mask.iter().all(|mask| !mask)
    }

    #[inline(always)]
    pub fn get_options(&self) -> &[T] {
        &self.options
    }

    #[inline(always)]
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
        let possibilities = MaskedPossibilities {
            options: vec![0, 1, 2],
            mask: vec![true; 3],
        };
        let result = possibilities.get_weighted_possibilities();
        assert_eq!(result, vec![1.0 / 3.0; 3]);

        let possibilities = MaskedPossibilities {
            options: vec![0, 1, 2],
            mask: vec![false, true, false],
        };
        let result = possibilities.get_weighted_possibilities();
        assert_eq!(result, vec![0.0, 1.0, 0.0]);

        let possibilities = MaskedPossibilities {
            options: vec![0, 1],
            mask: vec![true; 2],
        };
        let result = possibilities.get_weighted_possibilities();
        assert_eq!(result, vec![0.5; 2]);

        let possibilities = MaskedPossibilities {
            options: vec![0, 1],
            mask: vec![true, false],
        };
        let result = possibilities.get_weighted_possibilities();
        assert_eq!(result, vec![1.0, 0.0]);
    }

    #[test]
    #[should_panic(expected = "self.mask is all false")]
    fn test_get_weighted_possibilities_panics() {
        let possibilities = MaskedPossibilities {
            options: vec![0, 1],
            mask: vec![false; 2],
        };
        possibilities.get_weighted_possibilities();
    }
}
