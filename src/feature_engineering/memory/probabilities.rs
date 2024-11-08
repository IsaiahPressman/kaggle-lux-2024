use itertools::Itertools;
use std::fmt;

pub const PROBABILITY_EPSILON: f64 = 1e-10;

fn is_valid_probability_sum(sum: f64) -> bool {
    (sum - 1.0).abs() < PROBABILITY_EPSILON
}

/// Represents a valid probability distribution over options
#[derive(Debug)]
pub struct Probabilities<T> {
    options: Vec<T>,
    probs: Vec<f64>,
}

impl<T: Default> Default for Probabilities<T> {
    fn default() -> Self {
        Self {
            options: vec![T::default()],
            probs: vec![1.0],
        }
    }
}

impl<T: Copy> Probabilities<T> {
    pub fn new(options: Vec<T>, probs: Vec<f64>) -> Self {
        let result = Self { options, probs };
        result.validate();
        result
    }

    pub fn new_uniform(options: Vec<T>) -> Self {
        let probs = vec![1.0 / options.len() as f64; options.len()];
        Self::new(options, probs)
    }

    fn validate(&self) {
        assert_eq!(self.options.len(), self.probs.len());
        assert!(self.len() >= 1);
        assert!(self.probs.iter().all(|w| *w >= 0.));
        assert!(is_valid_probability_sum(self.probs.iter().copied().sum()));
    }

    pub fn len(&self) -> usize {
        self.options.len()
    }

    pub fn iter_options_probs(&self) -> impl Iterator<Item = (T, f64)> + '_ {
        self.options
            .iter()
            .zip_eq(self.probs.iter())
            .map(|(o, p)| (*o, *p))
    }

    pub fn iter_probs(&self) -> impl Iterator<Item = f64> + '_ {
        self.probs.iter().copied()
    }
}

#[derive(Debug, Clone)]
pub struct InvalidProbabilities(Vec<f64>, f64);

impl fmt::Display for InvalidProbabilities {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Invalid probabilities with sum {}: {:?}", self.1, self.0)
    }
}

impl<T> TryFrom<Likelihoods<T>> for Probabilities<T> {
    type Error = InvalidProbabilities;

    fn try_from(likelihoods: Likelihoods<T>) -> Result<Self, Self::Error> {
        let sum: f64 = likelihoods.weights.iter().sum();
        if is_valid_probability_sum(sum) {
            Ok(Self {
                options: likelihoods.options,
                probs: likelihoods.weights,
            })
        } else {
            Err(InvalidProbabilities(likelihoods.weights, sum))
        }
    }
}

/// Represents the likelihoods of various options, where likelihood >= 0
#[derive(Debug)]
pub struct Likelihoods<T> {
    options: Vec<T>,
    weights: Vec<f64>,
}

impl<T> Likelihoods<T> {
    pub fn new(options: Vec<T>, weights: Vec<f64>) -> Self {
        let result = Self { options, weights };
        result.validate();
        result
    }

    fn validate(&self) {
        assert_eq!(self.options.len(), self.weights.len());
        assert!(self.len() >= 1);
        assert!(self.weights.iter().all(|w| *w >= 0.));
    }

    pub fn len(&self) -> usize {
        self.options.len()
    }

    pub fn renormalize(&mut self) {
        let sum: f64 = self.weights.iter().sum();
        if sum <= 0.0 {
            panic!("Division by 0: {:?}", self.weights);
        }
        self.weights.iter_mut().for_each(|w| *w /= sum);
    }

    pub fn safe_renormalize(&mut self) {
        let sum: f64 = self.weights.iter().copied().sum();
        if sum <= 0.0 {
            let uniform_prob = 1.0 / self.len() as f64;
            self.weights.fill(uniform_prob);
        } else {
            self.weights.iter_mut().for_each(|w| *w /= sum);
        }
    }

    /// Renormalize, but don't allow probabilities to go (much) below min_probability
    pub fn conservative_renormalize(&mut self, min_probability: f64) {
        self.renormalize();
        self.weights
            .iter_mut()
            .for_each(|w| *w = w.max(min_probability));
        self.renormalize();
    }

    pub fn iter_options(&self) -> impl Iterator<Item = &T> {
        self.options.iter()
    }

    pub fn iter_mut_weights(&mut self) -> impl Iterator<Item = &mut f64> {
        self.weights.iter_mut()
    }
}

impl<T> From<Probabilities<T>> for Likelihoods<T> {
    fn from(probabilities: Probabilities<T>) -> Self {
        Self {
            options: probabilities.options,
            weights: probabilities.probs,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_likelihoods_renormalize() {
        let mut likelihoods =
            Likelihoods::new((0..4).collect_vec(), vec![0.2, 0.2, 0.1, 0.0]);
        likelihoods.renormalize();
        assert_eq!(likelihoods.weights, vec![0.4, 0.4, 0.2, 0.0]);
    }

    #[test]
    fn test_likelihoods_safe_renormalize() {
        let mut likelihoods =
            Likelihoods::new((0..4).collect_vec(), vec![0.0, 0.0, 0.0, 0.0]);
        likelihoods.safe_renormalize();
        assert_eq!(likelihoods.weights, vec![0.25; 4]);
    }

    #[test]
    fn test_likelihoods_conservative_renormalize() {
        let min_prob = 1e-4;
        let mut likelihoods = Likelihoods::new(
            (0..5).collect_vec(),
            vec![0.2, 0.2, 0.1, 0.0, 0.0],
        );
        likelihoods.conservative_renormalize(min_prob);
        likelihoods
            .weights
            .iter()
            .copied()
            .zip_eq(vec![0.4, 0.4, 0.2, min_prob, min_prob])
            .for_each(|(actual, expected)| {
                assert_ne!(actual, 0.0);
                assert!(actual / expected >= 1.0 - 2.0 * min_prob);
                assert!(actual / expected <= 1.0);
            })
    }
}
