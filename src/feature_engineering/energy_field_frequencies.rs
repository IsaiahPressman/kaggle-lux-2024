use crate::feature_engineering::memory::probabilities::Probabilities;
use serde::Deserialize;
use std::fs;
use std::path::Path;
use std::sync::LazyLock;

pub static ENERGY_FIELD_FREQUENCIES: LazyLock<EnergyFieldFrequencies> =
    LazyLock::new(load_energy_field_frequencies);
pub static ENERGY_FIELD_PROBABILITIES: LazyLock<Probabilities<i32>> =
    LazyLock::new(|| {
        Probabilities::from_counts(
            ENERGY_FIELD_FREQUENCIES.energy_deltas.clone(),
            ENERGY_FIELD_FREQUENCIES.counts.clone(),
        )
    });

fn load_energy_field_frequencies() -> EnergyFieldFrequencies {
    let path = Path::new(file!())
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("data")
        .join("../data/energy_field_frequencies.json");
    let json_data = fs::read_to_string(path).unwrap();
    serde_json::from_str(&json_data).unwrap()
}

#[derive(Debug, Clone, Deserialize)]
pub struct EnergyFieldFrequencies {
    seed: u32,
    pub energy_deltas: Vec<i32>,
    pub counts: Vec<usize>,
    frequencies: Vec<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use itertools::Itertools;

    #[test]
    fn test_energy_field_frequencies() {
        _ = ENERGY_FIELD_FREQUENCIES.clone();
    }

    #[test]
    fn test_energy_field_probabilities() {
        for (actual_p, expected_p) in ENERGY_FIELD_PROBABILITIES
            .iter_probs()
            .zip_eq(ENERGY_FIELD_FREQUENCIES.frequencies.iter().copied())
        {
            assert!((actual_p - expected_p).abs() <= 1e-10);
        }
    }
}
