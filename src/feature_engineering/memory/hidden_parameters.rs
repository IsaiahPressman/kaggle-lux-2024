use crate::rules_engine::action::Action;
use crate::rules_engine::env::estimate_vision_power_map;
use crate::rules_engine::params::KnownVariableParams;
use crate::rules_engine::state::Observation;
use itertools::Itertools;
use numpy::ndarray::Zip;

const MIN_PROBABILITY_EPSILON: f64 = 1e-4;

/// Represents the probability distribution over the likelihood of each option being the true one
#[derive(Debug, Default)]
pub struct Likelihoods<T> {
    options: Vec<T>,
    weights: Vec<f64>,
}

impl<T: Copy> Likelihoods<T> {
    fn renormalize(&mut self) {
        let sum: f64 = self.weights.iter().copied().sum();
        self.weights.iter_mut().for_each(|w| *w /= sum);
    }

    /// Renormalize, but don't allow probabilities to go (much) below EPSILON
    fn conservative_renormalize(&mut self) {
        self.renormalize();
        let sum: f64 = self
            .weights
            .iter()
            .map(|w| w.max(MIN_PROBABILITY_EPSILON))
            .sum();
        self.weights.iter_mut().for_each(|w| *w /= sum);
    }

    fn iter_options_weights(
        &self,
    ) -> impl Iterator<Item = (T, f64)> + use<'_, T> {
        self.options
            .iter()
            .copied()
            .zip_eq(self.weights.iter().copied())
    }

    fn iter_options_mut_weights(
        &mut self,
    ) -> impl Iterator<Item = (T, &mut f64)> {
        self.options.iter().copied().zip_eq(self.weights.iter_mut())
    }
}

#[derive(Debug, Default)]
pub struct HiddenParametersMemory {
    nebula_tile_vision_reduction_options: Vec<i32>,
    nebula_tile_vision_reduction_mask: Vec<bool>,
    nebula_tile_energy_reduction_likelihoods: Likelihoods<i32>,
    unit_sap_dropoff_factor_likelihoods: Likelihoods<f32>,
    unit_energy_void_dropoff_factor_likelihoods: Likelihoods<f32>,
}

impl HiddenParametersMemory {
    pub fn new(
        nebula_tile_vision_reduction_options: Vec<i32>,
        nebula_tile_energy_reduction_likelihoods: Likelihoods<i32>,
        unit_sap_dropoff_factor_likelihoods: Likelihoods<f32>,
        unit_energy_void_dropoff_factor_likelihoods: Likelihoods<f32>,
    ) -> Self {
        let nebula_tile_vision_reduction_mask =
            vec![true; nebula_tile_vision_reduction_options.len()];
        Self {
            nebula_tile_vision_reduction_options,
            nebula_tile_vision_reduction_mask,
            nebula_tile_energy_reduction_likelihoods,
            unit_sap_dropoff_factor_likelihoods,
            unit_energy_void_dropoff_factor_likelihoods,
        }
    }

    // TODO - consider whether or not we want this to be a categorical variable
    pub fn get_nebula_tile_vision_reduction_weights(&self) -> Vec<f32> {
        let sum = self
            .nebula_tile_vision_reduction_mask
            .iter()
            .filter(|mask| **mask)
            .count();
        assert!(sum > 0, "nebula_tile_vision_reduction_mask is all false");

        let weight = 1.0 / sum as f32;
        self.nebula_tile_vision_reduction_mask
            .iter()
            .map(|mask| if *mask { weight } else { 0.0 })
            .collect()
    }
}

fn determine_nebula_tile_vision_reduction(
    nebula_tile_vision_reduction_mask: &mut [bool],
    nebula_tile_vision_reduction_options: &[i32],
    obs: &Observation,
    map_size: [usize; 2],
    unit_sensor_range: usize,
) {
    if nebula_tile_vision_reduction_mask
        .iter()
        .filter(|mask| **mask)
        .count()
        == 1
    {
        return;
    }

    let expected_vision_power_map = estimate_vision_power_map(
        obs.get_my_units(),
        map_size,
        unit_sensor_range,
    );
    Zip::from(&expected_vision_power_map)
        .and(&obs.sensor_mask)
        .for_each(|expected_vision, can_see| {
            if *expected_vision > 0 && !can_see {
                nebula_tile_vision_reduction_options
                    .iter()
                    .zip_eq(nebula_tile_vision_reduction_mask.iter_mut())
                    .for_each(|(vision_reduction, mask)| {
                        if vision_reduction < expected_vision {
                            *mask = false
                        }
                    });
            }
        });

    if nebula_tile_vision_reduction_mask.iter().all(|mask| !mask) {
        panic!("nebula_tile_vision_reduction_mask is all false")
    }
}

fn estimate_nebula_tile_energy_reduction(
    unit_energies_last_turn: &mut [i32],
    nebula_tile_energy_reduction_likelihoods: &mut Likelihoods<i32>,
    energy_field_likelihoods: &Likelihoods<i32>,
    obs: &Observation,
    params: KnownVariableParams,
    last_actions: &[Action],
) {
    // NB: This assumes that units don't take invalid actions (like moving into an asteroid)
    let unit_energies_without_field = unit_energies_last_turn
        .iter()
        .copied()
        .zip_eq(last_actions)
        .map(|(e, a)| match a {
            Action::NoOp => e,
            Action::Up | Action::Right | Action::Down | Action::Left => {
                e - params.unit_move_cost
            },
            Action::Sap(_) => e - params.unit_sap_cost,
        })
        .collect_vec();

    let opp_units = obs.get_opp_units();
    for (base_e, actual) in obs
        .get_my_units()
        .iter()
        .filter(|u| {
            obs.nebulae.contains(&u.pos) || !obs.sensor_mask[u.pos.as_index()]
        })
        .filter(|u| {
            // Skip units that we think could have been sapped
            opp_units.iter().all(|opp_u| {
                let [dx, dy] = opp_u.pos.subtract(u.pos);
                dx.abs() > params.unit_sap_range
                    || dy.abs() > params.unit_sap_range
            })
        })
        .map(|u| (unit_energies_without_field[u.id], u.energy))
    {
        for (de, e_weight) in energy_field_likelihoods.iter_options_weights() {
            for (dn, n_weight) in nebula_tile_energy_reduction_likelihoods
                .iter_options_mut_weights()
            {
                if base_e + de - dn != actual {
                    continue;
                }

                *n_weight *= e_weight
            }
        }
    }
    nebula_tile_energy_reduction_likelihoods.conservative_renormalize();
    unit_energies_last_turn.fill(0);
    obs.get_my_units()
        .iter()
        .for_each(|u| unit_energies_last_turn[u.id] = u.energy);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules_engine::state::{Pos, Unit};
    use numpy::ndarray::arr2;
    use rstest::rstest;

    #[test]
    fn test_get_nebula_tile_vision_reduction_weights() {
        let mut memory = HiddenParametersMemory::default();
        memory.nebula_tile_vision_reduction_mask = vec![true; 3];
        let result = memory.get_nebula_tile_vision_reduction_weights();
        assert_eq!(result, vec![1.0 / 3.0; 3]);

        memory.nebula_tile_vision_reduction_mask = vec![false, true, false];
        let result = memory.get_nebula_tile_vision_reduction_weights();
        assert_eq!(result, vec![0.0, 1.0, 0.0]);

        memory.nebula_tile_vision_reduction_mask = vec![true; 2];
        let result = memory.get_nebula_tile_vision_reduction_weights();
        assert_eq!(result, vec![0.5; 2]);

        memory.nebula_tile_vision_reduction_mask = vec![true, false];
        let result = memory.get_nebula_tile_vision_reduction_weights();
        assert_eq!(result, vec![1.0, 0.0]);
    }

    #[test]
    #[should_panic(expected = "nebula_tile_vision_reduction_mask is all false")]
    fn test_get_nebula_tile_vision_reduction_weights_panics() {
        let mut memory = HiddenParametersMemory::default();
        memory.nebula_tile_vision_reduction_mask = vec![false, false];
        memory.get_nebula_tile_vision_reduction_weights();
    }

    #[test]
    fn test_determine_nebula_tile_vision_reduction() {
        let mut mask = vec![true, true, true];
        let options = vec![0, 1, 2];
        let map_size = [3, 3];
        let unit_sensor_range = 1;

        let mut obs = Observation::default();
        obs.sensor_mask = arr2(&[
            [true, false, true],
            [false, false, true],
            [true, true, true],
        ]);
        obs.units[0] = vec![Unit::with_pos(Pos::new(0, 0))];

        determine_nebula_tile_vision_reduction(
            &mut mask,
            &options,
            &obs,
            map_size,
            unit_sensor_range,
        );
        assert_eq!(mask, vec![false, true, true]);
    }

    #[rstest]
    #[case(vec![true, true, true])]
    #[should_panic(expected = "nebula_tile_vision_reduction_mask is all false")]
    #[case(vec![true, true, false])]
    fn test_determine_nebula_tile_vision_reduction_panics(
        #[case] mask: Vec<bool>,
    ) {
        let options = vec![0, 1, 2];
        let map_size = [3, 3];
        let unit_sensor_range = 1;

        let mut obs = Observation::default();
        obs.sensor_mask = arr2(&[
            [false, false, true],
            [false, false, true],
            [true, true, true],
        ]);
        obs.units[0] = vec![Unit::with_pos(Pos::new(0, 0))];

        determine_nebula_tile_vision_reduction(
            &mut mask.clone(),
            &options,
            &obs,
            map_size,
            unit_sensor_range,
        )
    }

    #[test]
    #[ignore]
    fn test_estimate_nebula_tile_energy_reduction() {
        todo!()
    }
}
