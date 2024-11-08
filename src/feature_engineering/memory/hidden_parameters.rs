use crate::feature_engineering::memory::probabilities::{
    Likelihoods, Probabilities,
};
use crate::rules_engine::action::Action;
use crate::rules_engine::env::estimate_vision_power_map;
use crate::rules_engine::param_ranges::ParamRanges;
use crate::rules_engine::params::{FixedParams, KnownVariableParams};
use crate::rules_engine::state::{Observation, Unit};
use itertools::Itertools;
use numpy::ndarray::Zip;
use std::cmp::{max, min};
use std::mem;

const MIN_PROBABILITY: f64 = 1e-4;

#[derive(Debug, Default)]
pub struct HiddenParametersMemory {
    nebula_tile_vision_reduction_options: Vec<i32>,
    nebula_tile_vision_reduction_mask: Vec<bool>,
    nebula_tile_energy_reduction_probs: Probabilities<i32>,
    my_unit_energies_last_turn: Vec<Option<i32>>,
    energy_field_probs: Probabilities<i32>,
    // TODO
    // unit_sap_dropoff_factor_probs: Probabilities<f32>,
    // unit_energy_void_factor_probs: Probabilities<f32>,
}

impl HiddenParametersMemory {
    pub fn new(
        param_ranges: &ParamRanges,
        energy_field_probs: Probabilities<i32>,
        max_units: usize,
    ) -> Self {
        let nebula_tile_vision_reduction_mask =
            vec![true; param_ranges.nebula_tile_vision_reduction.len()];
        let nebula_tile_energy_reduction_probs = Probabilities::new_uniform(
            param_ranges.nebula_tile_energy_reduction.clone(),
        );
        let my_unit_energies_last_turn = vec![None; max_units];
        Self {
            nebula_tile_vision_reduction_options: param_ranges
                .nebula_tile_vision_reduction
                .clone(),
            nebula_tile_vision_reduction_mask,
            nebula_tile_energy_reduction_probs,
            energy_field_probs,
            my_unit_energies_last_turn,
        }
    }

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

    pub fn get_nebula_tile_energy_reduction_weights(&self) -> Vec<f32> {
        self.nebula_tile_energy_reduction_probs
            .iter_probs()
            .map(|p| p as f32)
            .collect()
    }

    pub fn update_memory(
        &mut self,
        obs: &Observation,
        last_actions: &[Action],
        fixed_params: &FixedParams,
        variable_params: &KnownVariableParams,
    ) {
        determine_nebula_tile_vision_reduction(
            &mut self.nebula_tile_vision_reduction_mask,
            &self.nebula_tile_vision_reduction_options,
            obs,
            fixed_params.map_size,
            variable_params.unit_sensor_range,
        );
        self.nebula_tile_energy_reduction_probs =
            estimate_nebula_tile_energy_reduction(
                mem::take(&mut self.nebula_tile_energy_reduction_probs),
                obs,
                &self.my_unit_energies_last_turn,
                last_actions,
                &self.energy_field_probs,
                fixed_params,
                variable_params,
            );
        update_unit_energies(
            &mut self.my_unit_energies_last_turn,
            obs.get_my_units(),
        );
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

#[must_use]
fn estimate_nebula_tile_energy_reduction(
    nebula_tile_energy_reduction_probs: Probabilities<i32>,
    obs: &Observation,
    my_unit_energies_last_turn: &[Option<i32>],
    last_actions: &[Action],
    energy_field_probs: &Probabilities<i32>,
    fixed_params: &FixedParams,
    params: &KnownVariableParams,
) -> Probabilities<i32> {
    // NB: This assumes that units don't take invalid actions (like moving into an asteroid)
    let unit_energies_before_field = my_unit_energies_last_turn
        .iter()
        .zip_eq(last_actions)
        .map(|(e_opt, a)| {
            e_opt.map(|e| match a {
                Action::NoOp => e,
                Action::Up | Action::Right | Action::Down | Action::Left => {
                    e - params.unit_move_cost
                },
                Action::Sap(_) => e - params.unit_sap_cost,
            })
        })
        .collect_vec();

    let mut nebula_tile_energy_reduction_likelihoods =
        Likelihoods::from(nebula_tile_energy_reduction_probs);
    let opp_units = obs.get_opp_units();
    for (base_e, actual) in obs
        .get_my_units()
        .iter()
        .filter(|u| {
            obs.nebulae.contains(&u.pos) || !obs.sensor_mask[u.pos.as_index()]
        })
        // Skip units that we think could have been sapped
        .filter(|u| {
            u.energy < 0
                || opp_units.iter().all(|opp_u| {
                    let [dx, dy] = opp_u.pos.subtract(u.pos);
                    dx.abs() > params.unit_sap_range
                        || dy.abs() > params.unit_sap_range
                })
        })
        .filter_map(|u| unit_energies_before_field[u.id].map(|e| (e, u.energy)))
    {
        let mut likelihood_weights =
            vec![0.0; nebula_tile_energy_reduction_likelihoods.len()];
        let mut should_update = false;
        for (n_weight, de_nebula) in likelihood_weights.iter_mut().zip_eq(
            nebula_tile_energy_reduction_likelihoods
                .iter_options()
                .copied(),
        ) {
            for (de_field, e_prob) in energy_field_probs.iter_options_probs() {
                if min(
                    max(
                        base_e + de_field - de_nebula,
                        fixed_params.min_unit_energy,
                    ),
                    fixed_params.max_unit_energy,
                ) == actual
                {
                    should_update = true;
                    *n_weight += e_prob;
                }
            }
        }
        if should_update {
            nebula_tile_energy_reduction_likelihoods
                .iter_mut_weights()
                .zip_eq(likelihood_weights.iter())
                .for_each(|(n_weight, w)| *n_weight *= *w);
        }
    }
    nebula_tile_energy_reduction_likelihoods
        .conservative_renormalize(MIN_PROBABILITY);
    nebula_tile_energy_reduction_likelihoods
        .try_into()
        .unwrap_or_else(|err| panic!("{}", err))
}

fn update_unit_energies(unit_energies: &mut [Option<i32>], units: &[Unit]) {
    unit_energies.fill(None);
    units
        .iter()
        .for_each(|u| unit_energies[u.id] = Some(u.energy));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::feature_engineering::memory::probabilities::PROBABILITY_EPSILON;
    use crate::rules_engine::params::FIXED_PARAMS;
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

    #[rstest]
    // Not in nebula
    #[case(vec![Unit::new(Pos::new(0, 0), 10, 0)], vec![1.0 / 3.0; 3])]
    // In seen nebula
    #[case(
        vec![Unit::new(Pos::new(0, 1), 10, 0)],
        vec![0.4 / 0.7, 0.2 / 0.7, 0.1 / 0.7]
    )]
    // Can't be seen - must be in nebula
    #[case(
        vec![Unit::new(Pos::new(0, 3), 11, 0)],
        vec![
            2. / 3. / (1. + MIN_PROBABILITY),
            1. / 3. / (1. + MIN_PROBABILITY),
            MIN_PROBABILITY / (1. + MIN_PROBABILITY),
        ]
    )]
    // Could be sapped - should be ignored
    #[case(vec![Unit::new(Pos::new(3, 3), 10, 0)], vec![1.0 / 3.0; 3])]
    // Has negative energy - should be ignored
    #[case(vec![Unit::new(Pos::new(0, 1), -10, 0)], vec![1.0 / 3.0; 3])]
    // No energy data from last turn - should be ignored
    #[case(vec![Unit::new(Pos::new(0, 3), 10, 2)], vec![1.0 / 3.0; 3])]
    // Multiple units chain probabilities correctly
    #[case(
        vec![
            Unit::new(Pos::new(0, 1), 9, 0),
            Unit::new(Pos::new(0, 3), 8, 1),
        ],
        vec![
            // Denominator = 0.2 * 0.1 + 0.4 * 0.2 + 0.4 * 0.2
            0.2 * 0.1 / 0.18,
            0.4 * 0.2 / 0.18,
            0.2 * 0.4 / 0.18,
        ]
    )]
    fn test_estimate_nebula_tile_energy_reduction(
        #[case] my_units: Vec<Unit>,
        #[case] expected_probs: Vec<f64>,
    ) {
        let unit_energies_last_turn = vec![Some(10), Some(10), None];
        let mut obs = Observation::default();
        obs.sensor_mask = arr2(&[[true, true, true, false]; 4]);
        obs.units = [my_units.clone(), vec![Unit::with_pos(Pos::new(3, 3))]];
        obs.nebulae = vec![Pos::new(0, 1), Pos::new(3, 3)];
        let fixed_params = FIXED_PARAMS;
        let mut params = KnownVariableParams::default();
        params.unit_sap_range = 0;
        let last_actions = vec![Action::NoOp; unit_energies_last_turn.len()];
        let energy_field_probs = Probabilities::new(
            vec![-2, -1, 0, 1, 2],
            vec![0.1, 0.2, 0.4, 0.2, 0.1],
        );
        let nebula_tile_energy_reduction_probs =
            Probabilities::new_uniform(vec![0, 1, 2]);

        let probs_result = estimate_nebula_tile_energy_reduction(
            nebula_tile_energy_reduction_probs,
            &obs,
            &unit_energies_last_turn,
            &last_actions,
            &energy_field_probs,
            &fixed_params,
            &params,
        );
        for (result_p, expected_p) in
            probs_result.iter_probs().zip_eq(expected_probs)
        {
            assert!(
                (result_p - expected_p).abs() < PROBABILITY_EPSILON,
                "{} != {}",
                result_p,
                expected_p
            );
        }
    }
}
