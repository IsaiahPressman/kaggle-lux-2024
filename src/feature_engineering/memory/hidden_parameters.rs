use crate::feature_engineering::memory::probabilities::{
    Likelihoods, Probabilities,
};
use crate::rules_engine::action::Action;
use crate::rules_engine::action::Action::{Down, Left, NoOp, Right, Sap, Up};
use crate::rules_engine::env::estimate_vision_power_map;
use crate::rules_engine::param_ranges::ParamRanges;
use crate::rules_engine::params::{FixedParams, KnownVariableParams};
use crate::rules_engine::state::{Observation, Pos, Unit};
use itertools::Itertools;
use numpy::ndarray::Zip;
use std::cmp::{max, min};
use std::collections::BTreeMap;

const MIN_LIKELIHOOD_WEIGHT: f64 = 1e-4;

#[derive(Debug, Default)]
pub struct HiddenParametersMemory {
    nebula_tile_vision_reduction_options: Vec<i32>,
    nebula_tile_vision_reduction_mask: Vec<bool>,
    nebula_tile_energy_reduction_options: Vec<i32>,
    known_nebula_tile_energy_reduction: Option<i32>,
    unit_sap_dropoff_factor_probs: Probabilities<f32>,
    // TODO: Some way to not clone last_obs?
    last_obs_data: LastObservationData,
    // TODO
    // unit_energy_void_factor_probs: Probabilities<f32>,
}

impl HiddenParametersMemory {
    pub fn new(param_ranges: &ParamRanges) -> Self {
        let nebula_tile_vision_reduction_options = param_ranges
            .nebula_tile_vision_reduction
            .iter()
            .copied()
            .dedup()
            .collect_vec();
        let nebula_tile_vision_reduction_mask =
            vec![true; param_ranges.nebula_tile_vision_reduction.len()];
        let nebula_tile_energy_reduction_options = param_ranges
            .nebula_tile_energy_reduction
            .iter()
            .copied()
            .dedup()
            .collect_vec();
        let unit_sap_dropoff_factor_probs =
            Probabilities::new_uniform_unreduced(
                param_ranges.unit_sap_dropoff_factor.clone(),
            );
        let last_obs_data = LastObservationData::default();
        Self {
            nebula_tile_vision_reduction_options,
            nebula_tile_vision_reduction_mask,
            nebula_tile_energy_reduction_options,
            known_nebula_tile_energy_reduction: None,
            unit_sap_dropoff_factor_probs,
            last_obs_data,
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
        if let Some(known_reduction) = self.known_nebula_tile_energy_reduction {
            self.nebula_tile_energy_reduction_options
                .iter()
                .map(|reduction| {
                    if *reduction == known_reduction {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect()
        } else {
            let weight =
                1.0 / self.nebula_tile_energy_reduction_options.len() as f32;
            vec![weight; self.nebula_tile_energy_reduction_options.len()]
        }
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
        if self.known_nebula_tile_energy_reduction.is_none() {
            self.known_nebula_tile_energy_reduction =
                determine_nebula_tile_energy_reduction(
                    &self.nebula_tile_vision_reduction_options,
                    obs,
                    &self.last_obs_data.my_units_last_turn,
                    last_actions,
                    variable_params,
                );
        }

        // self.nebula_tile_energy_reduction_probs =
        //     determine_nebula_tile_energy_reduction(
        //         mem::take(&mut self.nebula_tile_energy_reduction_probs),
        //         obs,
        //         &self.last_obs_data.my_units_last_turn,
        //         last_actions,
        //         &self.energy_field_probs,
        //         fixed_params,
        //         variable_params,
        //     );
        todo!("self.unit_sap_dropoff_factor_probs = estimate_unit_sap_dropoff_factor();");
        todo!("Update last_obs_data");
    }
}

#[derive(Debug, Default)]
struct LastObservationData {
    my_units_last_turn: Vec<Unit>,
    opp_units_last_turn: Vec<Unit>,
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

fn determine_nebula_tile_energy_reduction(
    nebula_tile_energy_reduction_options: &[i32],
    obs: &Observation,
    my_units_last_turn: &[Unit],
    last_actions: &[Action],
    params: &KnownVariableParams,
) -> Option<i32> {
    // NB: This assumes that units don't take invalid actions (like moving into an asteroid)
    let unit_energies_before_nebula = my_units_last_turn
        .iter()
        .filter_map(|u| {
            let energy = match last_actions[u.id] {
                NoOp => u.energy,
                Up | Right | Down | Left => u.energy - params.unit_move_cost,
                Sap(_) => u.energy - params.unit_sap_cost,
            };
            obs.energy_field[u.pos.as_index()]
                .map(|energy_field| (u.id, energy + energy_field))
        })
        .collect::<BTreeMap<usize, i32>>();
    let opp_units = obs.get_opp_units();
    for (energy_before_nebula, actual) in obs
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
        .filter_map(|u| {
            unit_energies_before_nebula
                .get(&u.id)
                .map(|e| (e, u.energy))
        })
    {
        for &energy_loss in nebula_tile_energy_reduction_options.iter() {
            if energy_before_nebula - energy_loss == actual {
                return Some(energy_loss);
            }
        }
    }
    None
}

#[must_use]
fn _estimate_nebula_tile_energy_reduction(
    nebula_tile_energy_reduction_probs: Probabilities<i32>,
    obs: &Observation,
    my_units_last_turn: &[Unit],
    last_actions: &[Action],
    energy_field_probs: &Probabilities<i32>,
    fixed_params: &FixedParams,
    params: &KnownVariableParams,
) -> Probabilities<i32> {
    // NB: This assumes that units don't take invalid actions (like moving into an asteroid)
    let unit_energies_before_field = my_units_last_turn
        .iter()
        .map(|u| {
            let energy = match last_actions[u.id] {
                NoOp => u.energy,
                Up | Right | Down | Left => u.energy - params.unit_move_cost,
                Sap(_) => u.energy - params.unit_sap_cost,
            };
            (u.id, energy)
        })
        .collect::<BTreeMap<usize, i32>>();
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
        .filter_map(|u| {
            unit_energies_before_field.get(&u.id).map(|e| (e, u.energy))
        })
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
                .for_each(|(n_weight, w)| {
                    *n_weight *= w.max(MIN_LIKELIHOOD_WEIGHT)
                });
        }
    }
    nebula_tile_energy_reduction_likelihoods.renormalize();
    nebula_tile_energy_reduction_likelihoods
        .try_into()
        .unwrap_or_else(|err| panic!("{}", err))
}

#[must_use]
fn determine_unit_sap_dropoff_factor(
    unit_sap_dropoff_factor_probs: Probabilities<i32>,
    obs: &Observation,
    last_obs_data: &LastObservationData,
    my_last_actions: &[Action],
    energy_field_probs: &Probabilities<i32>,
    fixed_params: &FixedParams,
    params: &KnownVariableParams,
) -> Probabilities<i32> {
    let (sap_count, adjacent_sap_count) = compute_sap_count_maps(
        &last_obs_data.my_units_last_turn,
        my_last_actions,
        fixed_params,
    );
    // let unit_energies_before_field =
    //     reverse_engineer_energies_before_field(obs, last_obs_data, params);
    todo!()
}

fn compute_sap_count_maps(
    units_last_turn: &[Unit],
    last_actions: &[Action],
    fixed_params: &FixedParams,
) -> (BTreeMap<Pos, i32>, BTreeMap<Pos, i32>) {
    // NB: Assumes that all units that tried to sap had enough energy and were successful
    let mut sap_count = BTreeMap::new();
    let mut adjacent_sap_count = BTreeMap::new();
    for sap_target_pos in units_last_turn.iter().filter_map(|u| {
        if let Sap(sap_deltas) = last_actions[u.id] {
            Some(
                u.pos
                    .maybe_translate(sap_deltas, fixed_params.map_size)
                    .expect("Invalid sap_deltas"),
            )
        } else {
            None
        }
    }) {
        sap_count
            .entry(sap_target_pos)
            .and_modify(|count| *count += 1)
            .or_insert(1);
        for adjacent_pos in
            (-1..=1).cartesian_product(-1..=1).filter_map(|(dx, dy)| {
                if dx == 0 && dy == 0 {
                    None
                } else {
                    sap_target_pos
                        .maybe_translate([dx, dy], fixed_params.map_size)
                }
            })
        {
            adjacent_sap_count
                .entry(adjacent_pos)
                .and_modify(|count| *count += 1)
                .or_insert(1);
        }
    }
    (sap_count, adjacent_sap_count)
}

// fn reverse_engineer_energies_before_field(
//     obs: &Observation,
//     last_obs_data: &LastObservationData,
//     params: &KnownVariableParams,
//     sap_count: BTreeMap<Pos, i32>,
//     adjacent_sap_count: BTreeMap<Pos, i32>,
// ) -> BTreeMap<usize, i32> {
//     // NB: This assumes that units don't take invalid energy-wasting actions, such as
//     // moving off of the map
//     let id_to_unit: BTreeMap<usize, Unit> =
//         obs.get_opp_units().iter().map(|u| (u.id, u)).collect();
//     for (unit_last_turn, unit_now) in last_obs_data
//         .opp_units_last_turn
//         .iter()
//         .filter_map(|u_last_turn| {
//             id_to_unit
//                 .get(&u_last_turn.id)
//                 .map(|u_now| (u_last_turn, u_now))
//         })
//     {}
//     todo!()
// }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::feature_engineering::memory::probabilities::PROBABILITY_EPSILON;
    use crate::rules_engine::params::FIXED_PARAMS;
    use crate::rules_engine::state::{Pos, Unit};
    use numpy::ndarray::arr2;
    use pretty_assertions::assert_eq;
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
            0.2 / (0.3 + MIN_LIKELIHOOD_WEIGHT),
            0.1 / (0.3 + MIN_LIKELIHOOD_WEIGHT),
            MIN_LIKELIHOOD_WEIGHT / (0.3 + MIN_LIKELIHOOD_WEIGHT),
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
        let my_units_last_turn = vec![
            Unit::new(Pos::default(), 10, 0),
            Unit::new(Pos::default(), 10, 1),
        ];
        let mut obs = Observation::default();
        obs.sensor_mask = arr2(&[[true, true, true, false]; 4]);
        obs.units = [my_units.clone(), vec![Unit::with_pos(Pos::new(3, 3))]];
        obs.nebulae = vec![Pos::new(0, 1), Pos::new(3, 3)];
        let fixed_params = FIXED_PARAMS;
        let mut params = KnownVariableParams::default();
        params.unit_sap_range = 0;
        let last_actions = vec![NoOp; fixed_params.max_units];
        let energy_field_probs = Probabilities::new(
            vec![-2, -1, 0, 1, 2],
            vec![0.1, 0.2, 0.4, 0.2, 0.1],
        );
        let nebula_tile_energy_reduction_probs =
            Probabilities::new_uniform(vec![0, 1, 2]);

        let probs_result = _estimate_nebula_tile_energy_reduction(
            nebula_tile_energy_reduction_probs,
            &obs,
            &my_units_last_turn,
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

    #[test]
    #[ignore]
    fn test_estimate_unit_sap_dropoff_factor() {
        todo!()
    }

    #[test]
    fn test_compute_sap_count_maps() {
        let units = vec![
            Unit::new(Pos::new(2, 2), 0, 0),
            Unit::new(Pos::new(2, 2), 0, 2),
            Unit::new(Pos::new(2, 2), 0, 3),
            Unit::new(Pos::new(2, 1), 0, 4),
        ];
        let actions = vec![
            NoOp,
            // Ignore unused action
            Sap([-3, -3]),
            Sap([0, 0]),
            Sap([-2, -2]),
            Sap([-2, -1]),
        ];
        let fixed_params = FIXED_PARAMS;
        let (sap_count, adjacent_sap_count) =
            compute_sap_count_maps(&units, &actions, &fixed_params);
        let expected_sap_count =
            BTreeMap::from([(Pos::new(2, 2), 1), (Pos::new(0, 0), 2)]);
        assert_eq!(sap_count, expected_sap_count);
        let expected_adjacent_sap_count = BTreeMap::from([
            (Pos::new(0, 1), 2),
            (Pos::new(1, 0), 2),
            (Pos::new(1, 1), 3),
            (Pos::new(1, 2), 1),
            (Pos::new(1, 3), 1),
            (Pos::new(2, 1), 1),
            (Pos::new(2, 3), 1),
            (Pos::new(3, 1), 1),
            (Pos::new(3, 2), 1),
            (Pos::new(3, 3), 1),
        ]);
        assert_eq!(adjacent_sap_count, expected_adjacent_sap_count);
    }

    #[test]
    #[should_panic(expected = "Invalid sap_deltas")]
    fn test_test_compute_sap_count_maps_panics() {
        let units = vec![Unit::new(Pos::new(0, 0), 0, 0)];
        let actions = vec![Sap([-1, -1])];
        let fixed_params = FIXED_PARAMS;
        compute_sap_count_maps(&units, &actions, &fixed_params);
    }

    #[test]
    #[ignore]
    fn test_reverse_engineer_energies_before_field() {
        todo!()
    }
}
