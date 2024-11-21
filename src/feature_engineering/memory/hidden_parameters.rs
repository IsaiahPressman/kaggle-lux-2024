use crate::feature_engineering::memory::masked_possibilities::MaskedPossibilities;
use crate::rules_engine::action::Action;
use crate::rules_engine::action::Action::{Down, Left, NoOp, Right, Sap, Up};
use crate::rules_engine::env::{estimate_vision_power_map, ENERGY_VOID_DELTAS};
use crate::rules_engine::param_ranges::ParamRanges;
use crate::rules_engine::params::{FixedParams, KnownVariableParams};
use crate::rules_engine::state::{Observation, Pos, Unit};
use itertools::Itertools;
use numpy::ndarray::Zip;
use std::collections::BTreeMap;

const MIN_LIKELIHOOD_WEIGHT: f64 = 1e-4;

#[derive(Debug, Default)]
pub struct HiddenParametersMemory {
    pub nebula_tile_vision_reduction: MaskedPossibilities<i32>,
    pub nebula_tile_energy_reduction: MaskedPossibilities<i32>,
    pub unit_sap_dropoff_factor: MaskedPossibilities<f32>,
    // pub unit_energy_void_factor: MaskedPossibilities<f32>,
    last_obs_data: LastObservationData,
}

impl HiddenParametersMemory {
    pub fn new(param_ranges: &ParamRanges) -> Self {
        let nebula_tile_vision_reduction = MaskedPossibilities::from_options(
            param_ranges
                .nebula_tile_vision_reduction
                .iter()
                .copied()
                .sorted()
                .dedup()
                .collect_vec(),
        );
        let nebula_tile_energy_reduction = MaskedPossibilities::from_options(
            param_ranges
                .nebula_tile_energy_reduction
                .iter()
                .copied()
                .sorted()
                .dedup()
                .collect_vec(),
        );
        let unit_sap_dropoff_factor = MaskedPossibilities::from_options(
            param_ranges
                .unit_sap_dropoff_factor
                .iter()
                .copied()
                .sorted_by(|a, b| a.partial_cmp(b).unwrap())
                .dedup()
                .collect_vec(),
        );
        // let unit_energy_void_factor = MaskedPossibilities::from_options(
        //     param_ranges
        //         .unit_energy_void_factor
        //         .iter()
        //         .copied()
        //         .sorted_by(|a, b| a.partial_cmp(b).unwrap())
        //         .dedup()
        //         .collect_vec(),
        // );
        let last_obs_data = LastObservationData::default();
        Self {
            nebula_tile_vision_reduction,
            nebula_tile_energy_reduction,
            unit_sap_dropoff_factor,
            last_obs_data,
        }
    }

    pub fn update(
        &mut self,
        obs: &Observation,
        last_actions: &[Action],
        fixed_params: &FixedParams,
        variable_params: &KnownVariableParams,
    ) {
        if self.nebula_tile_vision_reduction.still_unsolved() {
            determine_nebula_tile_vision_reduction(
                &mut self.nebula_tile_vision_reduction,
                obs,
                fixed_params.map_size,
                variable_params.unit_sensor_range,
            );
        }
        let new_match_just_started = obs.match_steps == 1;
        if self.nebula_tile_energy_reduction.still_unsolved()
            && !new_match_just_started
        {
            determine_nebula_tile_energy_reduction(
                &mut self.nebula_tile_energy_reduction,
                obs,
                &self.last_obs_data,
                last_actions,
                fixed_params,
                variable_params,
            );
        }
        if self.unit_sap_dropoff_factor.still_unsolved()
            && !new_match_just_started
        {
            let sap_count_maps = compute_sap_count_maps(
                &self.last_obs_data.my_units,
                last_actions,
                fixed_params.map_size,
            );
            determine_unit_sap_dropoff_factor(
                &mut self.unit_sap_dropoff_factor,
                obs,
                &self.last_obs_data,
                &sap_count_maps,
                variable_params,
            );
        }
        self.last_obs_data = LastObservationData::copy_from_obs(obs);
    }
}

#[derive(Debug, Default)]
struct LastObservationData {
    my_units: Vec<Unit>,
    opp_units: Vec<Unit>,
    nebulae: Vec<Pos>,
}

impl LastObservationData {
    fn copy_from_obs(obs: &Observation) -> Self {
        Self {
            my_units: obs.get_my_units().to_vec(),
            opp_units: obs.get_opp_units().to_vec(),
            nebulae: obs.nebulae.clone(),
        }
    }
}

fn determine_nebula_tile_vision_reduction(
    nebula_tile_vision_reduction_options: &mut MaskedPossibilities<i32>,
    obs: &Observation,
    map_size: [usize; 2],
    unit_sensor_range: usize,
) {
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
                    .iter_options_mut_mask()
                    .for_each(|(vision_reduction, mask)| {
                        if vision_reduction < expected_vision {
                            *mask = false
                        }
                    });
            }
        });

    if nebula_tile_vision_reduction_options.all_masked() {
        panic!("nebula_tile_vision_reduction_mask is all false")
    }
}

fn determine_nebula_tile_energy_reduction(
    nebula_tile_energy_reduction_options: &mut MaskedPossibilities<i32>,
    obs: &Observation,
    last_obs: &LastObservationData,
    last_actions: &[Action],
    fixed_params: &FixedParams,
    params: &KnownVariableParams,
) {
    // Note that the environment resolution order goes:
    // - Move units
    // - Resolve energy field
    // - Move nebulae and energy nodes
    // Therefore, whenever we are comparing energies, we take the current unit's energy and
    // position and compare it to last turn's nebulae and energy field. Confusingly enough,
    // last turn's energy field is provided in this turn's observation.
    let id_to_unit: BTreeMap<usize, Unit> =
        obs.get_my_units().iter().map(|u| (u.id, *u)).collect();
    // NB: This assumes that units don't take invalid actions (like moving into an asteroid)
    for (energy_before_nebula, actual) in last_obs
        .my_units
        .iter()
        .filter_map(|unit_last_turn| {
            id_to_unit
                .get(&unit_last_turn.id)
                .map(|unit_now| (unit_last_turn, unit_now))
        })
        .filter(|(_, unit_now)| {
            last_obs.nebulae.contains(&unit_now.pos)
                && unit_now.energy >= fixed_params.min_unit_energy
                && obs.get_opp_units().iter().all(|opp_u| {
                    // Skip units that we think could have been sapped (or adjacent sapped)
                    let [dx, dy] = opp_u.pos.subtract(unit_now.pos);
                    dx.abs() > params.unit_sap_range + 1
                        || dy.abs() > params.unit_sap_range + 1
                })
        })
        .filter_map(|(unit_last_turn, unit_now)| {
            let energy_after_action = match last_actions[unit_last_turn.id] {
                NoOp => unit_last_turn.energy,
                Up | Right | Down | Left => {
                    unit_last_turn.energy - params.unit_move_cost
                },
                Sap(_) => unit_last_turn.energy - params.unit_sap_cost,
            };
            let energy_before_nebula = obs.energy_field
                [unit_now.pos.as_index()]
            .map(|energy_field| energy_after_action + energy_field);
            energy_before_nebula.map(|energy_before_nebula| {
                (energy_before_nebula, unit_now.energy)
            })
        })
    {
        for (&energy_loss, mask) in nebula_tile_energy_reduction_options
            .iter_options_mut_mask()
            .filter(|(_, mask)| **mask)
        {
            if (energy_before_nebula - energy_loss)
                .min(fixed_params.max_unit_energy)
                .max(fixed_params.min_unit_energy)
                != actual
            {
                *mask = false;
            }
        }
    }

    if nebula_tile_energy_reduction_options.all_masked() {
        let unit_energy_field = obs
            .get_my_units()
            .iter()
            .map(|u| obs.energy_field[u.pos.as_index()])
            .collect_vec();
        // TODO: For game-time build, don't panic and instead just fail to update mask
        panic!(
            "nebula_tile_energy_reduction_mask is all false.
            Units last turn: {:?}
            Units now: {:?}
            Nebulae: {:?}
            Energy field: {:?}",
            last_obs.my_units,
            obs.get_my_units(),
            last_obs.nebulae,
            unit_energy_field,
        )
    }
}

fn compute_sap_count_maps(
    units_last_turn: &[Unit],
    last_actions: &[Action],
    map_size: [usize; 2],
) -> (BTreeMap<Pos, i32>, BTreeMap<Pos, i32>) {
    // NB: Assumes that all units that tried to sap had enough energy and were successful
    let mut sap_count = BTreeMap::new();
    let mut adjacent_sap_count = BTreeMap::new();
    for sap_target_pos in units_last_turn.iter().filter_map(|u| {
        if let Sap(sap_deltas) = last_actions[u.id] {
            Some(
                u.pos
                    .maybe_translate(sap_deltas, map_size)
                    .expect("Invalid sap_deltas"),
            )
        } else {
            None
        }
    }) {
        *sap_count.entry(sap_target_pos).or_insert(0) += 1;
        for adjacent_pos in
            (-1..=1).cartesian_product(-1..=1).filter_map(|(dx, dy)| {
                if dx == 0 && dy == 0 {
                    None
                } else {
                    sap_target_pos.maybe_translate([dx, dy], map_size)
                }
            })
        {
            *adjacent_sap_count.entry(adjacent_pos).or_insert(0) += 1;
        }
    }
    (sap_count, adjacent_sap_count)
}

fn determine_unit_sap_dropoff_factor(
    unit_sap_dropoff_factor: &mut MaskedPossibilities<f32>,
    obs: &Observation,
    last_obs_data: &LastObservationData,
    sap_count_maps: &(BTreeMap<Pos, i32>, BTreeMap<Pos, i32>),
    params: &KnownVariableParams,
) {
    // Note that the environment resolution order goes:
    // - Move units
    // - Resolve sap actions
    // - Resolve energy field
    // - Move nebulae and energy nodes
    // Therefore, whenever we are comparing energies, we take the current unit's energy and
    // position and compare it to last turn's nebulae and energy field, minus any sap actions.
    //  Confusingly enough, last turn's energy field is provided in this turn's observation.
    let (sap_count_map, adjacent_sap_count_map) = sap_count_maps;
    // NB: Assumes that units don't take invalid energy-wasting actions, like moving off the map
    let id_to_opp_unit_now: BTreeMap<usize, Unit> =
        obs.get_opp_units().iter().map(|u| (u.id, *u)).collect();
    for (opp_unit_last_turn, opp_unit_now, adj_sap_count, energy_field_delta) in
        last_obs_data
            .opp_units
            .iter()
            .filter_map(|u_last_turn| {
                id_to_opp_unit_now
                    .get(&u_last_turn.id)
                    .map(|u_now| (u_last_turn, u_now))
            })
            // Skip units in nebulae
            .filter(|(_, u_now)| !last_obs_data.nebulae.contains(&u_now.pos))
            .filter_map(|(u_last_turn, u_now)| {
                adjacent_sap_count_map
                    .get(&u_now.pos)
                    .map(|&count| (u_last_turn, u_now, count))
            })
            // Skip units that have lost energy to energy void
            // NB: This won't always filter correctly if dead units are removed from observation,
            //  since they could apply an energy void and die in the same step
            .filter(|(_, opp_unit_now, _)| {
                obs.get_my_units().iter().all(|my_unit| {
                    !ENERGY_VOID_DELTAS
                        .contains(&opp_unit_now.pos.subtract(my_unit.pos))
                })
            })
            .filter_map(|(u_last_turn, u_now, sap_count)| {
                obs.energy_field[u_now.pos.as_index()].map(|energy_delta| {
                    (u_last_turn, u_now, sap_count, energy_delta)
                })
            })
    {
        let direct_sap_loss = sap_count_map
            .get(&opp_unit_now.pos)
            .map_or(0, |count| *count * params.unit_sap_cost);
        let energy_before_action =
            opp_unit_last_turn.energy - direct_sap_loss + energy_field_delta;
        for (&sap_dropoff_factor, mask) in unit_sap_dropoff_factor
            .iter_options_mut_mask()
            .filter(|(_, mask)| **mask)
        {
            let adj_sap_loss = ((adj_sap_count * params.unit_sap_cost) as f32
                * sap_dropoff_factor) as i32;
            if opp_unit_now.pos.subtract(opp_unit_last_turn.pos) == [0, 0] {
                // NoOp or Sap action was taken
                let expected_energy_noop = energy_before_action - adj_sap_loss;
                let expected_energy_sap =
                    energy_before_action - params.unit_sap_cost - adj_sap_loss;
                if expected_energy_noop != opp_unit_now.energy
                    && expected_energy_sap != opp_unit_now.energy
                {
                    *mask = false;
                }
            } else {
                // Move action was taken
                let expected_energy =
                    energy_before_action - params.unit_move_cost - adj_sap_loss;
                if expected_energy != opp_unit_now.energy {
                    *mask = false;
                }
            }
        }
    }

    if unit_sap_dropoff_factor.all_masked() {
        // TODO: For game-time build, don't panic and instead just fail to update mask
        panic!("unit_sap_dropoff_factor_mask is all false")
    }
}

// fn determine_unit_energy_void_factor(
//     unit_energy_void_factor: &mut MaskedPossibilities<f32>,
//     unit_sap_dropoff_factor: Option<f32>,
//     obs: &Observation,
//     last_obs_data: &LastObservationData,
//     sap_count_maps: &(BTreeMap<Pos, i32>, BTreeMap<Pos, i32>),
//     params: &KnownVariableParams,
// ) {
//     let (sap_count_map, adjacent_sap_count_map) = sap_count_maps;
//     let opp_unit_count_map = get_unit_counts_map(obs.get_opp_units());
//     let id_to_opp_unit: BTreeMap<usize, Unit> =
//         obs.get_opp_units().iter().map(|u| (u.id, *u)).collect();
//     let my_units = obs.get_my_units();
//     for (opp_unit_last_turn, opp_unit_now, energy_void_base) in last_obs_data
//         .opp_units_last_turn
//         .iter()
//         .filter_map(|u_last_turn| {
//             id_to_opp_unit
//                 .get(&u_last_turn.id)
//                 .map(|u_now| (u_last_turn, u_now))
//         })
//         // NB: This won't always filter correctly if dead units are removed from observation,
//         //  since they could apply an energy void and die in the same step
//         // TODO: calculate my unit energies after moving but before sapping for energy void base
//         // .filter_map(|(opp_unit_last_turn, opp_unit_now)| {
//         //     my_units.iter().filter(|my_unit| {
//         //         ENERGY_VOID_DELTAS
//         //             .contains(&opp_unit_now.pos.subtract(my_unit.pos))
//         //     }).map(|u| u.energy)
//         // })
//     {}
// }

fn get_unit_counts_map(units: &[Unit]) -> BTreeMap<Pos, u8> {
    let mut result = BTreeMap::new();
    for u in units {
        *result.entry(u.pos).or_insert(0) += 1;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules_engine::params::FIXED_PARAMS;
    use crate::rules_engine::state::{Pos, Unit};
    use numpy::ndarray::arr2;
    use pretty_assertions::assert_eq;
    use rstest::rstest;

    #[test]
    fn test_determine_nebula_tile_vision_reduction() {
        let mut possibilities =
            MaskedPossibilities::from_options(vec![0, 1, 2]);
        let map_size = [3, 3];
        let unit_sensor_range = 1;

        let obs = Observation {
            sensor_mask: arr2(&[
                [true, false, true],
                [false, false, true],
                [true, true, true],
            ]),
            units: [vec![Unit::with_pos(Pos::new(0, 0))], Vec::new()],
            ..Default::default()
        };

        determine_nebula_tile_vision_reduction(
            &mut possibilities,
            &obs,
            map_size,
            unit_sensor_range,
        );
        assert_eq!(possibilities.get_mask(), vec![false, true, true]);
    }

    #[rstest]
    #[case(vec![true, true, true])]
    #[should_panic(expected = "nebula_tile_vision_reduction_mask is all false")]
    #[case(vec![true, true, false])]
    fn test_determine_nebula_tile_vision_reduction_panics(
        #[case] mask: Vec<bool>,
    ) {
        let mut possibilities = MaskedPossibilities::new(vec![0, 1, 2], mask);
        let map_size = [3, 3];
        let unit_sensor_range = 1;

        let obs = Observation {
            sensor_mask: arr2(&[
                [false, false, true],
                [false, false, true],
                [true, true, true],
            ]),
            units: [vec![Unit::with_pos(Pos::new(0, 0))], Vec::new()],
            ..Default::default()
        };

        determine_nebula_tile_vision_reduction(
            &mut possibilities,
            &obs,
            map_size,
            unit_sensor_range,
        );
    }

    #[rstest]
    // Not in nebula
    #[case(
        vec![Unit::new(Pos::new(0, 0), 10, 0)],
        vec![NoOp],
        vec![Unit::new(Pos::new(0, 0), 10, 0)],
        vec![true, true, true, true],
    )]
    // In seen nebula
    #[case(
        vec![Unit::new(Pos::new(1, 3), 10, 0)],
        vec![NoOp],
        vec![Unit::new(Pos::new(1, 3), 7, 0)],
        vec![false, false, true, false],
    )]
    // In seen nebula after move action
    #[case(
        vec![Unit::new(Pos::new(1, 4), 10, 0)],
        vec![Up],
        vec![Unit::new(Pos::new(1, 3), 7, 0)],
        vec![true, false, false, false],
    )]
    // Could be sapped - should be ignored
    #[case(
        vec![Unit::new(Pos::new(3, 3), 10, 0)],
        vec![NoOp],
        vec![Unit::new(Pos::new(3, 3), 10, 0)],
        vec![true, true, true, true],
    )]
    // Has negative energy - should be ignored
    #[case(
        vec![Unit::new(Pos::new(1, 3), -10, 0)],
        vec![NoOp],
        vec![Unit::new(Pos::new(1, 3), -10, 0)],
        vec![true, true, true, true],
    )]
    // No energy data from last turn - should be ignored
    #[case(
        vec![Unit::new(Pos::new(1, 3), 10, 0)],
        vec![NoOp],
        vec![Unit::new(Pos::new(1, 3), 10, 1)],
        vec![true, true, true, true],
    )]
    // Multiple possibilities remaining for units left with min/max energy
    #[case(
        vec![Unit::new(Pos::new(1, 1), 10, 0)],
        vec![Sap([0, 0])],
        vec![Unit::new(Pos::new(1, 1), 0, 0)],
        vec![false, true, true, true],
    )]
    #[case(
        vec![Unit::new(Pos::new(1, 1), 10, 0), Unit::new(Pos::new(1, 0), 400, 1)],
        vec![Sap([0, 0]), NoOp],
        vec![Unit::new(Pos::new(1, 1), 0, 0), Unit::new(Pos::new(1, 0), 400, 1)],
        vec![false, true, true, false],
    )]
    fn test_determine_nebula_tile_energy_reduction(
        #[case] my_units_last_turn: Vec<Unit>,
        #[case] last_actions: Vec<Action>,
        #[case] my_units: Vec<Unit>,
        #[case] expected_result: Vec<bool>,
    ) {
        let obs = Observation {
            units: [my_units, vec![Unit::with_pos(Pos::new(3, 3))]],
            energy_field: arr2(
                &[[Some(2), Some(1), Some(0), Some(-1), Some(-2), None]; 6],
            ),
            ..Default::default()
        };
        let last_obs = LastObservationData {
            my_units: my_units_last_turn,
            nebulae: vec![
                Pos::new(1, 0),
                Pos::new(1, 1),
                Pos::new(1, 2),
                Pos::new(1, 3),
                Pos::new(1, 4),
                Pos::new(1, 5),
                Pos::new(3, 3),
            ],
            ..Default::default()
        };
        let fixed_params = FIXED_PARAMS;
        let params = KnownVariableParams {
            unit_sap_range: 0,
            ..Default::default()
        };
        let mut possibilities =
            MaskedPossibilities::from_options(vec![0, 1, 2, 10]);
        determine_nebula_tile_energy_reduction(
            &mut possibilities,
            &obs,
            &last_obs,
            &last_actions,
            &fixed_params,
            &params,
        );
        assert_eq!(possibilities.get_mask(), expected_result);
    }

    #[rstest]
    #[case(vec![true, true, true])]
    #[should_panic(expected = "nebula_tile_energy_reduction_mask is all false")]
    #[case(vec![true, true, false])]
    fn test_determine_nebula_tile_energy_reduction_panics(
        #[case] mask: Vec<bool>,
    ) {
        let mut possibilities = MaskedPossibilities::new(vec![0, 1, 2], mask);
        let obs = Observation {
            units: [vec![Unit::new(Pos::new(0, 0), 10, 0)], Vec::new()],
            energy_field: arr2(&[[Some(2)]]),
            ..Default::default()
        };
        let last_obs = LastObservationData {
            my_units: vec![Unit::new(Pos::new(0, 0), 10, 0)],
            nebulae: vec![Pos::new(0, 0)],
            ..Default::default()
        };
        let last_actions = vec![NoOp];
        determine_nebula_tile_energy_reduction(
            &mut possibilities,
            &obs,
            &last_obs,
            &last_actions,
            &FIXED_PARAMS,
            &KnownVariableParams::default(),
        );
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
        let (sap_count, adjacent_sap_count) =
            compute_sap_count_maps(&units, &actions, FIXED_PARAMS.map_size);
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
        compute_sap_count_maps(&units, &actions, FIXED_PARAMS.map_size);
    }

    #[rstest]
    // Not adjacent
    #[case(
        vec![Unit::new(Pos::new(0, 0), 10, 0)],
        vec![Sap([1, 1])],
        vec![Unit::new(Pos::new(1, 1), 100, 0)],
        vec![Unit::new(Pos::new(1, 1), 91, 0)],
        vec![true, true, true],
    )]
    // Adjacent to sap
    #[case(
        vec![Unit::new(Pos::new(4, 4), 10, 0)],
        vec![Sap([-3, -3])],
        vec![Unit::new(Pos::new(0, 0), 100, 0)],
        vec![Unit::new(Pos::new(0, 0), 90, 0)],
        vec![false, false, true],
    )]
    // Adjacent to sap, rounds down
    #[case(
        vec![Unit::new(Pos::new(4, 4), 10, 0)],
        vec![Sap([-3, -3])],
        vec![Unit::new(Pos::new(0, 0), 100, 0)],
        vec![Unit::new(Pos::new(0, 0), 98, 0)],
        vec![true, false, false],
    )]
    // Adjacent to multiple sap actions
    #[case(
        vec![Unit::new(Pos::new(4, 4), 10, 0), Unit::new(Pos::new(4, 4), 10, 1)],
        vec![Sap([-3, -3]), Sap([-3, -3])],
        vec![Unit::new(Pos::new(0, 0), 100, 0)],
        vec![Unit::new(Pos::new(0, 0), 95, 0)],
        vec![true, false, false],
    )]
    // Ignore units in nebulae
    #[case(
        vec![Unit::new(Pos::new(0, 0), 10, 0)],
        vec![Sap([0, 1])],
        vec![Unit::new(Pos::new(0, 2), 100, 0)],
        vec![Unit::new(Pos::new(0, 2), 75, 0)],
        vec![true, true, true],
    )]
    // Ignore units hit by energy void field
    #[case(
        vec![Unit::new(Pos::new(0, 1), 10, 0)],
        vec![Sap([1, 0])],
        vec![Unit::new(Pos::new(0, 0), 100, 0)],
        vec![Unit::new(Pos::new(0, 0), 90, 0)],
        vec![true, true, true],
    )]
    // Counts move action cost
    #[case(
        vec![Unit::new(Pos::new(4, 4), 10, 0)],
        vec![Sap([-3, -3])],
        vec![Unit::new(Pos::new(0, 1), 100, 0)],
        vec![Unit::new(Pos::new(0, 0), 93, 0)],
        vec![false, true, false],
    )]
    // Considers both NoOp and Sap actions for opposing units
    #[case(
        vec![Unit::new(Pos::new(4, 4), 10, 0), Unit::new(Pos::new(4, 4), 10, 1)],
        vec![Sap([-3, -3]), Sap([-3, -3])],
        vec![Unit::new(Pos::new(0, 0), 100, 0)],
        vec![Unit::new(Pos::new(0, 0), 80, 0)],
        vec![false, true, true],
    )]
    // Hit by direct and adjacent sap
    #[case(
        vec![Unit::new(Pos::new(4, 4), 10, 0), Unit::new(Pos::new(4, 4), 10, 1)],
        vec![Sap([-3, -3]), Sap([-4, -4])],
        vec![Unit::new(Pos::new(0, 0), 100, 0)],
        vec![Unit::new(Pos::new(0, 0), 85, 0)],
        vec![false, true, false],
    )]
    fn test_determine_unit_sap_dropoff_factor(
        #[case] my_units: Vec<Unit>,
        #[case] last_actions: Vec<Action>,
        #[case] opp_units_last_turn: Vec<Unit>,
        #[case] opp_units: Vec<Unit>,
        #[case] expected_result: Vec<bool>,
    ) {
        let mut possibilities =
            MaskedPossibilities::from_options(vec![0.25, 0.5, 1.0]);
        let obs = Observation {
            units: [my_units.clone(), opp_units],
            energy_field: arr2(&[[Some(0), Some(1), Some(2)]; 3]),
            ..Default::default()
        };
        let last_obs_data = LastObservationData {
            my_units,
            opp_units: opp_units_last_turn,
            nebulae: vec![Pos::new(0, 2)],
        };
        let params = KnownVariableParams {
            unit_move_cost: 2,
            unit_sap_cost: 10,
            ..Default::default()
        };
        let sap_count_maps = compute_sap_count_maps(
            &last_obs_data.my_units,
            &last_actions,
            FIXED_PARAMS.map_size,
        );
        determine_unit_sap_dropoff_factor(
            &mut possibilities,
            &obs,
            &last_obs_data,
            &sap_count_maps,
            &params,
        );
        assert_eq!(possibilities.get_mask(), expected_result);
    }

    #[rstest]
    #[case(vec![true, true, true])]
    #[should_panic(expected = "unit_sap_dropoff_factor_mask is all false")]
    #[case(vec![true, true, false])]
    fn test_determine_unit_sap_dropoff_factor_panics(
        #[case] dropoff_mask: Vec<bool>,
    ) {
        let mut possibilities =
            MaskedPossibilities::new(vec![0.25, 0.5, 1.0], dropoff_mask);
        let my_units = vec![Unit::new(Pos::new(4, 4), 10, 0)];
        let obs = Observation {
            units: [my_units.clone(), vec![Unit::new(Pos::new(0, 0), 90, 0)]],
            energy_field: arr2(&[[Some(0), Some(1), Some(2)]; 3]),
            ..Default::default()
        };
        let last_obs_data = LastObservationData {
            my_units,
            opp_units: vec![Unit::new(Pos::new(0, 0), 100, 0)],
            ..Default::default()
        };
        let last_actions = vec![Sap([-3, -3])];
        let params = KnownVariableParams {
            unit_move_cost: 2,
            unit_sap_cost: 10,
            ..Default::default()
        };
        let sap_count_maps = compute_sap_count_maps(
            &last_obs_data.my_units,
            &last_actions,
            FIXED_PARAMS.map_size,
        );
        determine_unit_sap_dropoff_factor(
            &mut possibilities,
            &obs,
            &last_obs_data,
            &sap_count_maps,
            &params,
        );
    }

    #[test]
    fn test_get_unit_counts_map() {
        let units = vec![
            // Handle stacked units
            Unit::with_pos(Pos::new(0, 0)),
            Unit::with_pos(Pos::new(0, 1)),
            Unit::with_pos(Pos::new(0, 0)),
            Unit::with_pos(Pos::new(0, 1)),
            Unit::with_pos(Pos::new(0, 1)),
        ];
        let expected_result =
            BTreeMap::from([(Pos::new(0, 0), 2), (Pos::new(0, 1), 3)]);
        let result = get_unit_counts_map(&units);
        assert_eq!(result, expected_result);
    }
}
