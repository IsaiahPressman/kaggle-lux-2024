use crate::feature_engineering::memory::masked_possibilities::MaskedPossibilities;
use crate::rules_engine::param_ranges::{
    ParamRanges, IRRELEVANT_ENERGY_NODE_DRIFT_SPEED,
};
use crate::rules_engine::state::{Observation, Pos};
use itertools::Itertools;
use numpy::ndarray::{Array2, ArrayView2, ArrayViewMut2, Zip};

/// Tracks everything known by a player currently about the energy field
#[derive(Debug)]
pub struct EnergyFieldMemory {
    pub energy_field: Array2<Option<i32>>,
    pub energy_node_drift_speed: MaskedPossibilities<f32>,
}

impl EnergyFieldMemory {
    pub fn new(param_ranges: &ParamRanges, map_size: [usize; 2]) -> Self {
        let energy_node_drift_speed = MaskedPossibilities::from_options(
            param_ranges
                .energy_node_drift_speed
                .iter()
                .copied()
                .sorted_by(|a, b| a.partial_cmp(b).unwrap())
                .filter(|&speed| speed != IRRELEVANT_ENERGY_NODE_DRIFT_SPEED)
                .dedup()
                .collect(),
        );
        EnergyFieldMemory {
            energy_field: Array2::default(map_size),
            energy_node_drift_speed,
        }
    }

    pub fn update(&mut self, obs: &Observation) {
        let mut new_energy_field = self.energy_field.clone();
        update_energy_field(
            new_energy_field.view_mut(),
            obs.energy_field.view(),
        );
        let err_msg =
            "Missing new_energy where there was observed energy last turn";
        let energy_field_moved = !Zip::from(&new_energy_field)
            .and(&self.energy_field)
            .all(|new_energy, energy_last_turn| {
                energy_last_turn.is_none_or(|e| e == new_energy.expect(err_msg))
            });
        if !energy_field_moved {
            // If there are no conflicts between this turn and last turns
            // non-null values, then we're good to update and move on
            self.energy_field = new_energy_field;
        } else {
            // If there are any conflicts, that means that some of the energy
            // nodes have moved, so we reset the known energy field and start
            // over from the current observation
            self.energy_field.fill(None);
            update_energy_field(
                self.energy_field.view_mut(),
                obs.energy_field.view(),
            );
        }
        // Subtract 2 steps: 1 for the delay in the observed energy field
        // and 1 because the energy field is moved before step is
        // incremented when creating the observation
        //
        // In case the speed was registered incorrectly, re-check the
        // solution if the energy field moved
        if self.energy_node_drift_speed.still_unsolved() || energy_field_moved {
            update_energy_node_drift_speed(
                &mut self.energy_node_drift_speed,
                obs.total_steps.saturating_sub(2),
                energy_field_moved,
            );
        }
    }
}

fn update_energy_field(
    mut known_energy_field: ArrayViewMut2<Option<i32>>,
    obs_energy_field: ArrayView2<Option<i32>>,
) {
    Zip::from(known_energy_field.view_mut())
        .and(obs_energy_field)
        .for_each(|known_energy, &obs_energy| {
            if obs_energy.is_some() {
                *known_energy = obs_energy;
            }
        });

    symmetrize(known_energy_field);
}

fn symmetrize(mut energy_field: ArrayViewMut2<Option<i32>>) {
    let (w, h) = energy_field.dim();
    let map_size = [w, h];
    for pos in (0..energy_field.nrows())
        .cartesian_product(0..energy_field.ncols())
        .map(Pos::from)
    {
        if let Some(e) = energy_field[pos.as_index()] {
            energy_field[pos.reflect(map_size).as_index()].get_or_insert(e);
        }
    }
}

fn update_energy_node_drift_speed(
    energy_node_drift_speed: &mut MaskedPossibilities<f32>,
    step: u32,
    energy_field_moved: bool,
) {
    // Sometimes the step where the energy field moved is missed by a couple steps.
    // In this case, omit updating energy_node_drift_speed
    if !energy_node_drift_speed
        .get_options()
        .iter()
        .any(|&speed| should_drift(step, speed) == energy_field_moved)
    {
        return;
    }

    for (&candidate_speed, mask) in
        energy_node_drift_speed.iter_unmasked_options_mut_mask()
    {
        if should_drift(step, candidate_speed) != energy_field_moved {
            *mask = false;
        }
    }

    if energy_node_drift_speed.all_masked() {
        // Reset the energy_node_drift_speed mask and retry in the failure case
        *energy_node_drift_speed = MaskedPossibilities::from_options(
            energy_node_drift_speed.get_options().to_vec(),
        );
        update_energy_node_drift_speed(
            energy_node_drift_speed,
            step,
            energy_field_moved,
        );
    }
}

fn should_drift(step: u32, speed: f32) -> bool {
    step as f32 * speed % 1.0 == 0.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules_engine::param_ranges::PARAM_RANGES;
    use crate::rules_engine::params::FIXED_PARAMS;
    use numpy::ndarray::arr2;
    use rstest::rstest;

    fn get_drift_schedule(speed: f32) -> Vec<bool> {
        (0..=FIXED_PARAMS.get_max_steps_in_game())
            .map(|step| should_drift(step, speed))
            .collect()
    }

    #[test]
    fn test_all_movement_schedules_included() {
        let memory =
            EnergyFieldMemory::new(&PARAM_RANGES, FIXED_PARAMS.map_size);
        assert!(
            memory.energy_node_drift_speed.mask.len()
                < PARAM_RANGES.energy_node_drift_speed.len()
        );

        for &speed in
            PARAM_RANGES
                .energy_node_drift_speed
                .iter()
                .filter(|&speed| {
                    !memory
                        .energy_node_drift_speed
                        .get_options()
                        .contains(speed)
                })
        {
            assert!(memory
                .energy_node_drift_speed
                .get_options()
                .iter()
                .any(|&candidate_speed| get_drift_schedule(speed)
                    == get_drift_schedule(candidate_speed)))
        }
    }

    #[rstest]
    // The observed field matches what's known - just update
    #[case(
        arr2(&[
            [Some(1), Some(2), None, Some(5)],
            [None; 4],
            [None, None, None, Some(2)],
            [Some(8), None, None, Some(1)],
        ]),
        arr2(&[
            [None, Some(2), Some(3), None],
            [None, Some(4), Some(6), None],
            [None, Some(7), None, None],
            [None; 4],
        ]),
        arr2(&[
            [Some(1), Some(2), Some(3), Some(5)],
            [None, Some(4), Some(6), Some(3)],
            [None, Some(7), Some(4), Some(2)],
            [Some(8), None, None, Some(1)],
        ])
    )]
    // The observed field doesn't match what's known - reset known to None and update
    #[case(
        arr2(&[
            [Some(1), Some(2), None, Some(5)],
            [None; 4],
            [None, None, None, Some(2)],
            [None, None, None, Some(1)],
        ]),
        arr2(&[
            [None, Some(2), None, Some(6)],
            [Some(2), Some(3), None, None],
            [None, None, Some(3), None],
            [None; 4],
        ]),
        arr2(&[
            [None, Some(2), None, Some(6)],
            [Some(2), Some(3), None, None],
            [None, None, Some(3), Some(2)],
            [None, None, Some(2), None],
        ])
    )]
    fn test_update(
        #[case] known_energy_field: Array2<Option<i32>>,
        #[case] obs_energy_field: Array2<Option<i32>>,
        #[case] expected_result: Array2<Option<i32>>,
    ) {
        let mut memory = EnergyFieldMemory::new(&PARAM_RANGES, [4, 4]);
        memory.energy_field = known_energy_field;
        let obs = Observation {
            energy_field: obs_energy_field,
            ..Default::default()
        };

        memory.update(&obs);
        assert_eq!(memory.energy_field, expected_result);
    }

    #[test]
    fn test_update_energy_field() {
        let mut known_energy_field = arr2(&[
            [Some(1), Some(2), None, Some(5)],
            [None; 4],
            [None, None, None, Some(2)],
            [None, None, None, Some(1)],
        ]);
        let obs_energy_field = arr2(&[
            [None, Some(2), None, Some(6)],
            [Some(3), Some(3), Some(4), None],
            [None, None, Some(3), None],
            [None; 4],
        ]);
        let expected_result = arr2(&[
            [Some(1), Some(2), None, Some(6)],
            [Some(3), Some(3), Some(4), None],
            [None, None, Some(3), Some(2)],
            [None, None, Some(3), Some(1)],
        ]);

        update_energy_field(
            known_energy_field.view_mut(),
            obs_energy_field.view(),
        );
        assert_eq!(known_energy_field, expected_result);
    }

    #[rstest]
    #[case(
        arr2(&[
            [Some(1), Some(2), Some(3)],
            [Some(4), Some(5), None],
            [None, None, None],
        ]),
        arr2(&[
            [Some(1), Some(2), Some(3)],
            [Some(4), Some(5), Some(2)],
            [None, Some(4), Some(1)],
        ]),
    )]
    // Reflection should work in either direction
    #[case(
        arr2(&[
            [Some(1), None, Some(3)],
            [None, Some(5), Some(6)],
            [None, None, None],
        ]),
        arr2(&[
            [Some(1), Some(6), Some(3)],
            [None, Some(5), Some(6)],
            [None, None, Some(1)],
        ]),
    )]
    // Reflection should not overwrite conflicts
    #[case(
        arr2(&[
            [Some(1), Some(2), None],
            [Some(4), None, Some(6)],
            [Some(7), Some(8), Some(9)],
        ]),
        arr2(&[
            [Some(1), Some(2), None],
            [Some(4), None, Some(6)],
            [Some(7), Some(8), Some(9)],
        ]),
    )]
    fn test_symmetrize(
        #[case] mut field: Array2<Option<i32>>,
        #[case] expected_result: Array2<Option<i32>>,
    ) {
        symmetrize(field.view_mut());
        assert_eq!(field, expected_result);
    }

    #[rstest]
    #[case(0, true, vec![true; 4])]
    #[case(20, true, vec![false, false, false, true])]
    #[case(20, false, vec![true, true, true, false])]
    #[case(25, true, vec![false, false, true, false])]
    #[case(25, false, vec![true, true, false, true])]
    #[case(50, true, vec![false, true, true, false])]
    #[case(50, false, vec![true, false, false, true])]
    #[case(100, true, vec![true; 4])]
    fn test_update_energy_node_drift_speed(
        #[case] step: u32,
        #[case] energy_field_moved: bool,
        #[case] expected_result: Vec<bool>,
    ) {
        let mut energy_node_drift_speed =
            EnergyFieldMemory::new(&PARAM_RANGES, [24, 24])
                .energy_node_drift_speed;
        update_energy_node_drift_speed(
            &mut energy_node_drift_speed,
            step,
            energy_field_moved,
        );
        assert_eq!(
            energy_node_drift_speed.get_mask().to_vec(),
            expected_result
        );
    }

    #[rstest]
    #[case([true, true, true, true])]
    #[case([true, true, true, false])]
    fn test_update_energy_node_drift_speed_resets(#[case] mask: [bool; 4]) {
        let mut energy_node_drift_speed =
            EnergyFieldMemory::new(&PARAM_RANGES, [24, 24])
                .energy_node_drift_speed;
        energy_node_drift_speed.mask = mask.to_vec();
        update_energy_node_drift_speed(&mut energy_node_drift_speed, 120, true);
        assert_eq!(
            energy_node_drift_speed.mask,
            vec![false, false, false, true]
        );
    }
}
