use crate::rules_engine::state::{Observation, Pos};
use itertools::Itertools;
use numpy::ndarray::{Array2, ArrayView2, ArrayViewMut2, Zip};

/// Tracks everything known by a player currently about the energy field
#[derive(Debug)]
pub struct EnergyFieldMemory {
    pub energy_field: Array2<Option<i32>>,
    map_size: [usize; 2],
}

impl EnergyFieldMemory {
    pub fn new(map_size: [usize; 2]) -> Self {
        EnergyFieldMemory {
            energy_field: Array2::default(map_size),
            map_size,
        }
    }

    pub fn update_memory(&mut self, obs: &Observation) {
        let mut new_energy_field = self.energy_field.clone();
        update_energy_field(
            new_energy_field.view_mut(),
            obs.energy_field.view(),
        );
        let err_msg =
            "Missing new_energy where there was observed energy last turn";
        if Zip::from(&new_energy_field).and(&self.energy_field).all(
            |new_energy, energy_last_turn| {
                energy_last_turn.is_none_or(|e| e == new_energy.expect(err_msg))
            },
        ) {
            // If there are no conflicts between this turn and last turns non-null values,
            // then we're good to update and move on
            self.energy_field = new_energy_field;
        } else {
            // If there are any conflicts, that means that some of the energy nodes have moved,
            // so we reset the known energy field and start over from the current observation
            self.energy_field.fill(None);
            update_energy_field(
                self.energy_field.view_mut(),
                obs.energy_field.view(),
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
    let map_size = [energy_field.nrows(), energy_field.ncols()];
    for pos in (0..energy_field.nrows())
        .cartesian_product(0..energy_field.ncols())
        .map(|(x, y)| Pos::new(x, y))
    {
        if let Some(e) = energy_field[pos.as_index()] {
            energy_field[pos.reflect(map_size).as_index()].get_or_insert(e);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::arr2;
    use rstest::rstest;

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
    fn test_update_memory(
        #[case] known_energy_field: Array2<Option<i32>>,
        #[case] obs_energy_field: Array2<Option<i32>>,
        #[case] expected_result: Array2<Option<i32>>,
    ) {
        let mut memory = EnergyFieldMemory::new([4, 4]);
        memory.energy_field = known_energy_field;
        let obs = Observation {
            energy_field: obs_energy_field,
            ..Default::default()
        };

        memory.update_memory(&obs);
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
}
