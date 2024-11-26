use crate::rules_engine::state::Observation;
use itertools::Itertools;
use numpy::ndarray::{ArrayViewMut1, ArrayViewMut2, ArrayViewMut3};

const UNIT_ENERGY_NORM: f32 = 400.0;

pub fn write_unit_features(
    mut unit_indices: ArrayViewMut3<isize>,
    mut unit_energies: ArrayViewMut2<f32>,
    mut units_mask: ArrayViewMut2<bool>,
    observations: &[Observation; 2],
) {
    for (((obs, unit_indices), unit_energies), units_mask) in observations
        .iter()
        .zip_eq(unit_indices.outer_iter_mut())
        .zip_eq(unit_energies.outer_iter_mut())
        .zip_eq(units_mask.outer_iter_mut())
    {
        write_team_unit_features(unit_indices, unit_energies, units_mask, obs);
    }
}

fn write_team_unit_features(
    mut unit_indices: ArrayViewMut2<isize>,
    mut unit_energies: ArrayViewMut1<f32>,
    mut units_mask: ArrayViewMut1<bool>,
    obs: &Observation,
) {
    for unit in obs.get_my_units().iter().filter(|u| u.alive()) {
        unit_indices[[unit.id, 0]] = unit.pos.x as isize;
        unit_indices[[unit.id, 1]] = unit.pos.y as isize;
        unit_energies[unit.id] = unit.energy as f32 / UNIT_ENERGY_NORM;
        units_mask[unit.id] = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules_engine::params::FIXED_PARAMS;
    use crate::rules_engine::state::{Pos, Unit};
    use numpy::ndarray::{arr2, arr3, Array2, Array3};

    #[test]
    fn test_write_unit_features() {
        let n_units = 4;
        let mut unit_indices = Array3::zeros((2, n_units, 2));
        let mut unit_energies = Array2::zeros((2, n_units));
        let mut units_mask = Array2::default((2, n_units));
        let units = [
            vec![
                // Normal case
                Unit::new(Pos::new(1, 1), 100, 0),
                // Keep units with no energy
                Unit::new(Pos::new(1, 1), 0, 1),
                // Ignore dead units
                Unit::new(Pos::new(1, 1), -1, 2),
                // Max energy
                Unit::new(Pos::new(2, 3), FIXED_PARAMS.max_unit_energy, 3),
            ],
            vec![
                // Ignore dead units
                Unit::new(Pos::new(1, 1), -100, 1),
                // Low energy
                Unit::new(Pos::new(4, 2), 1, 2),
            ],
        ];
        let obs = [
            Observation {
                units: units.clone(),
                team_id: 0,
                ..Default::default()
            },
            Observation {
                units,
                team_id: 1,
                ..Default::default()
            },
        ];

        write_unit_features(
            unit_indices.view_mut(),
            unit_energies.view_mut(),
            units_mask.view_mut(),
            &obs,
        );
        let expected_unit_indices = arr3(&[
            [[1, 1], [1, 1], [0, 0], [2, 3]],
            [[0, 0], [0, 0], [4, 2], [0, 0]],
        ]);
        let expected_unit_energies = arr2(&[
            [0.25, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0 / UNIT_ENERGY_NORM, 0.0],
        ]);
        let expected_units_mask =
            arr2(&[[true, true, false, true], [false, false, true, false]]);
        assert_eq!(unit_indices, expected_unit_indices);
        assert_eq!(unit_energies, expected_unit_energies);
        assert_eq!(units_mask, expected_units_mask);
    }
}
