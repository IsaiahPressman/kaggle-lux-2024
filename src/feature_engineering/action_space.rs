use crate::rules_engine::action::Action;
use crate::rules_engine::action::Action::{Down, Left, NoOp, Right, Sap, Up};
use crate::rules_engine::params::{FixedParams, KnownVariableParams};
use crate::rules_engine::state::{Observation, Unit};
use itertools::Itertools;
use numpy::ndarray::{
    s, Array2, ArrayViewMut2, ArrayViewMut3, ArrayViewMut4, Axis, Zip,
};
use strum::IntoEnumIterator;

/// Writes into action_mask of shape (teams, n_units, n_actions) and
/// sap_mask of shape (teams, n_units, map_width, map_height)
pub fn write_basic_action_space(
    action_mask: &mut ArrayViewMut3<bool>,
    sap_mask: &mut ArrayViewMut4<bool>,
    observations: &[Observation; 2],
    params: &KnownVariableParams,
    fixed_params: &FixedParams,
) {
    for ((obs, team_action_mask), team_sap_mask) in observations
        .iter()
        .zip_eq(action_mask.axis_iter_mut(Axis(0)))
        .zip_eq(sap_mask.axis_iter_mut(Axis(0)))
    {
        write_team_actions(
            team_action_mask,
            team_sap_mask,
            obs,
            params,
            fixed_params,
        );
    }
}

fn write_team_actions(
    mut action_mask: ArrayViewMut2<bool>,
    mut sap_mask: ArrayViewMut3<bool>,
    obs: &Observation,
    params: &KnownVariableParams,
    fixed_params: &FixedParams,
) {
    let sap_targets_map = get_sap_targets_map(obs, fixed_params.map_size);
    for unit in obs.get_my_units() {
        let can_sap = write_sap_mask(
            sap_mask.index_axis_mut(Axis(0), unit.id),
            &sap_targets_map,
            *unit,
            params,
            fixed_params,
        );
        for (action, is_legal) in Action::iter()
            .zip_eq(action_mask.index_axis_mut(Axis(0), unit.id).iter_mut())
        {
            match action {
                NoOp => {
                    *is_legal = true;
                },
                Up | Right | Down | Left => {
                    if unit.energy < params.unit_move_cost {
                        continue;
                    };
                    let Some(new_pos) = unit.pos.maybe_translate(
                        action.as_move_delta(),
                        fixed_params.map_size,
                    ) else {
                        continue;
                    };
                    if obs.asteroids.contains(&new_pos) {
                        continue;
                    }

                    *is_legal = true;
                },
                Sap(_) => {
                    *is_legal = can_sap;
                },
            }
        }
    }
}

fn write_sap_mask(
    mut sap_mask: ArrayViewMut2<bool>,
    sap_targets_map: &Array2<bool>,
    unit: Unit,
    params: &KnownVariableParams,
    fixed_params: &FixedParams,
) -> bool {
    if unit.energy < params.unit_sap_cost {
        return false;
    }

    let mut unit_can_sap = false;
    let [x, y]: [usize; 2] = unit.pos.into();
    let range = params.unit_sap_range as usize;
    let slice = s![
        x.saturating_sub(range)..=(x + range).min(fixed_params.map_width - 1),
        y.saturating_sub(range)..=(y + range).min(fixed_params.map_height - 1),
    ];
    Zip::from(sap_mask.slice_mut(slice))
        .and(sap_targets_map.slice(slice))
        .for_each(|sap_mask, &legal_sap| {
            *sap_mask = legal_sap;
            unit_can_sap = unit_can_sap || legal_sap;
        });
    unit_can_sap
}

fn get_sap_targets_map(
    obs: &Observation,
    map_size: [usize; 2],
) -> Array2<bool> {
    let mut sap_targets = Array2::default(map_size);
    let [width, height] = map_size;
    for unit in obs.get_opp_units() {
        let [x, y]: [usize; 2] = unit.pos.into();
        let slice = s![
            x.saturating_sub(1)..=(x + 1).min(width - 1),
            y.saturating_sub(1)..=(y + 1).min(height - 1),
        ];
        sap_targets.slice_mut(slice).fill(true);
    }
    sap_targets
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules_engine::params::FIXED_PARAMS;
    use crate::rules_engine::state::Pos;
    use numpy::ndarray::{arr2, arr3, Array3};
    use pretty_assertions::assert_eq;
    use strum::EnumCount;

    #[test]
    fn test_write_team_actions() {
        let mut variable_params = KnownVariableParams::default();
        variable_params.unit_move_cost = 1;
        variable_params.unit_sap_cost = 5;
        variable_params.unit_sap_range = 1;
        let mut fixed_params = FIXED_PARAMS;
        fixed_params.max_units = 8;
        fixed_params.set_map_size([5, 5]);
        let mut action_mask =
            Array2::default((fixed_params.max_units, Action::COUNT));
        let mut sap_mask = Array3::default((
            fixed_params.max_units,
            fixed_params.map_width,
            fixed_params.map_height,
        ));
        let mut obs = Observation::default();
        obs.units = [
            vec![
                // Can't move UP or LEFT, not in range to sap
                Unit::new(Pos::new(0, 0), 100, 0),
                // Can't move DOWN or RIGHT, can sap
                Unit::new(Pos::new(4, 4), 100, 1),
                // No units with ID 2
                // Not enough energy to sap or move
                Unit::new(Pos::new(1, 1), 0, 3),
                // Not enough energy to sap, blocked some by asteroid
                Unit::new(Pos::new(1, 1), variable_params.unit_move_cost, 4),
                // Can sap, but barely reaches, blocked some by asteroid
                Unit::new(Pos::new(2, 2), 100, 5),
                // Can sap, not blocked
                Unit::new(Pos::new(3, 3), 100, 6),
            ],
            vec![
                // One visible sap target
                Unit::with_pos(Pos::new(3, 4)),
            ],
        ];
        obs.asteroids = vec![Pos::new(1, 2)];

        write_team_actions(
            action_mask.view_mut(),
            sap_mask.view_mut(),
            &obs,
            &variable_params,
            &fixed_params,
        );
        let expected_action_mask = arr2(&[
            [true, false, true, true, false, false],
            [true, true, false, false, true, true],
            [false; 6],
            [true, false, false, false, false, false],
            [true, true, true, false, true, false],
            [true, true, true, true, false, true],
            [true; 6],
            [false; 6],
        ]);
        let expected_sap_mask = arr3(&[
            [[false; 5]; 5],
            [
                [false; 5],
                [false; 5],
                [false; 5],
                [false, false, false, true, true],
                [false, false, false, true, true],
            ],
            [[false; 5]; 5],
            [[false; 5]; 5],
            [[false; 5]; 5],
            [
                [false; 5],
                [false; 5],
                [false, false, false, true, false],
                [false, false, false, true, false],
                [false; 5],
            ],
            [
                [false; 5],
                [false; 5],
                [false, false, false, true, true],
                [false, false, false, true, true],
                [false, false, false, true, true],
            ],
            [[false; 5]; 5],
        ]);
        assert_eq!(action_mask, expected_action_mask);
        assert_eq!(sap_mask, expected_sap_mask);
    }

    #[test]
    fn test_get_sap_targets_map() {
        let mut obs = Observation::default();
        let map_size = [5, 5];
        obs.units = [
            Vec::new(),
            vec![
                Unit::with_pos(Pos::new(0, 0)),
                Unit::with_pos(Pos::new(0, 1)),
                Unit::with_pos(Pos::new(2, 3)),
            ],
        ];
        let sap_targets_mask = get_sap_targets_map(&obs, map_size);
        let expected_sap_targets_mask = arr2(&[
            [true, true, true, false, false],
            [true, true, true, true, true],
            [false, false, true, true, true],
            [false, false, true, true, true],
            [false, false, false, false, false],
        ]);
        assert_eq!(sap_targets_mask, expected_sap_targets_mask);
    }
}
