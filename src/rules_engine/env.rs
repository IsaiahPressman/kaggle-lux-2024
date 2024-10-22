use super::action::Action;
use super::params::Params;
use super::state::State;
use itertools::zip_eq;

pub fn step(
    state: &mut State,
    actions: &[Vec<Action>; 2],
    params: &Params,
    energy_node_deltas: Option<Vec<[usize; 2]>>,
) {
    // TODO: Do we need to precompute energy features
    if state.match_steps == 0 {
        state.units = [Vec::new(), Vec::new()];
    }
    remove_dead_units(state);
    move_units(state, actions, params);
    sap_units(state, actions, params);
    unimplemented!();
}

fn remove_dead_units(state: &mut State) {
    state.units[0].retain(|u| u.energy >= 0);
    state.units[1].retain(|u| u.energy >= 0);
}

fn move_units(state: &mut State, actions: &[Vec<Action>; 2], params: &Params) {
    let asteroid_mask = state.get_asteroid_mask(params.map_size);
    for p in [0, 1] {
        for (u, a) in zip_eq(state.units[p].iter_mut(), actions[p].iter()) {
            if u.energy < params.unit_move_cost {
                continue;
            }
            let deltas = match a {
                Action::Up => [0, -1],
                Action::Right => [1, 0],
                Action::Down => [0, 1],
                Action::Left => [-1, 0],
                Action::NoOp | Action::Sap(_) => continue,
            };
            let new_pos = u.pos.bounded_translate(deltas, params.map_size);
            if asteroid_mask[new_pos.as_index()] {
                continue;
            }
            u.pos = new_pos;
            u.energy -= params.unit_move_cost;
        }
    }
}

fn sap_units(state: &mut State, actions: &[Vec<Action>; 2], params: &Params) {
    let energy_before_sapping: [Vec<i32>; 2] = [
        state.units[0].iter().map(|u| u.energy).collect(),
        state.units[1].iter().map(|u| u.energy).collect(),
    ];
    for [p, opp] in [[0, 1], [1, 0]] {
        let mut sap_count = vec![0; state.units[opp].len()];
        let mut adjacent_sap_count = vec![0; state.units[opp].len()];
        for ((unit_idx, &cur_energy), a) in zip_eq(
            energy_before_sapping[p].iter().enumerate(),
            actions[p].iter(),
        ) {
            if cur_energy < params.unit_sap_cost {
                continue;
            }
            let Action::Sap(sap_deltas) = *a else {
                continue;
            };
            if sap_deltas[0] > params.unit_sap_range || sap_deltas[1] > params.unit_sap_range {
                continue;
            }
            let u = &mut state.units[p][unit_idx];
            let Some(target_pos) = u.pos.maybe_translate(sap_deltas, params.map_size) else {
                continue;
            };
            u.energy -= params.unit_sap_cost;
            for (i, opp_u) in state.units[opp].iter().enumerate() {
                match target_pos.subtract(opp_u.pos) {
                    [0, 0] => sap_count[i] += 1,
                    [-1..=1, -1..=1] => adjacent_sap_count[i] += 1,
                    _ => continue,
                }
            }
        }

        for ((saps, adj_saps), opp_u) in zip_eq(
            zip_eq(sap_count, adjacent_sap_count),
            state.units[opp].iter_mut(),
        ) {
            opp_u.energy -= saps * params.unit_sap_cost;
            let adj_sap_loss =
                (adj_saps * params.unit_sap_cost) as f64 * params.unit_sap_dropoff_factor;
            opp_u.energy -= adj_sap_loss as i32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules_engine::state::{Pos, Unit};

    #[test]
    fn test_move_units() {
        let params = Params::default();
        let mut state = State::empty();
        state.units = [
            vec![
                // Unit can't move without energy, costs no energy
                Unit::new(Pos::new(0, 0), params.unit_move_cost - 1),
                // Unit can't move off the bottom of the map, but still costs energy
                Unit::new(Pos::new(0, 0), 100),
                // Unit moves normally
                Unit::new(Pos::new(0, 0), 100),
            ],
            vec![
                // Unit can't move off the top of the map, but still costs energy
                Unit::new(Pos::new(23, 23), 100),
                // Unit can't move into an asteroid, costs no energy
                Unit::new(Pos::new(23, 23), 100),
            ],
        ];
        state.asteroids.push(Pos::new(23, 22));
        let actions = [
            vec![Action::Left, Action::Left, Action::Right],
            vec![Action::Down, Action::Up],
        ];
        let expected_moved_units = [
            vec![
                Unit::new(Pos::new(0, 0), params.unit_move_cost - 1),
                Unit::new(Pos::new(0, 0), 100 - params.unit_move_cost),
                Unit::new(Pos::new(1, 0), 100 - params.unit_move_cost),
            ],
            vec![
                Unit::new(Pos::new(23, 23), 100 - params.unit_move_cost),
                Unit::new(Pos::new(23, 23), 100),
            ],
        ];
        move_units(&mut state, &actions, &params);
        assert_eq!(state.units, expected_moved_units);
    }

    #[test]
    fn test_sap_units() {
        let mut params = Params::default();
        let sap_cost = 5;
        params.unit_sap_cost = sap_cost;
        let mut state = State::empty();
        state.units = [
            vec![
                // Can't sap off the edge of the map, costs no energy
                Unit::new(Pos::new(0, 0), 100),
                // Can't sap out of range, costs no energy
                Unit::new(Pos::new(1, 1), 100),
                // Can't sap without enough energy
                Unit::new(Pos::new(2, 2), sap_cost - 1),
                // Sap should work normally, hit all adjacent units, and not hit allied units
                Unit::new(Pos::new(1, 2), 100),
                // Sap should work normally at max range
                Unit::new(
                    Pos::new(
                        1 + params.unit_sap_range as usize,
                        1 + params.unit_sap_range as usize,
                    ),
                    100,
                ),
            ],
            vec![
                Unit::new(Pos::new(0, 0), 100),
                Unit::new(Pos::new(1, 1), 100),
                Unit::new(Pos::new(2, 2), 100),
            ],
        ];
        let actions = [
            vec![
                Action::Sap([-1, -1]),
                Action::Sap([params.unit_sap_range + 1, 0]),
                Action::Sap([0, 0]),
                Action::Sap([0, 0]),
                Action::Sap([-params.unit_sap_range, -params.unit_sap_range]),
            ],
            vec![Action::NoOp, Action::NoOp, Action::NoOp],
        ];

        let expected_sapped_units = [
            vec![
                Unit::new(Pos::new(0, 0), 100),
                Unit::new(Pos::new(1, 1), 100),
                Unit::new(Pos::new(2, 2), sap_cost - 1),
                Unit::new(Pos::new(1, 2), 100 - sap_cost),
                Unit::new(
                    Pos::new(
                        1 + params.unit_sap_range as usize,
                        1 + params.unit_sap_range as usize,
                    ),
                    100 - sap_cost,
                ),
            ],
            vec![
                Unit::new(
                    Pos::new(0, 0),
                    100 - (sap_cost as f64 * params.unit_sap_dropoff_factor) as i32,
                ),
                Unit::new(
                    Pos::new(1, 1),
                    100 - sap_cost - (sap_cost as f64 * params.unit_sap_dropoff_factor) as i32,
                ),
                Unit::new(
                    Pos::new(2, 2),
                    100 - (sap_cost as f64 * params.unit_sap_dropoff_factor * 2.) as i32,
                ),
            ],
        ];
        sap_units(&mut state, &actions, &params);
        assert_eq!(state.units, expected_sapped_units);
    }
}
