use super::action::Action;
use super::params::Params;
use super::state::{EnergyNode, Pos, State, Unit};
use itertools::{zip_eq, Itertools};
use numpy::ndarray::{Array2, Array3};
use std::cmp::{max, min};

pub fn step(
    state: &mut State,
    actions: &[Vec<Action>; 2],
    params: &Params,
    energy_node_deltas: Option<Vec<[usize; 2]>>,
) {
    if state.match_steps == 0 {
        state.units = [Vec::new(), Vec::new()];
    }
    remove_dead_units(&mut state.units);
    move_units(state, actions, params);
    let energy_before_sapping = get_unit_energies(&state.units);
    sap_units(&mut state.units, &energy_before_sapping, actions, params);
    resolve_collisions_and_energy_void_fields(
        &mut state.units,
        &energy_before_sapping,
        params,
    );
    apply_energy_field(state, params);
    unimplemented!("spawn_units_in");
}

fn remove_dead_units(units: &mut [Vec<Unit>; 2]) {
    units[0].retain(|u| u.energy >= 0);
    units[1].retain(|u| u.energy >= 0);
}

fn move_units(state: &mut State, actions: &[Vec<Action>; 2], params: &Params) {
    let asteroid_mask = get_map_mask(&state.asteroids, params.map_size);
    for (u, a) in zip_eq(state.units.iter_mut(), actions.iter())
        .flat_map(|(us, acts)| zip_eq(us, acts))
        .filter(|(u, _)| u.energy >= params.unit_move_cost)
    {
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

fn get_map_mask(positions: &[Pos], map_size: [usize; 2]) -> Array2<bool> {
    let mut result = Array2::default(map_size);
    for a in positions.iter() {
        result[a.as_index()] = true;
    }
    result
}

fn get_unit_energies(units: &[Vec<Unit>; 2]) -> [Vec<i32>; 2] {
    [
        units[0].iter().map(|u| u.energy).collect(),
        units[1].iter().map(|u| u.energy).collect(),
    ]
}

fn sap_units(
    units: &mut [Vec<Unit>; 2],
    unit_energies: &[Vec<i32>; 2],
    actions: &[Vec<Action>; 2],
    params: &Params,
) {
    for (t, opp) in [(0, 1), (1, 0)] {
        let mut sap_count = vec![0; units[opp].len()];
        let mut adjacent_sap_count = vec![0; units[opp].len()];
        for (unit_idx, sap_deltas) in zip_eq(
            unit_energies[t].iter().copied().enumerate(),
            actions[t].iter(),
        )
        .filter(|((_, cur_energy), _)| *cur_energy >= params.unit_sap_cost)
        .filter_map(|((unit_idx, _), a)| {
            if let Action::Sap(sap_deltas) = *a {
                Some((unit_idx, sap_deltas))
            } else {
                None
            }
        })
        .filter(|(_, [dx, dy])| {
            *dx <= params.unit_sap_range && *dy <= params.unit_sap_range
        }) {
            let u = &mut units[t][unit_idx];
            let Some(target_pos) =
                u.pos.maybe_translate(sap_deltas, params.map_size)
            else {
                continue;
            };
            u.energy -= params.unit_sap_cost;
            for (i, opp_u) in units[opp].iter().enumerate() {
                match target_pos.subtract(opp_u.pos) {
                    [0, 0] => sap_count[i] += 1,
                    [-1..=1, -1..=1] => adjacent_sap_count[i] += 1,
                    _ => {},
                }
            }
        }

        for ((saps, adj_saps), opp_u) in
            zip_eq(zip_eq(sap_count, adjacent_sap_count), units[opp].iter_mut())
        {
            opp_u.energy -= saps * params.unit_sap_cost;
            let adj_sap_loss = (adj_saps * params.unit_sap_cost) as f32
                * params.unit_sap_dropoff_factor;
            opp_u.energy -= adj_sap_loss as i32;
        }
    }
}

fn resolve_collisions_and_energy_void_fields(
    units: &mut [Vec<Unit>; 2],
    unit_energies: &[Vec<i32>; 2],
    params: &Params,
) {
    let unit_aggregate_energy_void_map = get_unit_aggregate_energy_void_map(
        units,
        unit_energies,
        params.map_size,
    );
    let unit_counts_map = get_unit_counts_map(units, params.map_size);
    let unit_aggregate_energy_map =
        get_unit_aggregate_energy_map(units, unit_energies, params.map_size);
    for (t, opp) in [(0, 1), (1, 0)] {
        let mut to_remove = Vec::new();
        for (i, u) in units[t].iter_mut().enumerate() {
            let [x, y] = u.pos.as_index();
            // Resolve collision
            if unit_counts_map[[opp, x, y]] > 0
                && unit_aggregate_energy_map[[opp, x, y]]
                    >= unit_aggregate_energy_map[[t, x, y]]
            {
                to_remove.push(i);
            }
            // Apply energy void fields
            u.energy -= (params.unit_energy_void_factor
                * unit_aggregate_energy_void_map[[opp, x, y]]
                / unit_counts_map[[t, x, y]] as f32)
                as i32;
        }
        let mut keep_iter =
            (0..units[t].len()).map(|i| !to_remove.contains(&i));
        units[t].retain(|_| keep_iter.next().unwrap());
    }
}

fn get_unit_aggregate_energy_void_map(
    units: &[Vec<Unit>; 2],
    unit_energies: &[Vec<i32>; 2],
    map_size: [usize; 2],
) -> Array3<f32> {
    let mut result = Array3::zeros((2, map_size[0], map_size[1]));
    for ((t, u, energy), delta) in zip_eq(units, unit_energies)
        .enumerate()
        .flat_map(|(t, (us, ues))| {
            zip_eq(us, ues.iter().copied()).map(move |(u, ue)| (t, u, ue))
        })
        .cartesian_product([[-1, 0], [1, 0], [0, -1], [0, 1]])
    {
        let Some(Pos { x, y }) = u.pos.maybe_translate(delta, map_size) else {
            continue;
        };
        result[[t, x, y]] += energy as f32;
    }
    result
}

fn get_unit_counts_map(
    units: &[Vec<Unit>; 2],
    map_size: [usize; 2],
) -> Array3<u8> {
    let mut result = Array3::zeros((2, map_size[0], map_size[1]));
    for (t, u) in units
        .iter()
        .enumerate()
        .flat_map(|(t, us)| us.iter().map(move |u| (t, u)))
    {
        result[[t, u.pos.x, u.pos.y]] += 1;
    }
    result
}

fn get_unit_aggregate_energy_map(
    units: &[Vec<Unit>; 2],
    unit_energies: &[Vec<i32>; 2],
    map_size: [usize; 2],
) -> Array3<i32> {
    let mut result = Array3::zeros((2, map_size[0], map_size[1]));
    for (t, u, energy) in
        zip_eq(units, unit_energies)
            .enumerate()
            .flat_map(|(t, (us, ues))| {
                zip_eq(us, ues.iter().copied()).map(move |(u, ue)| (t, u, ue))
            })
    {
        result[[t, u.pos.x, u.pos.y]] += energy;
    }
    result
}

fn apply_energy_field(state: &mut State, params: &Params) {
    let energy_field = get_energy_field(&state.energy_nodes);
    let nebula_mask = get_map_mask(&state.nebulae, params.map_size);
    for (u, energy_gain) in state
        .units
        .iter_mut()
        .flat_map(|us| us.iter_mut())
        .map(|u| {
            let u_pos_idx = u.pos.as_index();
            (
                u,
                if nebula_mask[u_pos_idx] {
                    energy_field[u_pos_idx]
                        - params.nebula_tile_energy_reduction
                } else {
                    energy_field[u_pos_idx]
                },
            )
        })
        .filter(|(u, eg)| u.energy >= 0 || u.energy + eg >= 0)
    {
        u.energy = min(
            max(u.energy + energy_gain, params.min_unit_energy),
            params.max_unit_energy,
        )
    }
}

fn get_energy_field(energy_nodes: &[EnergyNode]) -> Array2<i32> {
    // TODO: Left off here
    unimplemented!()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules_engine::state::{Pos, Unit};
    use numpy::ndarray::{arr2, arr3};

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
                Unit::new(Pos::new(23, 23), params.unit_move_cost),
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
                Unit::new(Pos::new(23, 23), 0),
                Unit::new(Pos::new(23, 23), 100),
            ],
        ];
        move_units(&mut state, &actions, &params);
        assert_eq!(state.units, expected_moved_units);
    }

    #[test]
    fn test_get_map_mask() {
        let asteroids = Vec::new();
        let expected_result = arr2(&[[false; 3]; 3]);
        assert_eq!(get_map_mask(&asteroids, [3, 3]), expected_result);

        let nebulae =
            vec![Pos { x: 0, y: 0 }, Pos { x: 0, y: 1 }, Pos { x: 2, y: 1 }];
        let expected_result =
            arr2(&[[true, true], [false, false], [false, true]]);
        assert_eq!(get_map_mask(&nebulae, [3, 2]), expected_result);
    }

    #[test]
    fn test_sap_units() {
        let mut params = Params::default();
        let sap_cost = 5;
        params.unit_sap_cost = sap_cost;
        params.unit_sap_dropoff_factor = 0.5;
        let mut units = [
            vec![
                // Can't sap off the edge of the map, costs no energy
                Unit::new(Pos::new(0, 0), 100),
                // Can't sap out of range, costs no energy
                Unit::new(Pos::new(1, 1), 100),
                // Can't sap without enough energy
                Unit::new(Pos::new(2, 2), sap_cost - 1),
                // Sap should work normally, hit all adjacent units, and not hit allied units
                Unit::new(Pos::new(1, 2), sap_cost),
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
                Unit::new(Pos::new(1, 2), 0),
                Unit::new(
                    Pos::new(
                        1 + params.unit_sap_range as usize,
                        1 + params.unit_sap_range as usize,
                    ),
                    100 - sap_cost,
                ),
            ],
            vec![
                Unit::new(Pos::new(0, 0), 100 - (sap_cost / 2)),
                Unit::new(Pos::new(1, 1), 100 - sap_cost - sap_cost / 2),
                Unit::new(Pos::new(2, 2), 100 - (2 * sap_cost / 2)),
            ],
        ];
        let unit_energies = get_unit_energies(&units);
        sap_units(&mut units, &unit_energies, &actions, &params);
        assert_eq!(units, expected_sapped_units);
    }

    #[test]
    fn test_resolve_collisions_and_energy_void_fields() {
        let mut params = Params::default();
        params.unit_energy_void_factor = 0.25;
        let mut units = [
            vec![
                // Don't collide with self, less energy loses
                Unit::new(Pos::new(0, 0), 1),
                Unit::new(Pos::new(0, 0), 1),
                Unit::new(Pos::new(0, 0), 1),
                // Everyone dies in a tie
                Unit::new(Pos::new(1, 1), 1),
                Unit::new(Pos::new(1, 1), 1),
                // Energy voids are combined/shared
                Unit::new(Pos::new(2, 2), 100),
            ],
            vec![
                // Don't collide with self, more energy wins
                Unit::new(Pos::new(0, 0), 2),
                Unit::new(Pos::new(0, 0), 2),
                // Everyone dies in a tie
                Unit::new(Pos::new(1, 1), 2),
                // Energy voids are combined/shared
                Unit::new(Pos::new(2, 3), 100),
                Unit::new(Pos::new(2, 3), 100),
            ],
        ];
        let expected_result = [
            vec![Unit::new(Pos::new(2, 2), 100 - 25 - 25)],
            vec![
                Unit::new(Pos::new(0, 0), 2),
                Unit::new(Pos::new(0, 0), 2),
                Unit::new(Pos::new(2, 3), 100 - 12),
                Unit::new(Pos::new(2, 3), 100 - 12),
            ],
        ];
        let unit_energies = get_unit_energies(&units);
        resolve_collisions_and_energy_void_fields(
            &mut units,
            &unit_energies,
            &params,
        );
        assert_eq!(units, expected_result);
    }

    #[test]
    fn test_get_unit_aggregate_energy_void_map() {
        let units = [
            vec![
                // Should appropriately handle units at edges of map and sum values
                Unit::new(Pos::new(0, 0), 1),
                Unit::new(Pos::new(1, 1), 2),
            ],
            vec![
                // Should handle different energy amounts and stacked units
                Unit::new(Pos::new(0, 0), 2),
                Unit::new(Pos::new(0, 0), 2),
                Unit::new(Pos::new(0, 1), 30),
            ],
        ];
        let expected_result =
            arr3(&[[[0., 3.], [3., 0.]], [[30., 4.], [4., 30.]]]);
        let unit_energies = get_unit_energies(&units);
        let result =
            get_unit_aggregate_energy_void_map(&units, &unit_energies, [2, 2]);
        assert_eq!(result, expected_result);
    }

    #[test]
    fn test_get_unit_counts_map() {
        let units = [
            vec![
                // Handle stacked units
                Unit::new_at(Pos::new(0, 0)),
                Unit::new_at(Pos::new(0, 0)),
                Unit::new_at(Pos::new(0, 1)),
                Unit::new_at(Pos::new(0, 1)),
                Unit::new_at(Pos::new(0, 1)),
            ],
            vec![
                // Different teams have different stacks
                Unit::new_at(Pos::new(1, 0)),
                Unit::new_at(Pos::new(1, 0)),
                Unit::new_at(Pos::new(1, 1)),
            ],
        ];
        let expected_result = arr3(&[[[2, 3], [0, 0]], [[0, 0], [2, 1]]]);
        let result = get_unit_counts_map(&units, [2, 2]);
        assert_eq!(result, expected_result);
    }

    #[test]
    fn test_get_unit_aggregate_energy_map() {
        let units = [
            vec![
                // Handle stacked units
                Unit::new(Pos::new(0, 0), 3),
                Unit::new(Pos::new(0, 0), 20),
                Unit::new(Pos::new(0, 0), 100),
            ],
            vec![
                // Different teams have different stacks
                Unit::new(Pos::new(0, 1), 40),
                Unit::new(Pos::new(0, 1), 5),
                Unit::new(Pos::new(1, 0), 67),
            ],
        ];
        let expected_result = arr3(&[[[123, 0], [0, 0]], [[0, 45], [67, 0]]]);
        let unit_energies = get_unit_energies(&units);
        let result =
            get_unit_aggregate_energy_map(&units, &unit_energies, [2, 2]);
        assert_eq!(result, expected_result);
    }

    #[test]
    fn test_apply_energy_field() {
        unimplemented!()
    }

    #[test]
    fn test_get_energy_field() {
        unimplemented!()
    }
}
