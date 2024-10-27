use super::action::Action;
use super::params::Params;
use super::state::{EnergyNode, GameResult, Observation, Pos, State, Unit};
use itertools::Itertools;
use numpy::ndarray::{s, Array2, Array3, Axis};
use rand::distributions::{Distribution, Uniform};
use rand::prelude::ThreadRng;
use std::cmp::{max, min};

pub fn step(
    state: &mut State,
    rng: &mut ThreadRng,
    actions: &[Vec<Action>; 2],
    params: &Params,
    energy_node_deltas: Option<Vec<[isize; 2]>>,
) -> ([Observation; 2], GameResult) {
    if state.match_steps == 0 {
        state.units = [Vec::new(), Vec::new()];
    }
    remove_dead_units(&mut state.units);
    move_units(
        &mut state.units,
        &get_map_mask(&state.asteroids, params.map_size),
        actions,
        params,
    );
    let energy_before_sapping = get_unit_energies(&state.units);
    sap_units(&mut state.units, &energy_before_sapping, actions, params);
    resolve_collisions_and_energy_void_fields(
        &mut state.units,
        &energy_before_sapping,
        params,
    );
    apply_energy_field(
        &mut state.units,
        &get_energy_field(&state.energy_nodes, params),
        &get_map_mask(&state.nebulae, params.map_size),
        params,
    );
    if state.match_steps % params.spawn_rate == 0 {
        spawn_units(&mut state.units, params)
    }
    let _vision_power_map =
        compute_vision_power_map(&state.units, &state.nebulae, params);
    move_space_objects(
        state,
        &energy_node_deltas
            .unwrap_or_else(|| get_random_energy_node_deltas(rng, params)),
        params,
    );
    update_relic_scores(
        &mut state.team_points,
        &state.units,
        &state.relic_node_points_map,
    );
    // TODO: Move the game_result logic into a function
    let mut game_result = GameResult::empty();
    if state.match_steps >= params.max_steps_in_match {
        let winner = if state.team_points[0] > state.team_points[1] {
            0
        } else if state.team_points[1] > state.team_points[0] {
            1
        } else {
            let (p1_energy, p2_energy) = state
                .units
                .iter()
                .map(|team_units| {
                    team_units.iter().map(|u| u.energy).sum::<i32>()
                })
                .collect_tuple()
                .unwrap();
            if p1_energy > p2_energy {
                0
            } else if p2_energy > p1_energy {
                1
            } else {
                // Congrats, p1 wins "randomly"
                0
            }
        };
        state.team_wins[winner] += 1;
        game_result.match_winner = Some(winner as u8);
        state.match_steps = 0;
    } else {
        state.match_steps += 1;
    }
    state.total_steps += 1;
    if state.total_steps
        >= (params.max_steps_in_match + 1) * params.match_count_per_episode
    {
        let winner = if state.team_wins[0] > state.team_wins[1] {
            0
        } else if state.team_wins[1] > state.team_wins[0] {
            1
        } else {
            panic!(
                "Team wins: {} == {}",
                state.team_wins[0], state.team_wins[1]
            );
        };
        game_result.final_winner = Some(winner as u8);
    };
    unimplemented!("Get per-player observations")
}

fn remove_dead_units(units: &mut [Vec<Unit>; 2]) {
    units[0].retain(|u| u.energy >= 0);
    units[1].retain(|u| u.energy >= 0);
}

fn move_units(
    units: &mut [Vec<Unit>; 2],
    asteroid_mask: &Array2<bool>,
    actions: &[Vec<Action>; 2],
    params: &Params,
) {
    for (unit, action) in units
        .iter_mut()
        .zip_eq(actions.iter())
        .flat_map(|(team_units, team_actions)| {
            team_units.iter_mut().zip_eq(team_actions)
        })
        .filter(|(u, _)| u.energy >= params.unit_move_cost)
    {
        let deltas = match action {
            Action::Up => [0, -1],
            Action::Right => [1, 0],
            Action::Down => [0, 1],
            Action::Left => [-1, 0],
            Action::NoOp | Action::Sap(_) => continue,
        };
        let new_pos = unit.pos.bounded_translate(deltas, params.map_size);
        if asteroid_mask[new_pos.as_index()] {
            continue;
        }
        unit.pos = new_pos;
        unit.energy -= params.unit_move_cost;
    }
}

fn get_map_mask(positions: &[Pos], map_size: [usize; 2]) -> Array2<bool> {
    let mut result = Array2::default(map_size);
    for pos in positions.iter() {
        result[pos.as_index()] = true;
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
    for (team, opp) in [(0, 1), (1, 0)] {
        let mut sap_count = vec![0; units[opp].len()];
        let mut adjacent_sap_count = vec![0; units[opp].len()];
        for (unit_idx, sap_deltas) in unit_energies[team]
            .iter()
            .copied()
            .enumerate()
            .zip_eq(actions[team].iter())
            .filter(|((_, cur_energy), _)| *cur_energy >= params.unit_sap_cost)
            .filter_map(|((unit_idx, _), action)| {
                if let Action::Sap(sap_deltas) = *action {
                    Some((unit_idx, sap_deltas))
                } else {
                    None
                }
            })
            .filter(|(_, [dx, dy])| {
                *dx <= params.unit_sap_range && *dy <= params.unit_sap_range
            })
        {
            let unit = &mut units[team][unit_idx];
            let Some(target_pos) =
                unit.pos.maybe_translate(sap_deltas, params.map_size)
            else {
                continue;
            };
            unit.energy -= params.unit_sap_cost;
            for (i, opp_u) in units[opp].iter().enumerate() {
                match target_pos.subtract(opp_u.pos) {
                    [0, 0] => sap_count[i] += 1,
                    [-1..=1, -1..=1] => adjacent_sap_count[i] += 1,
                    _ => {},
                }
            }
        }

        for ((saps, adj_saps), opp_u) in sap_count
            .into_iter()
            .zip_eq(adjacent_sap_count)
            .zip_eq(units[opp].iter_mut())
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
    for (team, opp) in [(0, 1), (1, 0)] {
        let mut to_remove = Vec::new();
        for (i, unit) in units[team].iter_mut().enumerate() {
            let [x, y] = unit.pos.as_index();
            // Resolve collision
            if unit_counts_map[[opp, x, y]] > 0
                && unit_aggregate_energy_map[[opp, x, y]]
                    >= unit_aggregate_energy_map[[team, x, y]]
            {
                to_remove.push(i);
            }
            // Apply energy void fields
            unit.energy -= (params.unit_energy_void_factor
                * unit_aggregate_energy_void_map[[opp, x, y]]
                / unit_counts_map[[team, x, y]] as f32)
                as i32;
        }
        let mut keep_iter =
            (0..units[team].len()).map(|i| !to_remove.contains(&i));
        units[team].retain(|_| keep_iter.next().unwrap());
    }
}

fn get_unit_aggregate_energy_void_map(
    units: &[Vec<Unit>; 2],
    unit_energies: &[Vec<i32>; 2],
    map_size: [usize; 2],
) -> Array3<f32> {
    let mut result = Array3::zeros((2, map_size[0], map_size[1]));
    for ((team, unit, energy), delta) in units
        .iter()
        .zip_eq(unit_energies)
        .enumerate()
        .flat_map(|(t, (team_units, team_energies))| {
            team_units
                .iter()
                .zip_eq(team_energies.iter().copied())
                .map(move |(u, ue)| (t, u, ue))
        })
        .cartesian_product([[-1, 0], [1, 0], [0, -1], [0, 1]])
    {
        let Some(Pos { x, y }) = unit.pos.maybe_translate(delta, map_size)
        else {
            continue;
        };
        result[[team, x, y]] += energy as f32;
    }
    result
}

fn get_unit_counts_map(
    units: &[Vec<Unit>; 2],
    map_size: [usize; 2],
) -> Array3<u8> {
    let mut result = Array3::zeros((2, map_size[0], map_size[1]));
    for (team, unit) in units
        .iter()
        .enumerate()
        .flat_map(|(t, team_units)| team_units.iter().map(move |u| (t, u)))
    {
        result[[team, unit.pos.x, unit.pos.y]] += 1;
    }
    result
}

fn get_unit_aggregate_energy_map(
    units: &[Vec<Unit>; 2],
    unit_energies: &[Vec<i32>; 2],
    map_size: [usize; 2],
) -> Array3<i32> {
    let mut result = Array3::zeros((2, map_size[0], map_size[1]));
    for (team, unit, energy) in
        units.iter().zip_eq(unit_energies).enumerate().flat_map(
            |(t, (team_units, team_energies))| {
                team_units
                    .iter()
                    .zip_eq(team_energies.iter().copied())
                    .map(move |(u, ue)| (t, u, ue))
            },
        )
    {
        result[[team, unit.pos.x, unit.pos.y]] += energy;
    }
    result
}

fn apply_energy_field(
    units: &mut [Vec<Unit>; 2],
    energy_field: &Array2<i32>,
    nebula_mask: &Array2<bool>,
    params: &Params,
) {
    for (unit, energy_gain) in units
        .iter_mut()
        .flat_map(|team_units| team_units.iter_mut())
        .map(|u| {
            let u_pos_idx = u.pos.as_index();
            let energy_gain = if nebula_mask[u_pos_idx] {
                energy_field[u_pos_idx] - params.nebula_tile_energy_reduction
            } else {
                energy_field[u_pos_idx]
            };
            (u, energy_gain)
        })
        .filter(|(u, energy_gain)| u.energy >= 0 || u.energy + energy_gain >= 0)
    {
        unit.energy = min(
            max(unit.energy + energy_gain, params.min_unit_energy),
            params.max_unit_energy,
        )
    }
}

fn get_energy_field(
    energy_nodes: &[EnergyNode],
    params: &Params,
) -> Array2<i32> {
    let [width, height] = params.map_size;
    let mut energy_field_3d =
        Array3::zeros((energy_nodes.len(), width, height));
    for (((i, node), x), y) in energy_nodes
        .iter()
        .enumerate()
        .cartesian_product(0..width)
        .cartesian_product(0..height)
    {
        let d = get_dist([node.pos.x, node.pos.y], [x, y]);
        energy_field_3d[[i, x, y]] = node.apply_energy_fn(d);
    }
    let mean_val = energy_field_3d.mean().unwrap();
    if mean_val < 0.25 {
        energy_field_3d += 0.25 - mean_val;
    };
    energy_field_3d
        .sum_axis(Axis(0))
        .map(|&v| v.round_ties_even() as i32)
        .map(|&v| {
            min(
                max(v, params.min_energy_per_tile),
                params.max_energy_per_tile,
            )
        })
}

fn get_dist(a: [usize; 2], b: [usize; 2]) -> f32 {
    let [x1, y1] = a;
    let [x2, y2] = b;
    let sum_of_squares =
        (x2 as f32 - x1 as f32).powi(2) + (y2 as f32 - y1 as f32).powi(2);
    sum_of_squares.sqrt()
}

fn spawn_units(units: &mut [Vec<Unit>; 2], params: &Params) {
    for (team, team_units) in units
        .iter_mut()
        .enumerate()
        .filter(|(_, team_units)| team_units.len() < params.max_units)
    {
        let u_id = if team_units.is_empty() || team_units[0].id > 0 {
            0
        } else if team_units[team_units.len() - 1].id == team_units.len() - 1 {
            team_units.len()
        } else {
            team_units
                .iter()
                .tuple_windows()
                .filter_map(|(unit, next_unit)| {
                    if next_unit.id - unit.id > 1 {
                        Some(unit.id + 1)
                    } else {
                        None
                    }
                })
                .next()
                .unwrap()
        };

        let pos = match team {
            0 => Pos::new(0, 0),
            1 => Pos::new(params.map_size[0] - 1, params.map_size[1] - 1),
            n => panic!("this town ain't big enough for the {} of us", n),
        };

        let new_unit = Unit::new_with_id(pos, params.init_unit_energy, u_id);
        team_units.insert(u_id, new_unit);
    }
}

fn compute_vision_power_map(
    units: &[Vec<Unit>; 2],
    nebulae: &[Pos],
    params: &Params,
) -> Array3<i32> {
    let [max_w, max_h] = params.map_size;
    let mut vision_power_map = Array3::zeros((2, max_w, max_h));
    for ((team, x, y), v) in units
        .iter()
        .enumerate()
        .flat_map(|(t, team_units)| {
            team_units.iter().map(move |u| (t, u.pos.x, u.pos.y))
        })
        .cartesian_product(0..=params.unit_sensor_range)
    {
        let range = params.unit_sensor_range - v;
        vision_power_map
            .slice_mut(s![
                team,
                x.saturating_sub(range)..min(x + range + 1, max_w),
                y.saturating_sub(range)..min(y + range + 1, max_h),
            ])
            .iter_mut()
            .for_each(|value| *value += 1);
    }
    for (x, y) in nebulae.iter().map(|n| (n.x, n.y)) {
        vision_power_map
            .slice_mut(s![.., x, y])
            .iter_mut()
            .for_each(|value| *value -= params.nebula_tile_vision_reduction);
    }
    vision_power_map
}

fn move_space_objects(
    state: &mut State,
    energy_node_deltas: &[[isize; 2]],
    params: &Params,
) {
    if state.total_steps as f32 * params.nebula_tile_drift_speed % 1.0 == 0.0
        && params.nebula_tile_drift_speed != 0.0
    {
        let deltas = [
            params.nebula_tile_drift_speed.signum() as isize,
            -params.nebula_tile_drift_speed.signum() as isize,
        ];
        for pos in state.asteroids.iter_mut().chain(state.nebulae.iter_mut()) {
            *pos = pos.wrapped_translate(deltas, params.map_size)
        }
    }

    if state.total_steps as f32 * params.energy_node_drift_speed % 1.0 == 0.0 {
        // Chain with symmetric energy_node_deltas
        for (deltas, node) in energy_node_deltas
            .iter()
            .copied()
            .chain(energy_node_deltas.iter().map(|[dx, dy]| [-dy, -dx]))
            .zip_eq(state.energy_nodes.iter_mut())
        {
            node.pos = node.pos.bounded_translate(deltas, params.map_size)
        }
    }
}

fn get_random_energy_node_deltas(
    rng: &mut ThreadRng,
    params: &Params,
) -> Vec<[isize; 2]> {
    let uniform = Uniform::new(
        -params.energy_node_drift_magnitude,
        params.energy_node_drift_magnitude,
    );
    (0..params.max_energy_nodes / 2)
        .map(|_| [uniform.sample(rng) as isize, uniform.sample(rng) as isize])
        .collect()
}

fn update_relic_scores(
    team_points: &mut [u32; 2],
    units: &[Vec<Unit>; 2],
    relic_node_points_map: &Array2<bool>,
) {
    for team in [0, 1] {
        team_points[team] += units[team]
            .iter()
            .map(|u| relic_node_points_map[u.pos.as_index()] as u32)
            .sum::<u32>();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules_engine::state::{Pos, Unit};
    use numpy::ndarray::{arr2, arr3};

    #[test]
    fn test_move_units() {
        let params = Params::default();
        let mut units = [
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
        let asteroid_mask =
            get_map_mask(&vec![Pos::new(23, 22)], params.map_size);
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
        move_units(&mut units, &asteroid_mask, &actions, &params);
        assert_eq!(units, expected_moved_units);
    }

    #[test]
    fn test_get_map_mask() {
        let asteroids = Vec::new();
        let expected_result = arr2(&[[false; 3]; 3]);
        assert_eq!(get_map_mask(&asteroids, [3, 3]), expected_result);

        let nebulae = vec![Pos::new(0, 0), Pos::new(0, 1), Pos::new(2, 1)];
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
        let mut params = Params::default();
        params.nebula_tile_energy_reduction = 10;
        params.min_unit_energy = 0;
        params.max_unit_energy = 100;
        let energy_field = arr2(&[[-10, 2], [5, 10]]);
        let nebula_mask = arr2(&[[false, false], [true, false]]);
        let mut units = [
            vec![
                // Units with negative energy that can't be saved are unaffected
                Unit::new(Pos::new(0, 1), -3),
                // Units with negative energy that would pass 0 are saved
                Unit::new(Pos::new(0, 1), -2),
                // Power gain is affected by nebulas
                Unit::new(Pos::new(1, 0), 10),
                // Power loss (due to fields or nebulas) cannot go below min_unit_energy
                Unit::new(Pos::new(0, 0), 1),
                Unit::new(Pos::new(1, 0), 1),
            ],
            vec![
                // Units can gain power
                Unit::new(Pos::new(1, 1), 10),
                // Units cannot gain power beyond max_unit_energy
                Unit::new(Pos::new(1, 1), 95),
            ],
        ];
        let expected_result = [
            vec![
                // Units with negative energy that can't be saved are unaffected
                Unit::new(Pos::new(0, 1), -3),
                // Units with negative energy that would pass 0 are saved
                Unit::new(Pos::new(0, 1), 0),
                // Power gain is affected by nebulas
                Unit::new(
                    Pos::new(1, 0),
                    10 + 5 - params.nebula_tile_energy_reduction,
                ),
                // Power loss (due to fields or nebulas) cannot bring a unit below min_unit_energy
                Unit::new(Pos::new(0, 0), params.min_unit_energy),
                Unit::new(Pos::new(1, 0), params.min_unit_energy),
            ],
            vec![
                // Units can gain power
                Unit::new(Pos::new(1, 1), 20),
                // Units cannot gain power beyond max_unit_energy
                Unit::new(Pos::new(1, 1), params.max_unit_energy),
            ],
        ];
        apply_energy_field(&mut units, &energy_field, &nebula_mask, &params);
        assert_eq!(units, expected_result);
    }

    #[test]
    #[ignore]
    fn test_get_energy_field() {
        // TODO
        unimplemented!("Write test cases using JSON replay files")
    }

    #[test]
    fn test_get_dist() {
        // Same points should return 0
        assert_eq!(get_dist([0, 0], [0, 0]), 0.);
        assert_eq!(get_dist([2, 3], [2, 3]), 0.);
        // Order shouldn't matter
        let (a, b) = ([1, 5], [22, 17]);
        assert_eq!(get_dist(a, b), get_dist(b, a));
        // Check a couple cases by hand
        assert_eq!(get_dist([1, 3], [5, 6]), 5.);
        assert_eq!(get_dist([2, 1], [14, 6]), 13.);
        assert_eq!(get_dist([3, 1], [6, 4]), 18_f32.sqrt());
    }

    #[test]
    fn test_spawn_units_id_0() {
        let params = Params::default();
        let mut units = [
            // Empty vector; should spawn unit with id 0
            vec![],
            // Vector missing id 0; add it
            vec![Unit::new_with_id(Pos::new(1, 1), 42, 1)],
        ];
        let expected_result = [
            // Empty vector; should spawn unit with id 0
            vec![Unit::new_with_id(
                Pos::new(0, 0),
                params.init_unit_energy,
                0,
            )],
            // Vector missing id 0; add it
            vec![
                Unit::new_with_id(Pos::new(23, 23), params.init_unit_energy, 0),
                Unit::new_with_id(Pos::new(1, 1), 42, 1),
            ],
        ];
        spawn_units(&mut units, &params);
        assert_eq!(units, expected_result);
    }

    #[test]
    fn test_spawn_units_id_last_and_max_units() {
        let params = Params::default();
        let mut units = [
            // Contains all IDs; should add next one
            vec![
                Unit::new_with_id(Pos::new(1, 1), -5, 0),
                Unit::new_with_id(Pos::new(1, 1), -5, 1),
                Unit::new_with_id(Pos::new(1, 1), -5, 2),
            ],
            // Contains max units; don't add any
            (0..params.max_units)
                .map(|id| Unit::new_with_id(Pos::new(9, 9), 42, id))
                .collect(),
        ];
        let expected_result = [
            // Contains all IDs; should add next one
            vec![
                Unit::new_with_id(Pos::new(1, 1), -5, 0),
                Unit::new_with_id(Pos::new(1, 1), -5, 1),
                Unit::new_with_id(Pos::new(1, 1), -5, 2),
                Unit::new_with_id(Pos::new(0, 0), params.init_unit_energy, 3),
            ],
            // Contains max units; don't add any
            units[1].clone(),
        ];
        spawn_units(&mut units, &params);
        assert_eq!(units, expected_result);
    }

    #[test]
    fn test_spawn_units_inserts_unit_id() {
        let params = Params::default();
        let mut units = [
            // Insert unit at correct location
            vec![
                Unit::new_with_id(Pos::new(1, 1), 42, 0),
                Unit::new_with_id(Pos::new(1, 1), 42, 2),
                Unit::new_with_id(Pos::new(1, 1), 42, 4),
            ],
            // Insert unit at correct location
            vec![
                Unit::new_with_id(Pos::new(9, 9), 42, 0),
                Unit::new_with_id(Pos::new(9, 9), 42, 1),
                Unit::new_with_id(Pos::new(9, 9), 42, 4),
            ],
        ];
        let expected_result = [
            // Insert unit at correct location
            vec![
                Unit::new_with_id(Pos::new(1, 1), 42, 0),
                Unit::new_with_id(Pos::new(0, 0), params.init_unit_energy, 1),
                Unit::new_with_id(Pos::new(1, 1), 42, 2),
                Unit::new_with_id(Pos::new(1, 1), 42, 4),
            ],
            // Insert unit at correct location
            vec![
                Unit::new_with_id(Pos::new(9, 9), 42, 0),
                Unit::new_with_id(Pos::new(9, 9), 42, 1),
                Unit::new_with_id(Pos::new(23, 23), params.init_unit_energy, 2),
                Unit::new_with_id(Pos::new(9, 9), 42, 4),
            ],
        ];
        spawn_units(&mut units, &params);
        assert_eq!(units, expected_result);
    }

    #[test]
    fn test_compute_vision_power_map_one_unit() {
        let mut params = Params::default();
        params.map_size = [5, 5];
        params.unit_sensor_range = 1;
        let units = [
            vec![Unit::new_at(Pos::new(2, 2))],
            vec![Unit::new_at(Pos::new(1, 1))],
        ];
        let expected_result = arr3(&[
            [
                [0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 2, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [1, 1, 1, 0, 0],
                [1, 2, 1, 0, 0],
                [1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
        ]);
        assert_eq!(
            compute_vision_power_map(&units, &Vec::new(), &params),
            expected_result
        );
    }

    #[test]
    fn test_compute_vision_power_map_handles_edge_of_map() {
        let mut params = Params::default();
        params.map_size = [3, 3];
        params.unit_sensor_range = 1;
        let units = [
            vec![Unit::new_at(Pos::new(0, 0))],
            vec![Unit::new_at(Pos::new(2, 2))],
        ];
        let expected_result = arr3(&[
            [[2, 1, 0], [1, 1, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 1, 1], [0, 1, 2]],
        ]);
        assert_eq!(
            compute_vision_power_map(&units, &Vec::new(), &params),
            expected_result
        );
    }

    #[test]
    fn test_compute_vision_power_map_is_additive_with_nebulae() {
        let mut params = Params::default();
        params.map_size = [5, 5];
        params.unit_sensor_range = 1;
        params.nebula_tile_vision_reduction = 5;
        let units = [
            vec![Unit::new_at(Pos::new(1, 1)), Unit::new_at(Pos::new(2, 2))],
            vec![Unit::new_at(Pos::new(2, 2)), Unit::new_at(Pos::new(2, 2))],
        ];
        let nebulae = vec![Pos::new(1, 1), Pos::new(2, 3)];
        let expected_result = arr3(&[
            [
                [1, 1, 1, 0, 0],
                [1, 3 - 5, 2, 1, 0],
                [1, 2, 3, 1 - 5, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0],
                [0, 2 - 5, 2, 2, 0],
                [0, 2, 4, 2 - 5, 0],
                [0, 2, 2, 2, 0],
                [0, 0, 0, 0, 0],
            ],
        ]);
        assert_eq!(
            compute_vision_power_map(&units, &nebulae, &params),
            expected_result
        );
    }

    #[test]
    fn test_move_space_objects() {
        let mut params = Params::default();
        params.nebula_tile_drift_speed = -0.05;
        params.energy_node_drift_speed = 0.02;
        params.energy_node_drift_magnitude = 5.0;
        let mut state = State::empty(params.map_size);
        state.asteroids = vec![
            // Moves normally
            Pos::new(10, 10),
            // Wraps as expected
            Pos::new(0, 0),
            Pos::new(23, 23),
        ];
        state.nebulae = vec![
            // Moves normally
            Pos::new(11, 11),
            // Wraps as expected
            Pos::new(0, 0),
            Pos::new(23, 23),
        ];
        state.energy_nodes = vec![
            // Moves normally
            EnergyNode::new_at(Pos::new(12, 12)),
            // Stops at edge of board
            EnergyNode::new_at(Pos::new(21, 22)),
            // Moves normally
            EnergyNode::new_at(Pos::new(14, 14)),
            // Stops at edge of board
            EnergyNode::new_at(Pos::new(1, 2)),
        ];
        let energy_node_deltas = vec![[-3, 4], [3, 3]];

        let expected_asteroids =
            vec![Pos::new(9, 11), Pos::new(23, 1), Pos::new(22, 0)];
        let expected_nebulae =
            vec![Pos::new(10, 12), Pos::new(23, 1), Pos::new(22, 0)];
        let expected_energy_nodes = vec![
            EnergyNode::new_at(Pos::new(9, 16)),
            EnergyNode::new_at(Pos::new(23, 23)),
            EnergyNode::new_at(Pos::new(10, 17)),
            EnergyNode::new_at(Pos::new(0, 0)),
        ];
        move_space_objects(&mut state, &energy_node_deltas, &params);
        assert_eq!(state.asteroids, expected_asteroids);
        assert_eq!(state.nebulae, expected_nebulae);
        assert_eq!(state.energy_nodes, expected_energy_nodes);
    }

    #[test]
    fn test_move_space_objects_no_op() {
        let mut params = Params::default();
        params.nebula_tile_drift_speed = -0.05;
        params.energy_node_drift_speed = 0.02;
        params.energy_node_drift_magnitude = 5.0;
        let mut state = State::empty(params.map_size);
        state.asteroids = vec![Pos::new(1, 1)];
        state.nebulae = vec![Pos::new(2, 2)];
        state.energy_nodes = vec![
            EnergyNode::new_at(Pos::new(3, 3)),
            EnergyNode::new_at(Pos::new(4, 4)),
        ];
        state.total_steps = 7;
        let orig_state = state.clone();
        move_space_objects(&mut state, &[[-1, 2]], &params);
        assert_eq!(state, orig_state);
    }
}
