use super::action::Action;
use super::params::Params;
use super::state::{EnergyNode, GameResult, Observation, Pos, State, Unit};
use itertools::Itertools;
use numpy::ndarray::{s, Array2, Array3, Axis};
use rand::distributions::{Distribution, Uniform};
use rand::prelude::ThreadRng;
use std::cmp::{max, min, Ordering};

pub fn step(
    state: &mut State,
    rng: &mut ThreadRng,
    actions: &[Vec<Action>; 2],
    params: &Params,
    energy_node_deltas: Option<Vec<[isize; 2]>>,
) -> ([Observation; 2], GameResult) {
    if state.done {
        panic!("Game over, need to reset State")
    }

    if state.match_steps == 0 {
        state.units = [Vec::new(), Vec::new()];
    }
    remove_dead_units(&mut state.units);
    let actions = get_relevant_actions(actions, &state.units);
    move_units(
        &mut state.units,
        &get_map_mask(&state.asteroids, params.get_map_size()),
        &actions,
        params,
    );
    let energy_before_sapping = get_unit_energies(&state.units);
    sap_units(&mut state.units, &energy_before_sapping, &actions, params);
    resolve_collisions_and_energy_void_fields(
        &mut state.units,
        &energy_before_sapping,
        params,
    );
    apply_energy_field(
        &mut state.units,
        &get_energy_field(&state.energy_nodes, params),
        &get_map_mask(&state.nebulae, params.get_map_size()),
        params,
    );
    if state.match_steps % params.spawn_rate == 0 {
        spawn_units(&mut state.units, params)
    }
    let vision_power_map = compute_vision_power_map_from_params(
        &state.units,
        &state.nebulae,
        params,
    );
    move_space_objects(
        state,
        &energy_node_deltas.unwrap_or_else(|| {
            get_random_energy_node_deltas(rng, state.energy_nodes.len(), params)
        }),
        params,
    );
    update_relic_scores(
        &mut state.team_points,
        &state.units,
        &state.relic_node_points_map,
    );
    let match_winner = get_match_result(state, params);
    step_match(state, match_winner);
    let game_winner = step_game(
        &mut state.total_steps,
        &mut state.done,
        &state.team_wins,
        params,
    );
    (
        get_observation(state, &vision_power_map),
        GameResult::new(match_winner, game_winner),
    )
}

fn remove_dead_units(units: &mut [Vec<Unit>; 2]) {
    units[0].retain(|u| u.energy >= 0);
    units[1].retain(|u| u.energy >= 0);
}

fn get_relevant_actions(
    actions: &[Vec<Action>; 2],
    units: &[Vec<Unit>; 2],
) -> [Vec<Action>; 2] {
    let mut result = [Vec::new(), Vec::new()];
    for (team, (team_actions, team_units)) in
        actions.iter().zip_eq(units.iter()).enumerate()
    {
        result[team] = team_units
            .iter()
            .map(|unit| team_actions[unit.id])
            .collect();
    }
    result
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
        // This behavior is almost certainly a bug in the main simulator
        if deltas[0] < 0 || deltas[1] < 0 {
            let wrapped_pos =
                unit.pos.wrapped_translate(deltas, params.get_map_size());
            if asteroid_mask[wrapped_pos.as_index()] {
                continue;
            }
        };
        let new_pos = unit.pos.bounded_translate(deltas, params.get_map_size());
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
                dx.abs() <= params.unit_sap_range
                    && dy.abs() <= params.unit_sap_range
            })
        {
            let unit = &mut units[team][unit_idx];
            let Some(target_pos) =
                unit.pos.maybe_translate(sap_deltas, params.get_map_size())
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
        params.get_map_size(),
    );
    // TODO: unit_counts_map and aggregate_energy_map are sparse - maybe could use a better
    //  datastructure? BTreeMap? Vec?
    let unit_counts_map = get_unit_counts_map(units, params.get_map_size());
    let unit_aggregate_energy_map = get_unit_aggregate_energy_map(
        units,
        unit_energies,
        params.get_map_size(),
    );
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
    let [width, height] = params.get_map_size();
    let mut energy_field_3d =
        Array3::zeros((params.max_energy_nodes, width, height));
    for (((i, node), x), y) in energy_nodes
        .iter()
        .enumerate()
        .cartesian_product(0..width)
        .cartesian_product(0..height)
    {
        let d = get_dist(node.pos.into(), [x, y]);
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
                .windows(2)
                .filter_map(|window| match window {
                    [unit, next_unit] => {
                        if next_unit.id - unit.id > 1 {
                            Some(unit.id + 1)
                        } else {
                            None
                        }
                    },
                    us => {
                        panic!("Got unexpected number of units: {}", us.len())
                    },
                })
                .next()
                .unwrap()
        };

        let pos = match team {
            0 => Pos::new(0, 0),
            1 => Pos::new(params.map_width - 1, params.map_height - 1),
            n => panic!("this town ain't big enough for the {} of us", n),
        };

        let new_unit = Unit::new(pos, params.init_unit_energy, u_id);
        team_units.insert(u_id, new_unit);
    }
}

pub fn estimate_vision_power_map(
    units: &[Unit],
    map_size: [usize; 2],
    unit_sensor_range: usize,
) -> Array2<i32> {
    let units = units.iter().cloned().collect_vec();
    let vision_estimate = compute_vision_power_map(
        &[units],
        &Vec::new(),
        map_size,
        unit_sensor_range,
        0,
    );
    vision_estimate.index_axis_move(Axis(0), 0)
}

fn compute_vision_power_map_from_params(
    units: &[Vec<Unit>],
    nebulae: &[Pos],
    params: &Params,
) -> Array3<i32> {
    compute_vision_power_map(
        units,
        nebulae,
        params.get_map_size(),
        params.unit_sensor_range,
        params.nebula_tile_vision_reduction,
    )
}

fn compute_vision_power_map(
    units: &[Vec<Unit>],
    nebulae: &[Pos],
    map_size: [usize; 2],
    unit_sensor_range: usize,
    nebula_tile_vision_reduction: i32,
) -> Array3<i32> {
    let [width, height] = map_size;
    let mut vision_power_map = Array3::zeros((units.len(), width, height));
    for ((team, x, y), v) in units
        .iter()
        .enumerate()
        .flat_map(|(t, team_units)| {
            team_units.iter().map(move |u| (t, u.pos.x, u.pos.y))
        })
        .cartesian_product(0..=unit_sensor_range)
    {
        let range = unit_sensor_range - v;
        vision_power_map
            .slice_mut(s![
                team,
                x.saturating_sub(range)..min(x + range + 1, width),
                y.saturating_sub(range)..min(y + range + 1, height),
            ])
            .iter_mut()
            .for_each(|value| *value += 1);
    }
    for (x, y) in nebulae.iter().map(|n| (n.x, n.y)) {
        vision_power_map
            .slice_mut(s![.., x, y])
            .iter_mut()
            .for_each(|value| *value -= nebula_tile_vision_reduction);
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
            *pos = pos.wrapped_translate(deltas, params.get_map_size())
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
            node.pos = node.pos.bounded_translate(deltas, params.get_map_size())
        }
    }
}

fn get_random_energy_node_deltas(
    rng: &mut ThreadRng,
    energy_node_count: usize,
    params: &Params,
) -> Vec<[isize; 2]> {
    let uniform = Uniform::new(
        -params.energy_node_drift_magnitude,
        params.energy_node_drift_magnitude,
    );
    (0..energy_node_count / 2)
        .map(|_| [uniform.sample(rng) as isize, uniform.sample(rng) as isize])
        .collect()
}

fn update_relic_scores(
    team_points: &mut [u32; 2],
    units: &[Vec<Unit>; 2],
    relic_node_points_map: &Array2<bool>,
) {
    for team in [0, 1] {
        let mut scored_positions = Vec::with_capacity(units[team].len());
        team_points[team] += units[team]
            .iter()
            .map(|u| {
                if relic_node_points_map[u.pos.as_index()]
                    && !scored_positions.contains(&u.pos)
                {
                    scored_positions.push(u.pos);
                    1
                } else {
                    0
                }
            })
            .sum::<u32>();
    }
}

fn get_match_result(state: &State, params: &Params) -> Option<u8> {
    if state.match_steps < params.max_steps_in_match {
        return None;
    }
    match state.team_points[0].cmp(&state.team_points[1]) {
        Ordering::Greater => Some(0),
        Ordering::Less => Some(1),
        Ordering::Equal => {
            let (p1_energy, p2_energy) = state
                .units
                .iter()
                .map(|team_units| {
                    team_units.iter().map(|u| u.energy).sum::<i32>()
                })
                .collect_tuple()
                .unwrap();
            match p1_energy.cmp(&p2_energy) {
                Ordering::Greater => Some(0),
                Ordering::Less => Some(1),
                // Congrats, p1 wins "randomly"
                Ordering::Equal => Some(0),
            }
        },
    }
}

fn step_match(state: &mut State, match_winner: Option<u8>) {
    if let Some(winner) = match_winner {
        state.team_points = [0, 0];
        state.team_wins[usize::from(winner)] += 1;
        state.match_steps = 0;
    } else {
        state.match_steps += 1;
    };
}

fn step_game(
    total_steps: &mut u32,
    game_over: &mut bool,
    team_wins: &[u32; 2],
    params: &Params,
) -> Option<u8> {
    // NB: Early termination is not used in the original implementation
    *total_steps += 1;
    let early_result = match team_wins {
        [3, _] => Some(0),
        [_, 3] => Some(1),
        _ => None,
    };
    if early_result.is_some() {
        *game_over = true;
        return early_result;
    }

    if *total_steps
        < (params.max_steps_in_match + 1) * params.match_count_per_episode
    {
        return None;
    }

    *game_over = true;
    match team_wins[0].cmp(&team_wins[1]) {
        Ordering::Greater => Some(0),
        Ordering::Less => Some(1),
        Ordering::Equal => {
            panic!("Team wins tied: {} == {}", team_wins[0], team_wins[1]);
        },
    }
}

fn get_observation(
    state: &State,
    vision_power_map: &Array3<i32>,
) -> [Observation; 2] {
    let [p1_mask, p2_mask] = get_sensor_masks(vision_power_map);
    let mut observations = [
        Observation::new(
            0,
            p1_mask,
            state.team_points,
            state.team_wins,
            state.total_steps,
            state.match_steps,
        ),
        Observation::new(
            1,
            p2_mask,
            state.team_points,
            state.team_wins,
            state.total_steps,
            state.match_steps,
        ),
    ];
    for ((team, opp), obs) in
        [(0, 1), (1, 0)].into_iter().zip_eq(observations.iter_mut())
    {
        obs.units[team] = state.units[team].clone();
        obs.units[opp] = state.units[opp]
            .iter()
            .filter(|u| obs.sensor_mask[u.pos.as_index()])
            .cloned()
            .collect();
        obs.asteroids = state
            .asteroids
            .iter()
            .copied()
            .filter(|a| obs.sensor_mask[a.as_index()])
            .collect();
        obs.nebulae = state
            .nebulae
            .iter()
            .copied()
            .filter(|a| obs.sensor_mask[a.as_index()])
            .collect();
        obs.relic_node_locations = state
            .relic_node_locations
            .iter()
            .copied()
            .filter(|a| obs.sensor_mask[a.as_index()])
            .collect();
    }
    observations
}

fn get_sensor_masks(vision_power_map: &Array3<i32>) -> [Array2<bool>; 2] {
    [
        vision_power_map.slice(s![0, .., ..]).map(|&v| v > 0),
        vision_power_map.slice(s![1, .., ..]).map(|&v| v > 0),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules_engine::params::{MAP_SIZE, MAX_RELIC_NODES};
    use crate::rules_engine::replay::FullReplay;
    use crate::rules_engine::state::{Pos, Unit};
    use numpy::ndarray::{arr2, arr3, stack};
    use pretty_assertions::assert_eq;
    use rstest::rstest;
    use serde::Deserialize;
    use std::fs;
    use std::path::Path;

    #[test]
    #[should_panic(expected = "Game over")]
    fn test_step_panics_without_reset() {
        let params = Params::default();
        let mut state = State::default();
        state.done = true;
        step(
            &mut state,
            &mut rand::thread_rng(),
            &[Vec::new(), Vec::new()],
            &params,
            None,
        );
    }

    #[test]
    fn test_move_units() {
        let params = Params::default();
        let mut units = [
            vec![
                // Unit can't move without energy, costs no energy
                Unit::with_pos_and_energy(
                    Pos::new(0, 0),
                    params.unit_move_cost - 1,
                ),
                // Unit can't move off the bottom of the map, but still costs energy
                Unit::with_pos_and_energy(Pos::new(0, 0), 100),
                // Unit moves normally
                Unit::with_pos_and_energy(Pos::new(0, 0), 100),
            ],
            vec![
                // Unit can't move off the top of the map, but still costs energy
                Unit::with_pos_and_energy(
                    Pos::new(23, 23),
                    params.unit_move_cost,
                ),
                // Unit can't move into an asteroid, costs no energy
                Unit::with_pos_and_energy(Pos::new(23, 23), 100),
            ],
        ];
        let asteroid_mask =
            get_map_mask(&vec![Pos::new(23, 22)], params.get_map_size());
        let actions = [
            vec![Action::Left, Action::Left, Action::Right],
            vec![Action::Down, Action::Up],
        ];
        let expected_moved_units = [
            vec![
                Unit::with_pos_and_energy(
                    Pos::new(0, 0),
                    params.unit_move_cost - 1,
                ),
                Unit::with_pos_and_energy(
                    Pos::new(0, 0),
                    100 - params.unit_move_cost,
                ),
                Unit::with_pos_and_energy(
                    Pos::new(1, 0),
                    100 - params.unit_move_cost,
                ),
            ],
            vec![
                Unit::with_pos_and_energy(Pos::new(23, 23), 0),
                Unit::with_pos_and_energy(Pos::new(23, 23), 100),
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
                Unit::with_pos_and_energy(Pos::new(0, 0), 100),
                // Can't sap out of range, costs no energy
                Unit::with_pos_and_energy(Pos::new(23, 23), 100),
                // Can't sap without enough energy
                Unit::with_pos_and_energy(Pos::new(2, 2), sap_cost - 1),
                // Sap should work normally, hit all adjacent units, and not hit allied units
                Unit::with_pos_and_energy(Pos::new(1, 2), sap_cost),
                // Sap should work normally at max range
                Unit::with_pos_and_energy(
                    Pos::new(
                        1 + params.unit_sap_range as usize,
                        1 + params.unit_sap_range as usize,
                    ),
                    100,
                ),
            ],
            vec![
                Unit::with_pos_and_energy(Pos::new(0, 0), 100),
                Unit::with_pos_and_energy(Pos::new(1, 1), 100),
                Unit::with_pos_and_energy(Pos::new(2, 2), 100),
            ],
        ];
        let actions = [
            vec![
                Action::Sap([-1, -1]),
                Action::Sap([-params.unit_sap_range - 1, 0]),
                Action::Sap([0, 0]),
                Action::Sap([0, 0]),
                Action::Sap([-params.unit_sap_range, -params.unit_sap_range]),
            ],
            vec![Action::NoOp, Action::NoOp, Action::NoOp],
        ];

        let expected_sapped_units = [
            vec![
                Unit::with_pos_and_energy(Pos::new(0, 0), 100),
                Unit::with_pos_and_energy(Pos::new(23, 23), 100),
                Unit::with_pos_and_energy(Pos::new(2, 2), sap_cost - 1),
                Unit::with_pos_and_energy(Pos::new(1, 2), 0),
                Unit::with_pos_and_energy(
                    Pos::new(
                        1 + params.unit_sap_range as usize,
                        1 + params.unit_sap_range as usize,
                    ),
                    100 - sap_cost,
                ),
            ],
            vec![
                Unit::with_pos_and_energy(Pos::new(0, 0), 100 - (sap_cost / 2)),
                Unit::with_pos_and_energy(
                    Pos::new(1, 1),
                    100 - sap_cost - sap_cost / 2,
                ),
                Unit::with_pos_and_energy(
                    Pos::new(2, 2),
                    100 - (2 * sap_cost / 2),
                ),
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
                Unit::with_pos_and_energy(Pos::new(0, 0), 1),
                Unit::with_pos_and_energy(Pos::new(0, 0), 1),
                Unit::with_pos_and_energy(Pos::new(0, 0), 1),
                // Everyone dies in a tie
                Unit::with_pos_and_energy(Pos::new(1, 1), 1),
                Unit::with_pos_and_energy(Pos::new(1, 1), 1),
                // Energy voids are combined/shared
                Unit::with_pos_and_energy(Pos::new(2, 2), 100),
            ],
            vec![
                // Don't collide with self, more energy wins
                Unit::with_pos_and_energy(Pos::new(0, 0), 2),
                Unit::with_pos_and_energy(Pos::new(0, 0), 2),
                // Everyone dies in a tie
                Unit::with_pos_and_energy(Pos::new(1, 1), 2),
                // Energy voids are combined/shared
                Unit::with_pos_and_energy(Pos::new(2, 3), 100),
                Unit::with_pos_and_energy(Pos::new(2, 3), 100),
            ],
        ];
        let expected_result = [
            vec![Unit::with_pos_and_energy(Pos::new(2, 2), 100 - 25 - 25)],
            vec![
                Unit::with_pos_and_energy(Pos::new(0, 0), 2),
                Unit::with_pos_and_energy(Pos::new(0, 0), 2),
                Unit::with_pos_and_energy(Pos::new(2, 3), 100 - 12),
                Unit::with_pos_and_energy(Pos::new(2, 3), 100 - 12),
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
                Unit::with_pos_and_energy(Pos::new(0, 0), 1),
                Unit::with_pos_and_energy(Pos::new(1, 1), 2),
            ],
            vec![
                // Should handle different energy amounts and stacked units
                Unit::with_pos_and_energy(Pos::new(0, 0), 2),
                Unit::with_pos_and_energy(Pos::new(0, 0), 2),
                Unit::with_pos_and_energy(Pos::new(0, 1), 30),
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
                Unit::with_pos(Pos::new(0, 0)),
                Unit::with_pos(Pos::new(0, 0)),
                Unit::with_pos(Pos::new(0, 1)),
                Unit::with_pos(Pos::new(0, 1)),
                Unit::with_pos(Pos::new(0, 1)),
            ],
            vec![
                // Different teams have different stacks
                Unit::with_pos(Pos::new(1, 0)),
                Unit::with_pos(Pos::new(1, 0)),
                Unit::with_pos(Pos::new(1, 1)),
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
                Unit::with_pos_and_energy(Pos::new(0, 0), 3),
                Unit::with_pos_and_energy(Pos::new(0, 0), 20),
                Unit::with_pos_and_energy(Pos::new(0, 0), 100),
            ],
            vec![
                // Different teams have different stacks
                Unit::with_pos_and_energy(Pos::new(0, 1), 40),
                Unit::with_pos_and_energy(Pos::new(0, 1), 5),
                Unit::with_pos_and_energy(Pos::new(1, 0), 67),
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
                Unit::with_pos_and_energy(Pos::new(0, 1), -3),
                // Units with negative energy that would pass 0 are saved
                Unit::with_pos_and_energy(Pos::new(0, 1), -2),
                // Power gain is affected by nebulas
                Unit::with_pos_and_energy(Pos::new(1, 0), 10),
                // Power loss (due to fields or nebulas) cannot go below min_unit_energy
                Unit::with_pos_and_energy(Pos::new(0, 0), 1),
                Unit::with_pos_and_energy(Pos::new(1, 0), 1),
            ],
            vec![
                // Units can gain power
                Unit::with_pos_and_energy(Pos::new(1, 1), 10),
                // Units cannot gain power beyond max_unit_energy
                Unit::with_pos_and_energy(Pos::new(1, 1), 95),
            ],
        ];
        let expected_result = [
            vec![
                // Units with negative energy that can't be saved are unaffected
                Unit::with_pos_and_energy(Pos::new(0, 1), -3),
                // Units with negative energy that would pass 0 are saved
                Unit::with_pos_and_energy(Pos::new(0, 1), 0),
                // Power gain is affected by nebulas
                Unit::with_pos_and_energy(
                    Pos::new(1, 0),
                    10 + 5 - params.nebula_tile_energy_reduction,
                ),
                // Power loss (due to fields or nebulas) cannot bring a unit below min_unit_energy
                Unit::with_pos_and_energy(
                    Pos::new(0, 0),
                    params.min_unit_energy,
                ),
                Unit::with_pos_and_energy(
                    Pos::new(1, 0),
                    params.min_unit_energy,
                ),
            ],
            vec![
                // Units can gain power
                Unit::with_pos_and_energy(Pos::new(1, 1), 20),
                // Units cannot gain power beyond max_unit_energy
                Unit::with_pos_and_energy(
                    Pos::new(1, 1),
                    params.max_unit_energy,
                ),
            ],
        ];
        apply_energy_field(&mut units, &energy_field, &nebula_mask, &params);
        assert_eq!(units, expected_result);
    }

    #[derive(Deserialize)]
    struct EnergyFieldTestCase {
        seed: u32,
        energy_nodes: Vec<[usize; 2]>,
        energy_node_fns: Vec<[f32; 4]>,
        energy_field: Vec<Vec<i32>>,
    }

    #[rstest]
    #[case("get_energy_field_407811525.json")]
    #[case("get_energy_field_425608142.json")]
    #[case("get_energy_field_1815350780.json")]
    fn test_get_energy_field(#[case] file_name: &str) {
        let path = Path::new(file!())
            .parent()
            .unwrap()
            .join("test_data")
            .join(file_name);
        let json_data = fs::read_to_string(path).unwrap();
        let test_case: EnergyFieldTestCase =
            serde_json::from_str(&json_data).unwrap();
        let params = Params::default();
        let energy_nodes: Vec<EnergyNode> = test_case
            .energy_nodes
            .iter()
            .copied()
            .zip_eq(test_case.energy_node_fns.iter().copied())
            .map(|(pos, [f_id, xyz @ ..])| {
                EnergyNode::new(pos.into(), f_id as u8, xyz)
            })
            .collect();
        let expected_result = Array2::from_shape_vec(
            params.get_map_size(),
            test_case.energy_field.iter().flatten().copied().collect(),
        )
        .unwrap();
        let result = get_energy_field(&energy_nodes, &params);
        println!("{}", result.clone() - expected_result.clone());
        assert_eq!(result, expected_result);
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
            vec![Unit::new(Pos::new(1, 1), 42, 1)],
        ];
        let expected_result = [
            // Empty vector; should spawn unit with id 0
            vec![Unit::new(Pos::new(0, 0), params.init_unit_energy, 0)],
            // Vector missing id 0; add it
            vec![
                Unit::new(Pos::new(23, 23), params.init_unit_energy, 0),
                Unit::new(Pos::new(1, 1), 42, 1),
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
                Unit::new(Pos::new(1, 1), -5, 0),
                Unit::new(Pos::new(1, 1), -5, 1),
                Unit::new(Pos::new(1, 1), -5, 2),
            ],
            // Contains max units; don't add any
            (0..params.max_units)
                .map(|id| Unit::new(Pos::new(9, 9), 42, id))
                .collect(),
        ];
        let expected_result = [
            // Contains all IDs; should add next one
            vec![
                Unit::new(Pos::new(1, 1), -5, 0),
                Unit::new(Pos::new(1, 1), -5, 1),
                Unit::new(Pos::new(1, 1), -5, 2),
                Unit::new(Pos::new(0, 0), params.init_unit_energy, 3),
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
                Unit::new(Pos::new(1, 1), 42, 0),
                Unit::new(Pos::new(1, 1), 42, 2),
                Unit::new(Pos::new(1, 1), 42, 4),
            ],
            // Insert unit at correct location
            vec![
                Unit::new(Pos::new(9, 9), 42, 0),
                Unit::new(Pos::new(9, 9), 42, 1),
                Unit::new(Pos::new(9, 9), 42, 4),
            ],
        ];
        let expected_result = [
            // Insert unit at correct location
            vec![
                Unit::new(Pos::new(1, 1), 42, 0),
                Unit::new(Pos::new(0, 0), params.init_unit_energy, 1),
                Unit::new(Pos::new(1, 1), 42, 2),
                Unit::new(Pos::new(1, 1), 42, 4),
            ],
            // Insert unit at correct location
            vec![
                Unit::new(Pos::new(9, 9), 42, 0),
                Unit::new(Pos::new(9, 9), 42, 1),
                Unit::new(Pos::new(23, 23), params.init_unit_energy, 2),
                Unit::new(Pos::new(9, 9), 42, 4),
            ],
        ];
        spawn_units(&mut units, &params);
        assert_eq!(units, expected_result);
    }

    #[test]
    fn test_compute_vision_power_map_one_unit() {
        let mut params = Params::default();
        params.map_width = 5;
        params.map_height = 5;
        params.unit_sensor_range = 1;
        let units = [
            vec![Unit::with_pos(Pos::new(2, 2))],
            vec![Unit::with_pos(Pos::new(1, 1))],
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
            compute_vision_power_map_from_params(&units, &Vec::new(), &params),
            expected_result
        );
    }

    #[test]
    fn test_compute_vision_power_map_handles_edge_of_map() {
        let mut params = Params::default();
        params.map_width = 3;
        params.map_height = 3;
        params.unit_sensor_range = 1;
        let units = [
            vec![Unit::with_pos(Pos::new(0, 0))],
            vec![Unit::with_pos(Pos::new(2, 2))],
        ];
        let expected_result = arr3(&[
            [[2, 1, 0], [1, 1, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 1, 1], [0, 1, 2]],
        ]);
        assert_eq!(
            compute_vision_power_map_from_params(&units, &Vec::new(), &params),
            expected_result
        );
    }

    #[test]
    fn test_compute_vision_power_map_is_additive_with_nebulae() {
        let mut params = Params::default();
        params.map_width = 5;
        params.map_height = 5;
        params.unit_sensor_range = 1;
        params.nebula_tile_vision_reduction = 5;
        let units = [
            vec![
                Unit::with_pos(Pos::new(1, 1)),
                Unit::with_pos(Pos::new(2, 2)),
            ],
            vec![
                Unit::with_pos(Pos::new(2, 2)),
                Unit::with_pos(Pos::new(2, 2)),
            ],
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
            compute_vision_power_map_from_params(&units, &nebulae, &params),
            expected_result
        );
    }

    #[test]
    fn test_move_space_objects() {
        let mut params = Params::default();
        params.nebula_tile_drift_speed = -0.05;
        params.energy_node_drift_speed = 0.02;
        params.energy_node_drift_magnitude = 5.0;
        let mut state = State::default();
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
        let mut state = State::default();
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

    #[test]
    fn test_update_relic_scores() {
        let units = [
            vec![
                // Earns points
                Unit::with_pos(Pos::new(0, 0)),
                // Does not earn points
                Unit::with_pos(Pos::new(0, 1)),
                // Duplicates only score once
                Unit::with_pos(Pos::new(1, 1)),
                Unit::with_pos(Pos::new(1, 1)),
            ],
            vec![
                // Does not earn points
                Unit::with_pos(Pos::new(0, 1)),
                // Earns points
                Unit::with_pos(Pos::new(1, 0)),
            ],
        ];
        let mut team_points = [2, 2];
        let points_map = arr2(&[[true, false], [true, true]]);
        update_relic_scores(&mut team_points, &units, &points_map);
        assert_eq!(team_points, [2 + 2, 2 + 1]);
    }

    #[test]
    fn test_get_match_result() {
        let params = Params::default();
        let mut state = State::default();
        state.team_points = [25, 24];
        state.match_steps = params.max_steps_in_match - 1;
        let result = get_match_result(&state, &params);
        assert!(result.is_none());

        state.match_steps += 1;
        let result = get_match_result(&state, &params);
        assert_eq!(result, Some(0));

        state.team_points = [25, 26];
        let result = get_match_result(&state, &params);
        assert_eq!(result, Some(1));
    }

    #[test]
    fn test_get_match_result_tiebreaks_points() {
        let params = Params::default();
        let mut state = State::default();
        state.team_points = [10, 10];
        state.match_steps = params.max_steps_in_match;
        state.units = [
            vec![Unit::with_energy(20), Unit::with_energy(30)],
            vec![Unit::with_energy(49)],
        ];
        let result = get_match_result(&state, &params);
        assert_eq!(result, Some(0));

        state.units = [
            vec![Unit::with_energy(20), Unit::with_energy(30)],
            vec![Unit::with_energy(51)],
        ];
        let result = get_match_result(&state, &params);
        assert_eq!(result, Some(1));

        // Ties favor player 1
        state.units = [vec![], vec![]];
        let result = get_match_result(&state, &params);
        assert_eq!(result, Some(0));
    }

    #[test]
    fn test_step_match() {
        let mut state = State::default();
        state.team_points = [20, 10];
        state.team_wins = [1, 1];
        state.match_steps = 5;
        step_match(&mut state, None);
        assert_eq!(state.team_points, [20, 10]);
        assert_eq!(state.team_wins, [1, 1]);
        assert_eq!(state.match_steps, 5 + 1);

        step_match(&mut state, Some(0));
        assert_eq!(state.team_points, [0, 0]);
        assert_eq!(state.team_wins, [2, 1]);
        assert_eq!(state.match_steps, 0);

        state.team_points = [20, 10];
        state.team_wins = [1, 1];
        state.match_steps = 5;
        step_match(&mut state, Some(1));
        assert_eq!(state.team_points, [0, 0]);
        assert_eq!(state.team_wins, [1, 2]);
        assert_eq!(state.match_steps, 0);
    }

    #[test]
    fn test_step_game() {
        let params = Params::default();
        let start_step = (params.max_steps_in_match + 1)
            * params.match_count_per_episode
            - 2;
        let mut total_steps = start_step;
        let mut game_over = false;
        let result =
            step_game(&mut total_steps, &mut game_over, &[2, 2], &params);
        assert_eq!(total_steps, start_step + 1);
        assert!(result.is_none());
        assert!(!game_over);

        let result =
            step_game(&mut total_steps, &mut game_over, &[3, 2], &params);
        assert_eq!(result, Some(0));
        assert!(game_over);

        total_steps -= 1;
        game_over = false;
        let result =
            step_game(&mut total_steps, &mut game_over, &[2, 3], &params);
        assert_eq!(result, Some(1));
        assert!(game_over);
    }

    #[test]
    fn test_step_game_finishes_early() {
        let params = Params::default();

        let mut total_steps = 0;
        let mut game_over = false;
        let result =
            step_game(&mut total_steps, &mut game_over, &[2, 1], &params);
        assert!(result.is_none());
        assert!(!game_over);

        let mut total_steps = 0;
        let mut game_over = false;
        let result =
            step_game(&mut total_steps, &mut game_over, &[3, 0], &params);
        assert_eq!(result, Some(0));
        assert!(game_over);

        let mut total_steps = 0;
        let mut game_over = false;
        let result =
            step_game(&mut total_steps, &mut game_over, &[1, 3], &params);
        assert_eq!(result, Some(1));
        assert!(game_over);
    }

    #[test]
    #[should_panic(expected = "Team wins tied")]
    fn test_step_game_panics() {
        let params = Params::default();
        let mut total_steps = (params.max_steps_in_match + 1)
            * params.match_count_per_episode
            - 1;
        step_game(&mut total_steps, &mut false, &[2, 2], &params);
    }

    #[test]
    fn test_get_observation() {
        let mut state = State::default();
        let vision_power_map = arr3(&[
            // P1 only sees top left corner [0-1, 0-1]
            [
                [1, 1, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            // P2 only sees bottom right corner [3-4, 3-4]
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1],
                [0, 0, 0, 1, 1],
            ],
        ]);
        // Player's units are always visible, but opposing units are visible only
        // when seen by sensor mask
        state.units = [
            vec![
                Unit::with_pos(Pos::new(0, 0)),
                Unit::with_pos(Pos::new(3, 3)),
            ],
            vec![
                Unit::with_pos(Pos::new(1, 1)),
                Unit::with_pos(Pos::new(4, 4)),
            ],
        ];
        state.asteroids = vec![
            // Visible to p1
            Pos::new(0, 1),
            // Invisible for all players
            Pos::new(2, 2),
            Pos::new(2, 3),
            // Visible to p2
            Pos::new(3, 4),
        ];
        state.nebulae = vec![
            // Visible to p1
            Pos::new(1, 0),
            // Invisible for all players
            Pos::new(0, 4),
            Pos::new(1, 4),
            // Visible to p2
            Pos::new(4, 3),
        ];
        state.relic_node_locations = vec![
            // Visible to p1
            Pos::new(0, 0),
            // Invisible for all players
            Pos::new(3, 0),
            Pos::new(3, 2),
            // Visible to p2
            Pos::new(4, 4),
        ];
        // Features always available
        state.team_points = [10, 20];
        state.team_wins = [1, 1];
        state.total_steps = 250;
        state.match_steps = 50;
        let expected_sensor_maps = [
            arr2(&[
                [true, true, false, false, false],
                [true, true, false, false, false],
                [false, false, false, false, false],
                [false, false, false, false, false],
                [false, false, false, false, false],
            ]),
            arr2(&[
                [false, false, false, false, false],
                [false, false, false, false, false],
                [false, false, false, false, false],
                [false, false, false, true, true],
                [false, false, false, true, true],
            ]),
        ];
        let mut expected_result = [
            Observation::new(
                0,
                expected_sensor_maps[0].clone(),
                state.team_points,
                state.team_wins,
                state.total_steps,
                state.match_steps,
            ),
            Observation::new(
                1,
                expected_sensor_maps[1].clone(),
                state.team_points,
                state.team_wins,
                state.total_steps,
                state.match_steps,
            ),
        ];

        expected_result[0].units = [
            vec![
                Unit::with_pos(Pos::new(0, 0)),
                Unit::with_pos(Pos::new(3, 3)),
            ],
            vec![Unit::with_pos(Pos::new(1, 1))],
        ];
        expected_result[1].units = [
            vec![Unit::with_pos(Pos::new(3, 3))],
            vec![
                Unit::with_pos(Pos::new(1, 1)),
                Unit::with_pos(Pos::new(4, 4)),
            ],
        ];

        expected_result[0].asteroids = vec![Pos::new(0, 1)];
        expected_result[1].asteroids = vec![Pos::new(3, 4)];

        expected_result[0].nebulae = vec![Pos::new(1, 0)];
        expected_result[1].nebulae = vec![Pos::new(4, 3)];

        expected_result[0].relic_node_locations = vec![Pos::new(0, 0)];
        expected_result[1].relic_node_locations = vec![Pos::new(4, 4)];

        let result = get_observation(&state, &vision_power_map);
        assert_eq!(result, expected_result);
    }

    #[rstest]
    #[case("replay_2202956.json")]
    fn test_full_game(#[case] file_name: &str) {
        let path = Path::new(file!())
            .parent()
            .unwrap()
            .join("test_data")
            .join(file_name);
        let json_data = fs::read_to_string(path).unwrap();
        let full_replay: FullReplay = serde_json::from_str(&json_data).unwrap();
        let all_states = full_replay.get_states();
        let all_vision_power_maps = full_replay.get_vision_power_maps();
        let all_energy_fields = full_replay.get_energy_fields();
        let mut rng = rand::thread_rng();
        let mut game_over = false;

        // Assert some constants are correct
        assert_eq!(full_replay.params.get_map_size(), MAP_SIZE);
        assert_eq!(full_replay.params.max_relic_nodes, MAX_RELIC_NODES);

        // Run the whole game checking the simulation matches on each step
        for (((s_next_s, actions), vision_power_map), energy_field) in
            all_states
                .windows(2)
                .zip_eq(full_replay.get_actions().iter())
                .zip_eq(all_vision_power_maps[1..].iter())
                .zip_eq(all_energy_fields[1..].iter())
        {
            let [state, next_state] = s_next_s else {
                panic!()
            };
            assert_eq!(
                get_energy_field(&state.energy_nodes, &full_replay.params),
                energy_field
            );

            let mut state = state.clone();
            let energy_node_deltas = state.get_energy_node_deltas(next_state);
            let ([p1_obs, p2_obs], game_result) = step(
                &mut state,
                &mut rng,
                actions,
                &full_replay.params,
                Some(
                    energy_node_deltas[0..energy_node_deltas.len() / 2].into(),
                ),
            );
            assert_eq!(
                stack![Axis(0), p1_obs.sensor_mask, p2_obs.sensor_mask],
                vision_power_map.map(|&v| v > 0)
            );

            state.sort();
            assert_eq!(state, *next_state);

            assert!(!game_over);
            game_over = state.done;
            assert_eq!(game_over, game_result.final_winner.is_some());
        }
        assert!(game_over);
    }
}
