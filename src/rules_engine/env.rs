use super::action::Action;
use super::params::{FixedParams, VariableParams, FIXED_PARAMS};
use super::state::{EnergyNode, GameResult, Observation, Pos, State, Unit};
use super::step_stats::StepStats;
use itertools::Itertools;
use numpy::ndarray::{s, Array2, Array3, ArrayView2, ArrayView3, Axis, Zip};
use rand::distributions::{Distribution, Uniform};
use rand::prelude::ThreadRng;
use std::cmp::Ordering;
use std::mem;
use TerminationMode::{FinalStep, ThirdMatchWin};

pub const ENERGY_VOID_DELTAS: [[isize; 2]; 4] =
    [[-1, 0], [1, 0], [0, -1], [0, 1]];

#[derive(Debug, Clone, Copy)]
pub enum TerminationMode {
    FinalStep,
    ThirdMatchWin,
}

pub fn get_reset_observation(
    state: &State,
    params: &VariableParams,
) -> [Observation; 2] {
    assert_eq!(state.total_steps, 0);
    let vision_power_map = compute_vision_power_map_from_params(
        &state.units,
        &state.nebulae,
        FIXED_PARAMS.map_size,
        params,
    );
    get_observation(state, vision_power_map.view(), state.energy_field.view())
}

pub fn step(
    state: &mut State,
    rng: &mut ThreadRng,
    actions: &[Vec<Action>; 2],
    params: &VariableParams,
    termination: TerminationMode,
    energy_node_deltas: Option<Vec<[isize; 2]>>,
) -> ([Observation; 2], GameResult, StepStats) {
    if state.done {
        panic!("Game over, need to reset State")
    }

    if state.match_steps == 0 {
        state.units = [Vec::new(), Vec::new()];
    }
    remove_dead_units(&mut state.units);
    let actions = get_relevant_actions(actions, &state.units);
    let (noop_count, move_count, sap_count) = move_units(
        &mut state.units,
        get_map_mask(&state.asteroids, FIXED_PARAMS.map_size).view(),
        &actions,
        FIXED_PARAMS.map_size,
        params,
    );
    let energy_before_sapping = get_unit_energies(&state.units);
    let (sap_direct_hits, sap_adjacent_hits) = sap_units(
        &mut state.units,
        &energy_before_sapping,
        &actions,
        FIXED_PARAMS.map_size,
        params,
    );
    let (energy_void_field_deltas, units_lost_to_collision) =
        resolve_collisions_and_energy_void_fields(
            &mut state.units,
            &energy_before_sapping,
            FIXED_PARAMS.map_size,
            params,
        );
    let (energy_field_deltas, nebula_energy_deltas) = apply_energy_field(
        &mut state.units,
        state.energy_field.view(),
        get_map_mask(&state.nebulae, FIXED_PARAMS.map_size).view(),
        &FIXED_PARAMS,
        params,
    );
    let units_lost_to_energy = state
        .units
        .iter()
        .flatten()
        .filter(|unit| !unit.alive())
        .count() as u16;
    if state.match_steps % FIXED_PARAMS.spawn_rate == 0 {
        spawn_units(&mut state.units, &FIXED_PARAMS)
    }
    let vision_power_map = compute_vision_power_map_from_params(
        &state.units,
        &state.nebulae,
        FIXED_PARAMS.map_size,
        params,
    );
    let prev_energy_field = move_space_objects(
        state,
        rng,
        energy_node_deltas,
        params,
        &FIXED_PARAMS,
    );
    let points_scored = get_relic_points_scored(
        &state.units,
        state.relic_node_points_map.view(),
    );
    state
        .team_points
        .iter_mut()
        .zip_eq(points_scored)
        .for_each(|(total, scored)| *total += scored);
    let match_winner = get_match_result(state, FIXED_PARAMS.max_steps_in_match);
    let terminal_points_scored =
        match_winner.is_some().then_some(state.team_points);
    step_match(state, match_winner);
    let game_winner = step_game(
        &mut state.total_steps,
        &mut state.done,
        state.team_wins,
        &FIXED_PARAMS,
        termination,
    );
    let observation = if let Some(energy_field) = prev_energy_field {
        get_observation(state, vision_power_map.view(), energy_field.view())
    } else {
        get_observation(
            state,
            vision_power_map.view(),
            state.energy_field.view(),
        )
    };
    (
        observation,
        GameResult::new(points_scored, match_winner, game_winner, state.done),
        StepStats {
            terminal_points_scored,
            energy_field_deltas,
            nebula_energy_deltas,
            energy_void_field_deltas,
            units_lost_to_energy,
            units_lost_to_collision,
            noop_count,
            move_count,
            sap_count,
            sap_direct_hits,
            sap_adjacent_hits,
        },
    )
}

fn remove_dead_units(units: &mut [Vec<Unit>; 2]) {
    units[0].retain(|u| u.alive());
    units[1].retain(|u| u.alive());
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

/// Returns (noop_count, move_count, sap_count). Counts all actions, even invalid ones, but
/// only executes valid move actions in the environment
fn move_units(
    units: &mut [Vec<Unit>; 2],
    asteroid_mask: ArrayView2<bool>,
    actions: &[Vec<Action>; 2],
    map_size: [usize; 2],
    params: &VariableParams,
) -> (u16, u16, u16) {
    let mut noop_count = 0;
    let mut move_count = 0;
    let mut sap_count = 0;
    for (unit, action) in units.iter_mut().zip_eq(actions.iter()).flat_map(
        |(team_units, team_actions)| team_units.iter_mut().zip_eq(team_actions),
    ) {
        let deltas = match action {
            Action::Up | Action::Right | Action::Down | Action::Left => {
                move_count += 1;
                action.as_move_delta()
            },
            Action::NoOp => {
                noop_count += 1;
                continue;
            },
            Action::Sap(_) => {
                sap_count += 1;
                continue;
            },
        };
        if unit.energy < params.unit_move_cost {
            continue;
        };
        // This behavior is almost certainly a bug in the main simulator
        if deltas[0] < 0 || deltas[1] < 0 {
            let wrapped_pos = unit.pos.wrapped_translate(deltas, map_size);
            if asteroid_mask[wrapped_pos.as_index()] {
                continue;
            }
        };
        let new_pos = unit.pos.bounded_translate(deltas, map_size);
        if asteroid_mask[new_pos.as_index()] {
            continue;
        }
        unit.pos = new_pos;
        unit.energy -= params.unit_move_cost;
    }
    (noop_count, move_count, sap_count)
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
    map_size: [usize; 2],
    params: &VariableParams,
) -> (u16, u16) {
    let mut sap_direct_hits = 0;
    let mut sap_adjacent_hits = 0;
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
                unit.pos.maybe_translate(sap_deltas, map_size)
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

        sap_direct_hits += sap_count.iter().sum::<u16>();
        sap_adjacent_hits += adjacent_sap_count.iter().sum::<u16>();
        for ((saps, adj_saps), opp_u) in sap_count
            .into_iter()
            .zip_eq(adjacent_sap_count)
            .zip_eq(units[opp].iter_mut())
        {
            opp_u.energy -= i32::from(saps) * params.unit_sap_cost;
            let adj_sap_loss = (i32::from(adj_saps) * params.unit_sap_cost)
                as f32
                * params.unit_sap_dropoff_factor;
            opp_u.energy -= adj_sap_loss as i32;
        }
    }
    (sap_direct_hits, sap_adjacent_hits)
}

fn resolve_collisions_and_energy_void_fields(
    units: &mut [Vec<Unit>; 2],
    unit_energies: &[Vec<i32>; 2],
    map_size: [usize; 2],
    params: &VariableParams,
) -> (Vec<i32>, u16) {
    let unit_aggregate_energy_void_map =
        get_unit_aggregate_energy_void_map(units, unit_energies, map_size);
    let unit_counts_map = get_unit_counts_map(units, map_size);
    let unit_aggregate_energy_map =
        get_unit_aggregate_energy_map(units, unit_energies, map_size);

    let mut energy_void_field_deltas =
        Vec::with_capacity(units[0].len() + units[1].len());
    let mut units_lost_to_collision = 0;
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
            let void_field_loss = (params.unit_energy_void_factor
                * unit_aggregate_energy_void_map[[opp, x, y]]
                / unit_counts_map[[team, x, y]] as f32)
                as i32;
            unit.energy -= void_field_loss;
            energy_void_field_deltas.push(-void_field_loss);
        }

        let mut keep_iter =
            (0..units[team].len()).map(|i| !to_remove.contains(&i));
        units[team].retain(|_| keep_iter.next().unwrap());
        units_lost_to_collision += to_remove.len();
    }
    (energy_void_field_deltas, units_lost_to_collision as u16)
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
        .cartesian_product(ENERGY_VOID_DELTAS)
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
    energy_field: ArrayView2<i32>,
    nebula_mask: ArrayView2<bool>,
    fixed_params: &FixedParams,
    params: &VariableParams,
) -> (Vec<i32>, Vec<i32>) {
    let capacity = units[0].len() + units[1].len();
    let mut energy_field_deltas = Vec::with_capacity(capacity);
    let mut nebula_energy_deltas = Vec::with_capacity(capacity);
    for (unit, field_delta, nebula_delta) in units
        .iter_mut()
        .flat_map(|team_units| team_units.iter_mut())
        .map(|u| {
            let u_pos_idx = u.pos.as_index();
            let nebula_delta = if nebula_mask[u_pos_idx] {
                -params.nebula_tile_energy_reduction
            } else {
                0
            };
            (u, energy_field[u_pos_idx], nebula_delta)
        })
        .filter(|(u, df, dn)| u.energy >= 0 || u.energy + df + dn >= 0)
    {
        energy_field_deltas.push(field_delta);
        nebula_energy_deltas.push(nebula_delta);
        unit.energy = (unit.energy + field_delta + nebula_delta)
            .min(fixed_params.max_unit_energy)
            .max(fixed_params.min_unit_energy)
    }
    (energy_field_deltas, nebula_energy_deltas)
}

pub fn get_energy_field(
    energy_nodes: &[EnergyNode],
    fixed_params: &FixedParams,
) -> Array2<i32> {
    let width = fixed_params.map_width;
    let height = fixed_params.map_height;
    let mut energy_field_3d =
        Array3::zeros((fixed_params.max_energy_nodes, width, height));
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
            v.min(fixed_params.max_energy_per_tile)
                .max(fixed_params.min_energy_per_tile)
        })
}

fn get_dist(a: [usize; 2], b: [usize; 2]) -> f32 {
    let [x1, y1] = a;
    let [x2, y2] = b;
    let sum_of_squares =
        (x2 as f32 - x1 as f32).powi(2) + (y2 as f32 - y1 as f32).powi(2);
    sum_of_squares.sqrt()
}

fn spawn_units(units: &mut [Vec<Unit>; 2], fixed_params: &FixedParams) {
    for (team, team_units) in units
        .iter_mut()
        .enumerate()
        .filter(|(_, team_units)| team_units.len() < fixed_params.max_units)
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
            1 => Pos::new(
                fixed_params.map_width - 1,
                fixed_params.map_height - 1,
            ),
            n => panic!("this town ain't big enough for the {n} of us"),
        };

        let new_unit = Unit::new(pos, fixed_params.init_unit_energy, u_id);
        team_units.insert(u_id, new_unit);
    }
}

pub fn estimate_vision_power_map(
    units: &[Unit],
    map_size: [usize; 2],
    unit_sensor_range: usize,
) -> Array2<i32> {
    let vision_estimate = compute_vision_power_map(
        &[units.to_vec()],
        &Vec::new(),
        map_size,
        unit_sensor_range,
        0,
    );
    vision_estimate.index_axis_move(Axis(0), 0)
}

fn compute_vision_power_map_from_params(
    units: &[Vec<Unit>; 2],
    nebulae: &[Pos],
    map_size: [usize; 2],
    params: &VariableParams,
) -> Array3<i32> {
    compute_vision_power_map(
        units,
        nebulae,
        map_size,
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
                x.saturating_sub(range)..(x + range + 1).min(width),
                y.saturating_sub(range)..(y + range + 1).min(height),
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

/// Returns the energy field from before moving energy nodes, if it has changed
fn move_space_objects(
    state: &mut State,
    rng: &mut ThreadRng,
    energy_node_deltas: Option<Vec<[isize; 2]>>,
    params: &VariableParams,
    fixed_params: &FixedParams,
) -> Option<Array2<i32>> {
    if state.total_steps as f32 * params.nebula_tile_drift_speed % 1.0 == 0.0
        && params.nebula_tile_drift_speed != 0.0
    {
        let deltas = [
            params.nebula_tile_drift_speed.signum() as isize,
            -params.nebula_tile_drift_speed.signum() as isize,
        ];
        for pos in state.asteroids.iter_mut().chain(state.nebulae.iter_mut()) {
            *pos = pos.wrapped_translate(deltas, fixed_params.map_size)
        }
    }

    if state.total_steps as f32 * params.energy_node_drift_speed % 1.0 == 0.0 {
        let energy_node_deltas = energy_node_deltas.unwrap_or_else(|| {
            get_random_energy_node_deltas(rng, state.energy_nodes.len(), params)
        });
        // Chain with symmetric energy_node_deltas
        for (deltas, node) in energy_node_deltas
            .iter()
            .copied()
            .chain(energy_node_deltas.iter().map(|[dx, dy]| [-dy, -dx]))
            .zip_eq(state.energy_nodes.iter_mut())
        {
            node.pos = node.pos.bounded_translate(deltas, fixed_params.map_size)
        }
        let energy_field = mem::replace(
            &mut state.energy_field,
            get_energy_field(&state.energy_nodes, fixed_params),
        );
        Some(energy_field)
    } else {
        None
    }
}

fn get_random_energy_node_deltas(
    rng: &mut ThreadRng,
    energy_node_count: usize,
    params: &VariableParams,
) -> Vec<[isize; 2]> {
    let uniform = Uniform::new(
        -params.energy_node_drift_magnitude,
        params.energy_node_drift_magnitude,
    );
    (0..energy_node_count / 2)
        .map(|_| [uniform.sample(rng) as isize, uniform.sample(rng) as isize])
        .collect()
}

fn get_relic_points_scored(
    units: &[Vec<Unit>; 2],
    relic_node_points_map: ArrayView2<bool>,
) -> [u32; 2] {
    let mut points_scored = [0; 2];
    for (points, units) in points_scored.iter_mut().zip_eq(units) {
        let mut scored_positions = Vec::with_capacity(units.len());
        *points += units
            .iter()
            .map(|u| {
                if u.alive()
                    && relic_node_points_map[u.pos.as_index()]
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
    points_scored
}

fn get_match_result(state: &State, max_steps_in_match: u32) -> Option<u8> {
    if state.match_steps < max_steps_in_match {
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
    team_wins: [u32; 2],
    fixed_params: &FixedParams,
    termination: TerminationMode,
) -> Option<u8> {
    *total_steps += 1;
    match termination {
        FinalStep => {},
        ThirdMatchWin => {
            let early_result = match team_wins {
                [3, _] => Some(0),
                [_, 3] => Some(1),
                _ => None,
            };
            if early_result.is_some() {
                *game_over = true;
                return early_result;
            }
        },
    }

    if *total_steps
        < (fixed_params.max_steps_in_match + 1)
            * fixed_params.match_count_per_episode
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
    vision_power_map: ArrayView3<i32>,
    energy_field: ArrayView2<i32>,
) -> [Observation; 2] {
    let [p1_mask, p2_mask] = get_sensor_masks(vision_power_map);
    let p1_energy_field = get_masked_energy_field(p1_mask.view(), energy_field);
    let p2_energy_field = get_masked_energy_field(p2_mask.view(), energy_field);
    let mut observations = [
        Observation::new(
            0,
            p1_mask,
            p1_energy_field,
            state.team_points,
            state.team_wins,
            state.total_steps,
            state.match_steps,
        ),
        Observation::new(
            1,
            p2_mask,
            p2_energy_field,
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

fn get_sensor_masks(vision_power_map: ArrayView3<i32>) -> [Array2<bool>; 2] {
    [
        vision_power_map.slice(s![0, .., ..]).map(|&v| v > 0),
        vision_power_map.slice(s![1, .., ..]).map(|&v| v > 0),
    ]
}

fn get_masked_energy_field(
    vision_mask: ArrayView2<bool>,
    energy_field: ArrayView2<i32>,
) -> Array2<Option<i32>> {
    Zip::from(vision_mask)
        .and(energy_field)
        .map_collect(|visible, &e| visible.then_some(e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules_engine::replay::FullReplay;
    use crate::rules_engine::state::{Pos, Unit};
    use numpy::ndarray::{arr2, arr3, stack};
    use pretty_assertions::assert_eq;
    use rstest::rstest;
    use serde::Deserialize;
    use std::fs;
    use std::iter::once;
    use std::path::Path;

    #[test]
    #[should_panic(expected = "Game over")]
    fn test_step_panics_without_reset() {
        let params = VariableParams::default();
        let mut state = State {
            done: true,
            ..Default::default()
        };
        step(
            &mut state,
            &mut rand::thread_rng(),
            &[Vec::new(), Vec::new()],
            &params,
            FinalStep,
            None,
        );
    }

    #[test]
    fn test_move_units() {
        let params = VariableParams::default();
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
                // Unit takes NoOp action
                Unit::with_pos(Pos::new(0, 0)),
            ],
            vec![
                // Unit can't move off the top of the map, but still costs energy
                Unit::with_pos_and_energy(
                    Pos::new(23, 23),
                    params.unit_move_cost,
                ),
                // Unit can't move into an asteroid, costs no energy
                Unit::with_pos_and_energy(Pos::new(23, 23), 100),
                // Unit takes Sap action
                Unit::with_pos(Pos::new(23, 23)),
            ],
        ];
        let asteroid_mask =
            get_map_mask(&[Pos::new(23, 22)], FIXED_PARAMS.map_size);
        let actions = [
            vec![Action::Left, Action::Left, Action::Right, Action::NoOp],
            vec![Action::Down, Action::Up, Action::Sap([0, 0])],
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
                Unit::with_pos(Pos::new(0, 0)),
            ],
            vec![
                Unit::with_pos_and_energy(Pos::new(23, 23), 0),
                Unit::with_pos_and_energy(Pos::new(23, 23), 100),
                Unit::with_pos(Pos::new(23, 23)),
            ],
        ];
        let expected_noop_move_sap_counts = (1, 5, 1);
        let noop_move_sap_counts = move_units(
            &mut units,
            asteroid_mask.view(),
            &actions,
            FIXED_PARAMS.map_size,
            &params,
        );
        assert_eq!(units, expected_moved_units);
        assert_eq!(noop_move_sap_counts, expected_noop_move_sap_counts);
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
        let sap_cost = 5;
        let params = VariableParams {
            unit_sap_cost: sap_cost,
            unit_sap_dropoff_factor: 0.5,
            ..Default::default()
        };
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
        sap_units(
            &mut units,
            &unit_energies,
            &actions,
            FIXED_PARAMS.map_size,
            &params,
        );
        assert_eq!(units, expected_sapped_units);
    }

    #[test]
    fn test_resolve_collisions_and_energy_void_fields() {
        let params = VariableParams {
            unit_energy_void_factor: 0.25,
            ..Default::default()
        };
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
        let expected_energy_void_field_deltas =
            vec![0, 0, 0, 0, 0, -50, 0, 0, 0, -12, -12];
        let expected_units_lost_to_collision = 6;

        let unit_energies = get_unit_energies(&units);
        let (energy_void_field_deltas, units_lost_to_collision) =
            resolve_collisions_and_energy_void_fields(
                &mut units,
                &unit_energies,
                FIXED_PARAMS.map_size,
                &params,
            );
        assert_eq!(units, expected_result);
        assert_eq!(energy_void_field_deltas, expected_energy_void_field_deltas);
        assert_eq!(units_lost_to_collision, expected_units_lost_to_collision);
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
        let fixed_params = FixedParams {
            min_unit_energy: 0,
            max_unit_energy: 100,
            ..FIXED_PARAMS
        };
        let params = VariableParams {
            nebula_tile_energy_reduction: 10,
            ..Default::default()
        };
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
                    fixed_params.min_unit_energy,
                ),
                Unit::with_pos_and_energy(
                    Pos::new(1, 0),
                    fixed_params.min_unit_energy,
                ),
            ],
            vec![
                // Units can gain power
                Unit::with_pos_and_energy(Pos::new(1, 1), 20),
                // Units cannot gain power beyond max_unit_energy
                Unit::with_pos_and_energy(
                    Pos::new(1, 1),
                    fixed_params.max_unit_energy,
                ),
            ],
        ];
        let expected_energy_field_deltas = vec![2, 5, -10, 5, 10, 10];
        let expected_nebula_energy_deltas = vec![
            0,
            -params.nebula_tile_energy_reduction,
            0,
            -params.nebula_tile_energy_reduction,
            0,
            0,
        ];
        let (energy_field_deltas, nebula_energy_deltas) = apply_energy_field(
            &mut units,
            energy_field.view(),
            nebula_mask.view(),
            &fixed_params,
            &params,
        );
        assert_eq!(units, expected_result);
        assert_eq!(energy_field_deltas, expected_energy_field_deltas);
        assert_eq!(nebula_energy_deltas, expected_nebula_energy_deltas);
    }

    #[derive(Deserialize)]
    struct EnergyFieldTestCase {
        #[allow(dead_code)]
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
            FIXED_PARAMS.map_size,
            test_case.energy_field.iter().flatten().copied().collect(),
        )
        .unwrap();
        let result = get_energy_field(&energy_nodes, &FIXED_PARAMS);
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
        let fixed_params = FIXED_PARAMS;
        let mut units = [
            // Empty vector; should spawn unit with id 0
            vec![],
            // Vector missing id 0; add it
            vec![Unit::new(Pos::new(1, 1), 42, 1)],
        ];
        let expected_result = [
            // Empty vector; should spawn unit with id 0
            vec![Unit::new(Pos::new(0, 0), fixed_params.init_unit_energy, 0)],
            // Vector missing id 0; add it
            vec![
                Unit::new(Pos::new(23, 23), fixed_params.init_unit_energy, 0),
                Unit::new(Pos::new(1, 1), 42, 1),
            ],
        ];
        spawn_units(&mut units, &fixed_params);
        assert_eq!(units, expected_result);
    }

    #[test]
    fn test_spawn_units_id_last_and_max_units() {
        let fixed_params = FIXED_PARAMS;
        let mut units = [
            // Contains all IDs; should add next one
            vec![
                Unit::new(Pos::new(1, 1), -5, 0),
                Unit::new(Pos::new(1, 1), -5, 1),
                Unit::new(Pos::new(1, 1), -5, 2),
            ],
            // Contains max units; don't add any
            (0..fixed_params.max_units)
                .map(|id| Unit::new(Pos::new(9, 9), 42, id))
                .collect(),
        ];
        let expected_result = [
            // Contains all IDs; should add next one
            vec![
                Unit::new(Pos::new(1, 1), -5, 0),
                Unit::new(Pos::new(1, 1), -5, 1),
                Unit::new(Pos::new(1, 1), -5, 2),
                Unit::new(Pos::new(0, 0), fixed_params.init_unit_energy, 3),
            ],
            // Contains max units; don't add any
            units[1].clone(),
        ];
        spawn_units(&mut units, &fixed_params);
        assert_eq!(units, expected_result);
    }

    #[test]
    fn test_spawn_units_inserts_unit_id() {
        let fixed_params = FIXED_PARAMS;
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
                Unit::new(Pos::new(0, 0), fixed_params.init_unit_energy, 1),
                Unit::new(Pos::new(1, 1), 42, 2),
                Unit::new(Pos::new(1, 1), 42, 4),
            ],
            // Insert unit at correct location
            vec![
                Unit::new(Pos::new(9, 9), 42, 0),
                Unit::new(Pos::new(9, 9), 42, 1),
                Unit::new(Pos::new(23, 23), fixed_params.init_unit_energy, 2),
                Unit::new(Pos::new(9, 9), 42, 4),
            ],
        ];
        spawn_units(&mut units, &fixed_params);
        assert_eq!(units, expected_result);
    }

    #[test]
    fn test_compute_vision_power_map_one_unit() {
        let mut fixed_params = FIXED_PARAMS;
        fixed_params.set_map_size([5, 5]);
        let params = VariableParams {
            unit_sensor_range: 1,
            ..Default::default()
        };
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
            compute_vision_power_map_from_params(
                &units,
                &Vec::new(),
                fixed_params.map_size,
                &params
            ),
            expected_result
        );
    }

    #[test]
    fn test_compute_vision_power_map_handles_edge_of_map() {
        let mut fixed_params = FIXED_PARAMS;
        fixed_params.set_map_size([3, 3]);
        let params = VariableParams {
            unit_sensor_range: 1,
            ..Default::default()
        };
        let units = [
            vec![Unit::with_pos(Pos::new(0, 0))],
            vec![Unit::with_pos(Pos::new(2, 2))],
        ];
        let expected_result = arr3(&[
            [[2, 1, 0], [1, 1, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 1, 1], [0, 1, 2]],
        ]);
        assert_eq!(
            compute_vision_power_map_from_params(
                &units,
                &Vec::new(),
                fixed_params.map_size,
                &params
            ),
            expected_result
        );
    }

    #[test]
    fn test_compute_vision_power_map_is_additive_with_nebulae() {
        let mut fixed_params = FIXED_PARAMS;
        fixed_params.set_map_size([5, 5]);
        let params = VariableParams {
            unit_sensor_range: 1,
            nebula_tile_vision_reduction: 5,
            ..Default::default()
        };
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
            compute_vision_power_map_from_params(
                &units,
                &nebulae,
                fixed_params.map_size,
                &params
            ),
            expected_result
        );
    }

    #[test]
    fn test_move_space_objects() {
        let params = VariableParams {
            nebula_tile_drift_speed: -0.05,
            energy_node_drift_speed: 0.02,
            energy_node_drift_magnitude: 5.0,
            ..Default::default()
        };
        let mut state = State {
            asteroids: vec![
                // Moves normally
                Pos::new(10, 10),
                // Wraps as expected
                Pos::new(0, 0),
                Pos::new(23, 23),
            ],
            nebulae: vec![
                // Moves normally
                Pos::new(11, 11),
                // Wraps as expected
                Pos::new(0, 0),
                Pos::new(23, 23),
            ],
            energy_nodes: vec![
                // Moves normally
                EnergyNode::new_at(Pos::new(12, 12)),
                // Stops at edge of board
                EnergyNode::new_at(Pos::new(21, 22)),
                // Moves normally
                EnergyNode::new_at(Pos::new(14, 14)),
                // Stops at edge of board
                EnergyNode::new_at(Pos::new(1, 2)),
            ],
            ..Default::default()
        };
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
        move_space_objects(
            &mut state,
            &mut rand::thread_rng(),
            Some(energy_node_deltas),
            &params,
            &FIXED_PARAMS,
        );
        assert_eq!(state.asteroids, expected_asteroids);
        assert_eq!(state.nebulae, expected_nebulae);
        assert_eq!(state.energy_nodes, expected_energy_nodes);
    }

    #[test]
    fn test_move_space_objects_no_op() {
        let params = VariableParams {
            nebula_tile_drift_speed: -0.05,
            energy_node_drift_speed: 0.02,
            energy_node_drift_magnitude: 5.0,
            ..Default::default()
        };
        let mut state = State {
            asteroids: vec![Pos::new(1, 1)],
            nebulae: vec![Pos::new(2, 2)],
            energy_nodes: vec![
                EnergyNode::new_at(Pos::new(3, 3)),
                EnergyNode::new_at(Pos::new(4, 4)),
            ],
            total_steps: 7,
            ..Default::default()
        };
        let orig_state = state.clone();
        move_space_objects(
            &mut state,
            &mut rand::thread_rng(),
            Some(vec![[-1, 2]]),
            &params,
            &FIXED_PARAMS,
        );
        assert_eq!(state, orig_state);
    }

    #[test]
    fn test_get_relic_points_scored() {
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
                // Not alive - does not earn points
                Unit::with_pos_and_energy(Pos::new(0, 0), -1),
                // Does not earn points
                Unit::with_pos(Pos::new(0, 1)),
                // Earns points
                Unit::with_pos(Pos::new(1, 0)),
            ],
        ];
        let points_map = arr2(&[[true, false], [true, true]]);
        let result = get_relic_points_scored(&units, points_map.view());
        assert_eq!(result, [2, 1]);
    }

    #[test]
    fn test_get_match_result() {
        let max_steps_in_match = FIXED_PARAMS.max_steps_in_match;
        let mut state = State {
            team_points: [25, 24],
            match_steps: max_steps_in_match - 1,
            ..Default::default()
        };
        let result = get_match_result(&state, max_steps_in_match);
        assert!(result.is_none());

        state.match_steps += 1;
        let result = get_match_result(&state, max_steps_in_match);
        assert_eq!(result, Some(0));

        state.team_points = [25, 26];
        let result = get_match_result(&state, max_steps_in_match);
        assert_eq!(result, Some(1));
    }

    #[test]
    fn test_get_match_result_tiebreaks_points() {
        let max_steps_in_match = FIXED_PARAMS.max_steps_in_match;
        let mut state = State {
            team_points: [10, 10],
            match_steps: max_steps_in_match,
            units: [
                vec![Unit::with_energy(20), Unit::with_energy(30)],
                vec![Unit::with_energy(49)],
            ],
            ..Default::default()
        };
        let result = get_match_result(&state, max_steps_in_match);
        assert_eq!(result, Some(0));

        state.units = [
            vec![Unit::with_energy(20), Unit::with_energy(30)],
            vec![Unit::with_energy(51)],
        ];
        let result = get_match_result(&state, max_steps_in_match);
        assert_eq!(result, Some(1));

        // Ties favor player 1
        state.units = [vec![], vec![]];
        let result = get_match_result(&state, max_steps_in_match);
        assert_eq!(result, Some(0));
    }

    #[test]
    fn test_step_match() {
        let mut state = State {
            team_points: [20, 10],
            team_wins: [1, 1],
            match_steps: 5,
            ..Default::default()
        };
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
        let fixed_params = FIXED_PARAMS;
        let start_step = (fixed_params.max_steps_in_match + 1)
            * fixed_params.match_count_per_episode
            - 2;
        let mut total_steps = start_step;
        let mut game_over = false;
        let result = step_game(
            &mut total_steps,
            &mut game_over,
            [2, 2],
            &fixed_params,
            FinalStep,
        );
        assert_eq!(total_steps, start_step + 1);
        assert!(result.is_none());
        assert!(!game_over);

        let result = step_game(
            &mut total_steps,
            &mut game_over,
            [3, 2],
            &fixed_params,
            FinalStep,
        );
        assert_eq!(result, Some(0));
        assert!(game_over);

        total_steps -= 1;
        game_over = false;
        let result = step_game(
            &mut total_steps,
            &mut game_over,
            [2, 3],
            &fixed_params,
            FinalStep,
        );
        assert_eq!(result, Some(1));
        assert!(game_over);
    }

    #[rstest]
    #[case([2, 1], ThirdMatchWin, None)]
    #[case([3, 0], FinalStep, None)]
    #[case([3, 0], ThirdMatchWin, Some(0))]
    #[case([1, 3], FinalStep, None)]
    #[case([1, 3], ThirdMatchWin, Some(1))]
    fn test_step_game_finishes_early(
        #[case] team_wins: [u32; 2],
        #[case] termination: TerminationMode,
        #[case] expected_result: Option<u8>,
    ) {
        let mut game_over = false;
        let result = step_game(
            &mut 0,
            &mut game_over,
            team_wins,
            &FIXED_PARAMS,
            termination,
        );
        assert_eq!(result, expected_result);
        assert_eq!(game_over, expected_result.is_some());
    }

    #[test]
    #[should_panic(expected = "Team wins tied")]
    fn test_step_game_panics() {
        let fixed_params = FIXED_PARAMS;
        let mut total_steps = (fixed_params.max_steps_in_match + 1)
            * fixed_params.match_count_per_episode
            - 1;
        step_game(
            &mut total_steps,
            &mut false,
            [2, 2],
            &fixed_params,
            FinalStep,
        );
    }

    #[test]
    fn test_get_observation() {
        let vision_power_map = arr3(&[
            // P1 only sees top left corner [0-1, 0-1] and bottom left
            [
                [1, 1, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
            ],
            // P2 only sees bottom right corner [3-4, 3-4] and bottom left
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1],
                [1, 0, 0, 1, 1],
            ],
        ]);
        let energy_field = arr2(&[
            [0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9],
            [10, 11, 12, 13, 14],
            [15, 16, 17, 18, 19],
            [20, 21, 22, 23, 24],
        ]);
        let state = State {
            // Player's units are always visible, but opposing units are visible only
            // when seen by sensor mask
            units: [
                vec![
                    Unit::with_pos(Pos::new(0, 0)),
                    Unit::with_pos(Pos::new(3, 3)),
                ],
                vec![
                    Unit::with_pos(Pos::new(1, 1)),
                    Unit::with_pos(Pos::new(4, 4)),
                ],
            ],
            asteroids: vec![
                // Visible to p1
                Pos::new(0, 1),
                // Invisible for all players
                Pos::new(2, 2),
                Pos::new(2, 3),
                // Visible to p2
                Pos::new(3, 4),
            ],
            nebulae: vec![
                // Visible to p1
                Pos::new(1, 0),
                // Invisible for all players
                Pos::new(0, 4),
                Pos::new(1, 4),
                // Visible to p2
                Pos::new(4, 3),
            ],
            relic_node_locations: vec![
                // Visible to p1
                Pos::new(0, 0),
                // Invisible for all players
                Pos::new(3, 0),
                Pos::new(3, 2),
                // Visible to p2
                Pos::new(4, 4),
            ],
            // Features always available
            team_points: [10, 20],
            team_wins: [1, 1],
            total_steps: 250,
            match_steps: 50,
            ..Default::default()
        };
        let p1_expected_sensor_map = arr2(&[
            [true, true, false, false, false],
            [true, true, false, false, false],
            [false, false, false, false, false],
            [false, false, false, false, false],
            [true, false, false, false, false],
        ]);
        let p2_expected_sensor_map = arr2(&[
            [false, false, false, false, false],
            [false, false, false, false, false],
            [false, false, false, false, false],
            [false, false, false, true, true],
            [true, false, false, true, true],
        ]);
        let p1_expected_energy_field = arr2(&[
            [Some(0), Some(1), None, None, None],
            [Some(5), Some(6), None, None, None],
            [None; 5],
            [None; 5],
            [Some(20), None, None, None, None],
        ]);
        let p2_expected_energy_field = arr2(&[
            [None; 5],
            [None; 5],
            [None; 5],
            [None, None, None, Some(18), Some(19)],
            [Some(20), None, None, Some(23), Some(24)],
        ]);
        let mut expected_result = [
            Observation::new(
                0,
                p1_expected_sensor_map,
                p1_expected_energy_field,
                state.team_points,
                state.team_wins,
                state.total_steps,
                state.match_steps,
            ),
            Observation::new(
                1,
                p2_expected_sensor_map,
                p2_expected_energy_field,
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

        let result = get_observation(
            &state,
            vision_power_map.view(),
            energy_field.view(),
        );
        assert_eq!(result, expected_result);
    }

    #[rstest]
    // Two energy nodes
    #[ignore = "slow"]
    #[case("processed_replay_6155879.json")]
    // Four energy nodes
    #[ignore = "slow"]
    #[case("processed_replay_4086850.json")]
    // Six energy nodes
    #[ignore = "slow"]
    #[case("processed_replay_2462211601.json")]
    fn test_full_game(#[case] file_name: &str) {
        let path = Path::new(file!())
            .parent()
            .unwrap()
            .join("test_data")
            .join("processed_replays")
            .join(file_name);
        let json_data = fs::read_to_string(path).unwrap();
        let full_replay: FullReplay = serde_json::from_str(&json_data).unwrap();
        let all_states = full_replay.get_states();
        let all_vision_power_maps = full_replay.get_vision_power_maps();
        let mut rng = rand::thread_rng();
        let mut last_points = [0; 2];
        let mut game_over = false;

        // Assert fixed params are correct
        assert_eq!(full_replay.params.fixed, FIXED_PARAMS);

        let player_observations = full_replay.get_player_observations();
        let player_reset_obs =
            get_reset_observation(&all_states[0], &full_replay.params.variable);
        assert_eq!(player_reset_obs, player_observations[0]);
        // Run the whole game checking the simulation matches on each step
        for (
            ((s_next_s, actions), vision_power_map),
            expected_player_observations,
        ) in all_states
            .windows(2)
            .zip_eq(full_replay.get_actions().iter())
            .zip_eq(all_vision_power_maps[1..].iter())
            .zip_eq(
                player_observations[1..]
                    .iter()
                    .map(|obs| Some(obs.clone()))
                    .chain(once(None)),
            )
        {
            let [state, next_state] = s_next_s else {
                panic!()
            };
            let mut state = state.clone();
            let mut next_state = next_state.clone();
            // Currently in the replay file, each observed energy field is from the previous
            // step's computed energy field
            state.energy_field = get_energy_field(
                &state.energy_nodes,
                &full_replay.params.fixed,
            );
            assert_eq!(state.energy_field, next_state.energy_field,);

            let energy_node_deltas = state.get_energy_node_deltas(&next_state);
            let ([mut p1_obs, mut p2_obs], game_result, _) = step(
                &mut state,
                &mut rng,
                actions,
                &full_replay.params.variable,
                FinalStep,
                Some(
                    energy_node_deltas[0..energy_node_deltas.len() / 2]
                        .to_vec(),
                ),
            );
            assert_eq!(
                stack![Axis(0), p1_obs.sensor_mask, p2_obs.sensor_mask],
                vision_power_map.map(|&v| v > 0)
            );
            if let Some([mut p1_expected, mut p2_expected]) =
                expected_player_observations
            {
                p1_obs.sort();
                p2_obs.sort();
                p1_expected.sort();
                p2_expected.sort();
                assert_eq!([p1_obs, p2_obs], [p1_expected, p2_expected]);
            }

            state.energy_field = Array2::zeros(state.energy_field.dim());
            state.sort();
            next_state.energy_field =
                Array2::zeros(next_state.energy_field.dim());
            next_state.sort();
            assert_eq!(state, next_state);

            if state.match_steps == 0 {
                last_points.fill(0);
                assert_eq!(game_result.match_winner.is_some(), true);
            } else {
                last_points
                    .iter_mut()
                    .zip_eq(game_result.points_scored.into_iter())
                    .for_each(|(total, scored)| *total += scored);
            }
            assert_eq!(state.team_points, last_points);
            assert!(!game_over);
            game_over = game_result.done;
            assert_eq!(game_over, game_result.final_winner.is_some());
        }
        assert!(game_over);
    }
}
