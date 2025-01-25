use crate::feature_engineering::memory::masked_possibilities::MaskedPossibilities;
use crate::feature_engineering::utils::memory_error;
use crate::rules_engine::action::Action;
use crate::rules_engine::action::Action::{Down, Left, NoOp, Right, Sap, Up};
use crate::rules_engine::env::{
    estimate_vision_power_map, get_spawn_position, just_respawned,
    ENERGY_VOID_DELTAS,
};
use crate::rules_engine::param_ranges::ParamRanges;
use crate::rules_engine::params::{FixedParams, KnownVariableParams};
use crate::rules_engine::state::{Observation, Pos, Unit};
use itertools::Itertools;
use numpy::ndarray::{Array2, Zip};
use std::collections::BTreeMap;

#[derive(Debug, Default)]
pub struct HiddenParameterMemory {
    pub nebula_tile_vision_reduction: MaskedPossibilities<i32>,
    pub nebula_tile_energy_reduction: MaskedPossibilities<i32>,
    pub unit_sap_dropoff_factor: MaskedPossibilities<f32>,
    pub unit_energy_void_factor: MaskedPossibilities<f32>,
    last_obs_data: LastObservationData,
}

impl HiddenParameterMemory {
    pub fn new(param_ranges: &ParamRanges) -> Self {
        let nebula_tile_vision_reduction = MaskedPossibilities::from_options(
            param_ranges
                .nebula_tile_vision_reduction
                .iter()
                .copied()
                .sorted()
                .dedup()
                .collect(),
        );
        let nebula_tile_energy_reduction = MaskedPossibilities::from_options(
            param_ranges
                .nebula_tile_energy_reduction
                .iter()
                .copied()
                .sorted()
                .dedup()
                .collect(),
        );
        let unit_sap_dropoff_factor = MaskedPossibilities::from_options(
            param_ranges
                .unit_sap_dropoff_factor
                .iter()
                .copied()
                .sorted_by(|a, b| a.partial_cmp(b).unwrap())
                .dedup()
                .collect(),
        );
        let unit_energy_void_factor = MaskedPossibilities::from_options(
            param_ranges
                .unit_energy_void_factor
                .iter()
                .copied()
                .sorted_by(|a, b| a.partial_cmp(b).unwrap())
                .dedup()
                .collect(),
        );
        let last_obs_data = LastObservationData::default();
        Self {
            nebula_tile_vision_reduction,
            nebula_tile_energy_reduction,
            unit_sap_dropoff_factor,
            unit_energy_void_factor,
            last_obs_data,
        }
    }

    pub fn update(
        &mut self,
        obs: &Observation,
        last_actions: &[Action],
        fixed_params: &FixedParams,
        variable_params: &KnownVariableParams,
        nebulae_could_have_moved: bool,
    ) {
        let last_actions = get_realized_last_actions(
            last_actions,
            &self.last_obs_data,
            obs,
            fixed_params,
        );
        if self.nebula_tile_vision_reduction.still_unsolved() {
            determine_nebula_tile_vision_reduction(
                &mut self.nebula_tile_vision_reduction,
                obs,
                fixed_params.map_size,
                variable_params.unit_sensor_range,
                nebulae_could_have_moved,
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
                &last_actions,
                fixed_params,
                variable_params,
            );
        }
        let sap_count_maps = compute_sap_count_maps(
            &self.last_obs_data.my_units,
            &last_actions,
            fixed_params.map_size,
        );
        let aggregate_energy_void_map = compute_aggregate_energy_void_map(
            &self.last_obs_data.my_units,
            &last_actions,
            fixed_params.map_size,
            variable_params.unit_move_cost,
        );
        if self.unit_sap_dropoff_factor.still_unsolved()
            && !new_match_just_started
        {
            determine_unit_sap_dropoff_factor(
                &mut self.unit_sap_dropoff_factor,
                self.unit_energy_void_factor.get_solution().copied(),
                obs,
                &self.last_obs_data,
                &sap_count_maps,
                &aggregate_energy_void_map,
                fixed_params,
                variable_params,
            );
        }
        if self.unit_energy_void_factor.still_unsolved()
            && !new_match_just_started
        {
            determine_unit_energy_void_factor(
                &mut self.unit_energy_void_factor,
                self.unit_sap_dropoff_factor.get_solution().copied(),
                obs,
                &self.last_obs_data,
                &sap_count_maps,
                &aggregate_energy_void_map,
                fixed_params,
                variable_params,
            )
        }
        self.last_obs_data = LastObservationData::copy_from_obs(obs);
    }
}

#[derive(Debug, Default)]
struct LastObservationData {
    my_units: Vec<Unit>,
    opp_units: Vec<Unit>,
    nebulae: Vec<Pos>,
    sensor_mask: Array2<bool>,
}

impl LastObservationData {
    fn copy_from_obs(obs: &Observation) -> Self {
        Self {
            my_units: obs.get_my_units().to_vec(),
            opp_units: obs.get_opp_units().to_vec(),
            nebulae: obs.nebulae.clone(),
            sensor_mask: obs.sensor_mask.clone(),
        }
    }
}

/// Sometimes, the agent will try to move into an asteroid that it cannot see. In these cases
/// no energy is spent, so it's functionally the same as a NoOp.
fn get_realized_last_actions(
    last_actions: &[Action],
    last_obs: &LastObservationData,
    obs: &Observation,
    fixed_params: &FixedParams,
) -> Vec<Option<Action>> {
    let id_to_unit_now: BTreeMap<usize, Unit> =
        obs.get_my_units().iter().map(|u| (u.id, *u)).collect();
    let mut updated_actions =
        last_actions.iter().copied().map(Option::from).collect_vec();
    for (unit_last_turn, unit_now, expected_pos) in last_obs
        .my_units
        .iter()
        .filter(|u_last| u_last.alive())
        .filter_map(|u_last| match last_actions[u_last.id] {
            move_action @ (Up | Right | Down | Left) => {
                let new_pos = u_last
                    .pos
                    .maybe_translate(
                        move_action.as_move_delta().unwrap(),
                        fixed_params.map_size,
                    )
                    .unwrap();
                Some((u_last, id_to_unit_now.get(&u_last.id), new_pos))
            },
            NoOp | Sap(_) => None,
        })
        // If we could see where we were moving to, then it's known that there was no asteroid
        .filter(|(_, _, new_pos)| !last_obs.sensor_mask[new_pos.as_index()])
    {
        // If we can now see the tile, we can deduce what happened
        if obs.sensor_mask[expected_pos.as_index()] {
            if obs.asteroids.contains(&expected_pos) {
                updated_actions[unit_last_turn.id] = Some(NoOp);
            }
            continue;
        }
        // If our unit died and was removed on the spot (such as from a collision), then we can't
        // say for sure if the move succeeded or not
        let Some(unit_now) = unit_now else {
            updated_actions[unit_last_turn.id] = None;
            continue;
        };
        if just_respawned(unit_now, obs.match_steps, obs.team_id, fixed_params)
        {
            updated_actions[unit_last_turn.id] = None;
            continue;
        }
        // The move action succeeded
        if unit_now.pos == expected_pos {
            continue;
        }
        assert_eq!(
            unit_now.pos,
            unit_last_turn.pos,
            "{:?}",
            (unit_last_turn, unit_now)
        );
        updated_actions[unit_last_turn.id] = Some(NoOp);
    }
    updated_actions
}

fn determine_nebula_tile_vision_reduction(
    nebula_tile_vision_reduction: &mut MaskedPossibilities<i32>,
    obs: &Observation,
    map_size: [usize; 2],
    unit_sensor_range: usize,
    nebulae_could_have_moved: bool,
) {
    let nebula_tile_vision_reduction_backup_mask =
        nebula_tile_vision_reduction.mask.clone();
    let expected_vision_power_map = estimate_vision_power_map(
        obs.get_my_units(),
        map_size,
        unit_sensor_range,
    );
    if !nebulae_could_have_moved {
        for expected_vision in obs
            .nebulae
            .iter()
            .map(|n| expected_vision_power_map[n.as_index()])
        {
            nebula_tile_vision_reduction
                .iter_unmasked_options_mut_mask()
                .for_each(|(vision_reduction, mask)| {
                    if *vision_reduction >= expected_vision {
                        *mask = false
                    }
                });
        }
    }

    Zip::from(&expected_vision_power_map)
        .and(&obs.sensor_mask)
        .for_each(|expected_vision, can_see| {
            if *expected_vision > 0 && !can_see {
                nebula_tile_vision_reduction
                    .iter_unmasked_options_mut_mask()
                    .for_each(|(vision_reduction, mask)| {
                        if vision_reduction < expected_vision {
                            *mask = false
                        }
                    });
            }
        });

    if nebula_tile_vision_reduction.all_masked() {
        memory_error("nebula_tile_vision_reduction mask is all false");
        nebula_tile_vision_reduction.mask =
            nebula_tile_vision_reduction_backup_mask;
    }
}

fn determine_nebula_tile_energy_reduction(
    nebula_tile_energy_reduction_options: &mut MaskedPossibilities<i32>,
    obs: &Observation,
    last_obs: &LastObservationData,
    last_actions: &[Option<Action>],
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
    let options_before_update = nebula_tile_energy_reduction_options.clone();
    let id_to_unit_now: BTreeMap<usize, Unit> =
        obs.get_my_units().iter().map(|u| (u.id, *u)).collect();
    // NB: This assumes that units don't take invalid actions (like moving into an asteroid)
    for (energy_before_nebula, actual) in last_obs
        .my_units
        .iter()
        .filter_map(|unit_last_turn| {
            id_to_unit_now
                .get(&unit_last_turn.id)
                .map(|unit_now| (unit_last_turn, unit_now))
        })
        // Skip units that could have just respawned
        .filter(|(_, u_now)| {
            !just_respawned(u_now, obs.match_steps, obs.team_id, fixed_params)
        })
        .filter(|(_, unit_now)| {
            last_obs.nebulae.contains(&unit_now.pos)
                && unit_now.alive()
                && last_obs.opp_units.iter().all(|opp_u| {
                    // Skip units that we think could have been sapped, adjacent sapped,
                    // or affected by an energy void
                    let [dx, dy] = opp_u.pos.subtract(unit_now.pos);
                    dx.abs() > params.unit_sap_range + 1
                        || dy.abs() > params.unit_sap_range + 1
                })
        })
        .filter_map(|(unit_last_turn, unit_now)| {
            // The only way to be missing the last action is if it just respawned or is missing
            // from id_to_unit_now - both conditions that should be filtered away
            let energy_after_action = match last_actions[unit_last_turn.id]
                .expect("Missing last_action")
            {
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
            .iter_unmasked_options_mut_mask()
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
        // In edge cases, such as where we're sapped while unable to see the sapper,
        // reset the memory to what it was before this turn
        *nebula_tile_energy_reduction_options = options_before_update;
    }
}

fn compute_sap_count_maps(
    units_last_turn: &[Unit],
    last_actions: &[Option<Action>],
    map_size: [usize; 2],
) -> (BTreeMap<Pos, i32>, BTreeMap<Pos, i32>) {
    // NB: Assumes that all units that tried to sap had enough energy and were successful
    let mut sap_count = BTreeMap::new();
    let mut adjacent_sap_count = BTreeMap::new();
    for sap_target_pos in units_last_turn.iter().filter_map(|u| {
        if let Some(Sap(sap_deltas)) = last_actions[u.id] {
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

fn compute_aggregate_energy_void_map(
    units_last_turn: &[Unit],
    last_actions: &[Option<Action>],
    map_size: [usize; 2],
    unit_move_cost: i32,
) -> Option<Array2<i32>> {
    let mut result = Array2::<i32>::zeros(map_size);
    for (unit, action) in units_last_turn
        .iter()
        .filter(|u| u.alive())
        .map(|u| (u, last_actions[u.id]))
    {
        // A missing action means that we can't determine where the unit was when its
        // energy void was computed
        let action = action?;
        let (pos_after_action, void_energy) = match action {
            NoOp | Sap(_) => (unit.pos, unit.energy),
            direction @ (Up | Right | Down | Left) => (
                unit.pos
                    .maybe_translate(
                        direction.as_move_delta().unwrap(),
                        map_size,
                    )
                    .expect("Got invalid move action"),
                unit.energy - unit_move_cost,
            ),
        };
        for delta in ENERGY_VOID_DELTAS {
            let Some(void_pos) =
                pos_after_action.maybe_translate(delta, map_size)
            else {
                continue;
            };
            assert!(void_energy >= 0, "void_energy < 0");
            result[void_pos.as_index()] += void_energy;
        }
    }
    Some(result)
}

#[allow(clippy::too_many_arguments)]
fn determine_unit_sap_dropoff_factor(
    unit_sap_dropoff_factor: &mut MaskedPossibilities<f32>,
    unit_energy_void_factor: Option<f32>,
    obs: &Observation,
    last_obs_data: &LastObservationData,
    sap_count_maps: &(BTreeMap<Pos, i32>, BTreeMap<Pos, i32>),
    aggregate_energy_void_map: &Option<Array2<i32>>,
    fixed_params: &FixedParams,
    params: &KnownVariableParams,
) {
    let unit_sap_dropoff_factor_backup_mask =
        unit_sap_dropoff_factor.mask.clone();
    // Note that the environment resolution order goes:
    // - Move units
    // - Resolve sap actions
    // - Resolve energy field
    // - Move nebulae and energy nodes
    // Therefore, whenever we are comparing energies, we take the current unit's energy and
    // position and compare it to last turn's nebulae and energy field, minus any sap actions.
    // Confusingly enough, last turn's energy field is provided in this turn's observation.
    let (sap_count_map, adjacent_sap_count_map) = sap_count_maps;
    // NB: Assumes that units don't take invalid energy-wasting actions, like moving off the map
    let id_to_opp_unit_now: BTreeMap<usize, Unit> =
        obs.get_opp_units().iter().map(|u| (u.id, *u)).collect();
    let unit_counts_map = get_unit_counts_map(obs.get_opp_units());
    let mut relevant_data = Vec::new();
    for (
        opp_unit_last_turn,
        opp_unit_now,
        adj_sap_count,
        energy_void_delta,
        energy_field_delta,
    ) in last_obs_data
        .opp_units
        .iter()
        .filter_map(|u_last_turn| {
            id_to_opp_unit_now
                .get(&u_last_turn.id)
                .map(|u_now| (u_last_turn, u_now))
        })
        // Skip units that could have just respawned
        .filter(|(_, u_now)| {
            !just_respawned(
                u_now,
                obs.match_steps,
                obs.opp_team_id(),
                fixed_params,
            )
        })
        // Skip units in nebulae or that could be in nebulae
        .filter(|(_, u_now)| {
            !last_obs_data.nebulae.contains(&u_now.pos)
                && last_obs_data.sensor_mask[u_now.pos.as_index()]
        })
        .filter_map(|(u_last_turn, u_now)| {
            adjacent_sap_count_map
                .get(&u_now.pos)
                .map(|&count| (u_last_turn, u_now, count))
        })
        .filter_map(|(u_last_turn, u_now, adj_sap_count)| {
            let energy_void = aggregate_energy_void_map
                .as_ref()
                .map(|void_map| void_map[u_now.pos.as_index()])?;
            if energy_void == 0 {
                Some((u_last_turn, u_now, adj_sap_count, 0))
            } else if stacked_with_just_respawned_unit(
                *u_now,
                obs,
                fixed_params,
            ) {
                // Skip units that could be stacked with units that just respawned
                None
            } else if let Some(void_factor) = unit_energy_void_factor {
                let void_delta = void_factor * energy_void as f32
                    / unit_counts_map[&u_now.pos] as f32;
                Some((u_last_turn, u_now, adj_sap_count, void_delta as i32))
            } else {
                // Skip units that have lost energy to energy void if the
                // unit_energy_void_factor is still unknown
                None
            }
        })
        .map(|(u_last_turn, u_now, sap_count, void_delta)| {
            let energy_delta = obs.energy_field[u_now.pos.as_index()].unwrap();
            (u_last_turn, u_now, sap_count, void_delta, energy_delta)
        })
    {
        let direct_sap_loss = sap_count_map
            .get(&opp_unit_now.pos)
            .map_or(0, |count| *count * params.unit_sap_cost);
        relevant_data.push((
            opp_unit_last_turn,
            opp_unit_now,
            adj_sap_count,
            direct_sap_loss,
            energy_void_delta,
            energy_field_delta,
        ));
        for (&sap_dropoff_factor, mask) in
            unit_sap_dropoff_factor.iter_unmasked_options_mut_mask()
        {
            let adj_sap_loss = ((adj_sap_count * params.unit_sap_cost) as f32
                * sap_dropoff_factor) as i32;
            let energy_after_combat = opp_unit_last_turn.energy
                - energy_void_delta
                - direct_sap_loss
                - adj_sap_loss;
            if !is_possible_energy_level(
                *opp_unit_now,
                opp_unit_last_turn.pos,
                energy_after_combat,
                energy_field_delta,
                fixed_params,
                params,
            ) {
                *mask = false;
            }
        }
    }

    if unit_sap_dropoff_factor.all_masked() {
        unit_sap_dropoff_factor.mask =
            unit_sap_dropoff_factor_backup_mask.clone();
        // TODO: Clean up this error message
        let msg = format!(
            "unit_sap_dropoff_factor mask is all false\n{:?}\n{:?}\n\n{:?}",
            unit_sap_dropoff_factor, params, relevant_data
        );
        memory_error(&msg);
        unit_sap_dropoff_factor.mask = unit_sap_dropoff_factor_backup_mask;
    }
}

fn stacked_with_just_respawned_unit(
    unit: Unit,
    obs: &Observation,
    fixed_params: &FixedParams,
) -> bool {
    obs.match_steps.saturating_sub(1) % fixed_params.spawn_rate == 0
        && unit.pos
            == get_spawn_position(obs.opp_team_id(), fixed_params.map_size)
}

fn get_expected_energy(
    pre_field_energy: i32,
    energy_field_delta: i32,
    fixed_params: &FixedParams,
) -> i32 {
    if pre_field_energy >= 0 || pre_field_energy + energy_field_delta >= 0 {
        (pre_field_energy + energy_field_delta)
            .min(fixed_params.max_unit_energy)
            .max(fixed_params.min_unit_energy)
    } else {
        pre_field_energy
    }
}

fn is_possible_energy_level(
    unit_now: Unit,
    pos_last_turn: Pos,
    energy_after_combat: i32,
    energy_field_delta: i32,
    fixed_params: &FixedParams,
    params: &KnownVariableParams,
) -> bool {
    if unit_now.pos.subtract(pos_last_turn) == [0, 0] {
        // NoOp or Sap action was taken
        let expected_energy_noop = get_expected_energy(
            energy_after_combat,
            energy_field_delta,
            fixed_params,
        );
        let expected_energy_sap = get_expected_energy(
            energy_after_combat - params.unit_sap_cost,
            energy_field_delta,
            fixed_params,
        );
        expected_energy_noop == unit_now.energy
            || expected_energy_sap == unit_now.energy
    } else {
        // Move action was taken
        let expected_energy = get_expected_energy(
            energy_after_combat - params.unit_move_cost,
            energy_field_delta,
            fixed_params,
        );
        expected_energy == unit_now.energy
    }
}

#[allow(clippy::too_many_arguments)]
fn determine_unit_energy_void_factor(
    unit_energy_void_factor: &mut MaskedPossibilities<f32>,
    unit_sap_dropoff_factor: Option<f32>,
    obs: &Observation,
    last_obs_data: &LastObservationData,
    sap_count_maps: &(BTreeMap<Pos, i32>, BTreeMap<Pos, i32>),
    aggregate_energy_void_map: &Option<Array2<i32>>,
    fixed_params: &FixedParams,
    params: &KnownVariableParams,
) {
    let unit_energy_void_factor_backup_mask =
        unit_energy_void_factor.mask.clone();
    // Note that the environment resolution order goes:
    // - Move units
    // - Resolve sap actions
    // - Resolve energy field
    // - Move nebulae and energy nodes
    // Therefore, whenever we are comparing energies, we take the current unit's energy and
    // position and compare it to last turn's nebulae and energy field, minus any sap actions.
    // Confusingly enough, last turn's energy field is provided in this turn's observation.
    let (sap_count_map, adjacent_sap_count_map) = sap_count_maps;
    // NB: Assumes that units don't take invalid energy-wasting actions, like moving off the map
    let id_to_opp_unit_now: BTreeMap<usize, Unit> =
        obs.get_opp_units().iter().map(|u| (u.id, *u)).collect();
    let unit_counts_map = get_unit_counts_map(obs.get_opp_units());
    let mut relevant_data = Vec::new();
    for (
        opp_unit_last_turn,
        opp_unit_now,
        agg_energy_void,
        adj_sap_loss,
        energy_field_delta,
    ) in last_obs_data
        .opp_units
        .iter()
        .filter_map(|u_last_turn| {
            id_to_opp_unit_now
                .get(&u_last_turn.id)
                .map(|u_now| (u_last_turn, u_now))
        })
        // Skip units that could be stacked with units that just respawned
        .filter(|(_, u_now)| {
            !stacked_with_just_respawned_unit(**u_now, obs, fixed_params)
        })
        // Skip units in nebulae or that could be in nebulae
        .filter(|(_, u_now)| {
            !last_obs_data.nebulae.contains(&u_now.pos)
                && last_obs_data.sensor_mask[u_now.pos.as_index()]
        })
        .filter_map(|(u_last_turn, u_now)| {
            let agg_energy_void = aggregate_energy_void_map
                .as_ref()
                .map(|void_map| void_map[u_now.pos.as_index()]);
            match agg_energy_void {
                None | Some(0) => None,
                Some(void) => Some((u_last_turn, u_now, void)),
            }
        })
        .filter_map(|(u_last_turn, u_now, agg_energy_void)| {
            let adj_sap_count = adjacent_sap_count_map.get(&u_now.pos);
            match (adj_sap_count, unit_sap_dropoff_factor) {
                (None, _) => Some((u_last_turn, u_now, agg_energy_void, 0)),
                (_, None) => None,
                (Some(&sap_count), Some(sap_dropoff_factor)) => {
                    let adj_sap_loss = (sap_count * params.unit_sap_cost)
                        as f32
                        * sap_dropoff_factor;
                    Some((
                        u_last_turn,
                        u_now,
                        agg_energy_void,
                        adj_sap_loss as i32,
                    ))
                },
            }
        })
        .map(|(u_last_turn, u_now, energy_void, adj_sap_loss)| {
            let energy_delta = obs.energy_field[u_now.pos.as_index()].unwrap();
            (u_last_turn, u_now, energy_void, adj_sap_loss, energy_delta)
        })
    {
        let direct_sap_loss = sap_count_map
            .get(&opp_unit_now.pos)
            .map_or(0, |count| *count * params.unit_sap_cost);
        relevant_data.push((
            opp_unit_last_turn,
            opp_unit_now,
            agg_energy_void,
            direct_sap_loss,
            adj_sap_loss,
            energy_field_delta,
        ));
        for (&void_factor, mask) in
            unit_energy_void_factor.iter_unmasked_options_mut_mask()
        {
            let energy_void_loss = (void_factor * agg_energy_void as f32
                / unit_counts_map[&opp_unit_now.pos] as f32)
                as i32;
            let energy_after_combat = opp_unit_last_turn.energy
                - energy_void_loss
                - direct_sap_loss
                - adj_sap_loss;
            if !is_possible_energy_level(
                *opp_unit_now,
                opp_unit_last_turn.pos,
                energy_after_combat,
                energy_field_delta,
                fixed_params,
                params,
            ) {
                *mask = false;
            }
        }
    }

    if unit_energy_void_factor.all_masked() {
        unit_energy_void_factor.mask =
            unit_energy_void_factor_backup_mask.clone();
        // TODO: Clean up this error message
        let msg = format!(
            "unit_energy_void_factor mask is all false\n{:?}\n{:?}\n\n{:?}",
            unit_energy_void_factor, params, relevant_data
        );
        memory_error(&msg);
        unit_energy_void_factor.mask = unit_energy_void_factor_backup_mask;
    }
}

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
    use pretty_assertions::assert_eq as pretty_assert_eq;
    use rstest::rstest;

    #[test]
    fn test_get_realized_last_actions() {
        let last_actions = vec![Down, Sap([0, 0]), Down, Sap([0, 0]), Up, Up];
        let zipped_units = vec![
            // Base case
            (
                Unit::new(Pos::new(1, 1), 100, 0),
                Unit::new(Pos::new(1, 2), 100, 0),
            ),
            // Fails to move - replace with NoOp (and skips id = 1)
            (
                Unit::new(Pos::new(1, 1), 100, 2),
                Unit::new(Pos::new(1, 1), 100, 2),
            ),
            // Leaves sap actions as-is
            (
                Unit::new(Pos::new(1, 1), 100, 3),
                Unit::new(Pos::new(1, 1), 100, 3),
            ),
            // Includes units that just respawned if we can see their move destination
            (
                Unit::new(Pos::new(0, 1), 100, 4),
                Unit::new(Pos::new(0, 0), 100, 4),
            ),
            // Ignores units that just respawned if they could have run into a hidden asteroid
            (
                Unit::new(Pos::new(1, 1), 50, 5),
                Unit::new(Pos::new(0, 0), 100, 5),
            ),
        ];
        let (units_last_turn, units_now): (Vec<_>, Vec<_>) =
            zipped_units.into_iter().unzip();
        let obs = Observation {
            team_id: 0,
            units: [units_now, Vec::new()],
            sensor_mask: Array2::default(FIXED_PARAMS.map_size),
            match_steps: FIXED_PARAMS.spawn_rate + 1,
            ..Default::default()
        };
        let mut last_obs = LastObservationData {
            my_units: units_last_turn,
            sensor_mask: Array2::from_elem(FIXED_PARAMS.map_size, true),
            ..Default::default()
        };
        last_obs.sensor_mask[[1, 2]] = false;
        last_obs.sensor_mask[[1, 0]] = false;
        let result = get_realized_last_actions(
            &last_actions,
            &last_obs,
            &obs,
            &FIXED_PARAMS,
        );
        let expected_result = vec![
            Some(Down),
            Some(Sap([0, 0])),
            Some(NoOp),
            Some(Sap([0, 0])),
            Some(Up),
            None,
        ];
        assert_eq!(result, expected_result);
    }

    #[test]
    fn test_determine_nebula_tile_vision_reduction() {
        let mut possibilities =
            MaskedPossibilities::from_options(vec![0, 1, 2]);
        let map_size = [4, 4];
        let unit_sensor_range = 2;

        let obs = Observation {
            sensor_mask: arr2(&[
                [true, true, true, false],
                [true, true, true, true],
                [true, true, true, true],
                [false, false, false, false],
            ]),
            units: [vec![Unit::with_pos(Pos::new(0, 1))], Vec::new()],
            nebulae: vec![Pos::new(0, 0)],
            ..Default::default()
        };

        determine_nebula_tile_vision_reduction(
            &mut possibilities,
            &obs,
            map_size,
            unit_sensor_range,
            false,
        );
        assert_eq!(possibilities.get_mask(), vec![false, true, false]);
    }

    #[rstest]
    #[case([true, true, true])]
    #[should_panic(expected = "nebula_tile_vision_reduction mask is all false")]
    #[case([true, true, false])]
    fn test_determine_nebula_tile_vision_reduction_panics(
        #[case] mask: [bool; 3],
    ) {
        let mut possibilities =
            MaskedPossibilities::new(vec![0, 1, 2], mask.to_vec());
        let map_size = [3, 3];
        let unit_sensor_range = 2;

        let obs = Observation {
            sensor_mask: arr2(&[
                [true, false, true],
                [true, true, true],
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
            false,
        );
    }

    fn to_vec_some<T>(vec: Vec<T>) -> Vec<Option<T>> {
        vec.into_iter().map(Option::from).collect_vec()
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
            units: [my_units, Vec::new()],
            energy_field: arr2(
                &[[Some(2), Some(1), Some(0), Some(-1), Some(-2), None]; 6],
            ),
            ..Default::default()
        };
        let last_obs = LastObservationData {
            my_units: my_units_last_turn,
            opp_units: vec![Unit::with_pos(Pos::new(3, 3))],
            nebulae: vec![
                Pos::new(1, 0),
                Pos::new(1, 1),
                Pos::new(1, 2),
                Pos::new(1, 3),
                Pos::new(1, 4),
                Pos::new(1, 5),
                Pos::new(3, 3),
            ],
            sensor_mask: Array2::default(FIXED_PARAMS.map_size),
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
            &to_vec_some(last_actions),
            &fixed_params,
            &params,
        );
        assert_eq!(possibilities.get_mask(), expected_result);
    }

    #[rstest]
    #[case(
        vec![true, true, true],
        vec![false, false, true],
    )]
    #[case(
        vec![true, true, false],
        vec![true, true, false],
    )]
    fn test_determine_nebula_tile_energy_reduction_resets(
        #[case] mask_before_update: Vec<bool>,
        #[case] mask_after_update: Vec<bool>,
    ) {
        let mut possibilities =
            MaskedPossibilities::new(vec![0, 1, 2], mask_before_update);
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
            &to_vec_some(last_actions),
            &FIXED_PARAMS,
            &KnownVariableParams::default(),
        );
        assert_eq!(possibilities.get_mask(), &mask_after_update);
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
        let (sap_count, adjacent_sap_count) = compute_sap_count_maps(
            &units,
            &to_vec_some(actions),
            FIXED_PARAMS.map_size,
        );
        let expected_sap_count =
            BTreeMap::from([(Pos::new(2, 2), 1), (Pos::new(0, 0), 2)]);
        pretty_assert_eq!(sap_count, expected_sap_count);
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
        pretty_assert_eq!(adjacent_sap_count, expected_adjacent_sap_count);
    }

    #[test]
    #[should_panic(expected = "Invalid sap_deltas")]
    fn test_test_compute_sap_count_maps_panics() {
        let units = vec![Unit::new(Pos::new(0, 0), 0, 0)];
        let actions = vec![Sap([-1, -1])];
        compute_sap_count_maps(
            &units,
            &to_vec_some(actions),
            FIXED_PARAMS.map_size,
        );
    }

    #[test]
    fn test_compute_aggregate_energy_void_map() {
        let units = vec![
            Unit::new(Pos::new(0, 0), 100, 0),
            Unit::new(Pos::new(0, 2), 100, 2),
            Unit::new(Pos::new(0, 0), 50, 3),
            Unit::new(Pos::new(1, 1), 50, 4),
        ];
        let actions = vec![
            NoOp,
            // Skip unused action
            Left,
            NoOp,
            Down,
            Sap([0, 0]),
        ];
        let aggregate_energy_void_map = compute_aggregate_energy_void_map(
            &units,
            &to_vec_some(actions),
            [4, 4],
            5,
        );
        let expected_aggregate_energy_void_map = Some(arr2(&[
            [45, 250, 45, 100],
            [150, 45, 150, 0],
            [0, 50, 0, 0],
            [0, 0, 0, 0],
        ]));
        pretty_assert_eq!(
            aggregate_energy_void_map,
            expected_aggregate_energy_void_map
        );
    }

    #[test]
    #[should_panic(expected = "Got invalid move action")]
    fn test_compute_aggregate_energy_void_map_panics_on_invalid_move() {
        let units = vec![Unit::new(Pos::new(0, 0), 100, 0)];
        let actions = vec![Up];
        compute_aggregate_energy_void_map(
            &units,
            &to_vec_some(actions),
            FIXED_PARAMS.map_size,
            5,
        );
    }

    #[test]
    #[should_panic(expected = "void_energy < 0")]
    fn test_compute_aggregate_energy_void_map_panics_with_negative_energy() {
        let units = vec![Unit::new(Pos::new(0, 0), 4, 0)];
        let actions = vec![Down];
        compute_aggregate_energy_void_map(
            &units,
            &to_vec_some(actions),
            FIXED_PARAMS.map_size,
            5,
        );
    }

    #[rstest]
    // Not adjacent
    #[case(
        vec![Unit::new(Pos::new(0, 0), 10, 0)],
        vec![Unit::new(Pos::new(0, 0), 10, 0)],
        vec![Sap([1, 1])],
        vec![Unit::new(Pos::new(1, 1), 100, 0)],
        vec![Unit::new(Pos::new(1, 1), 91, 0)],
        None,
        vec![true, true, true],
    )]
    // Adjacent to sap
    #[case(
        vec![Unit::new(Pos::new(4, 4), 10, 0)],
        vec![Unit::new(Pos::new(4, 4), 10, 0)],
        vec![Sap([-3, -3])],
        vec![Unit::new(Pos::new(0, 0), 100, 0)],
        vec![Unit::new(Pos::new(0, 0), 90, 0)],
        None,
        vec![false, false, true],
    )]
    // Adjacent to sap, rounds down
    #[case(
        vec![Unit::new(Pos::new(4, 4), 10, 0)],
        vec![Unit::new(Pos::new(4, 4), 10, 0)],
        vec![Sap([-3, -3])],
        vec![Unit::new(Pos::new(0, 0), 100, 0)],
        vec![Unit::new(Pos::new(0, 0), 98, 0)],
        None,
        vec![true, false, false],
    )]
    // Adjacent to multiple sap actions
    #[case(
        vec![Unit::new(Pos::new(4, 4), 10, 0), Unit::new(Pos::new(4, 4), 10, 1)],
        vec![Unit::new(Pos::new(4, 4), 10, 0), Unit::new(Pos::new(4, 4), 10, 1)],
        vec![Sap([-3, -3]), Sap([-3, -3])],
        vec![Unit::new(Pos::new(0, 0), 100, 0)],
        vec![Unit::new(Pos::new(0, 0), 95, 0)],
        None,
        vec![true, false, false],
    )]
    // Ignore units in nebulae
    #[case(
        vec![Unit::new(Pos::new(0, 0), 10, 0)],
        vec![Unit::new(Pos::new(0, 0), 10, 0)],
        vec![Sap([0, 1])],
        vec![Unit::new(Pos::new(0, 2), 100, 0)],
        vec![Unit::new(Pos::new(0, 2), 75, 0)],
        None,
        vec![true, true, true],
    )]
    // Ignore units hit by energy void field
    #[case(
        vec![Unit::new(Pos::new(0, 1), 10, 0)],
        vec![Unit::new(Pos::new(0, 1), 10, 0)],
        vec![Sap([1, 0])],
        vec![Unit::new(Pos::new(0, 0), 100, 0)],
        vec![Unit::new(Pos::new(0, 0), 90, 0)],
        None,
        vec![true, true, true],
    )]
    // Ignore units hit by energy void field of now-dead unit
    #[case(
        vec![Unit::new(Pos::new(0, 1), 10, 0)],
        Vec::new(),
        vec![Sap([1, 0])],
        vec![Unit::new(Pos::new(0, 0), 100, 0)],
        vec![Unit::new(Pos::new(0, 0), 90, 0)],
        None,
        vec![true, true, true],
    )]
    // Include units hit by energy void field if unit_energy_void_factor is known
    #[case(
        vec![Unit::new(Pos::new(0, 1), 100, 0)],
        vec![Unit::new(Pos::new(0, 1), 100, 0)],
        vec![Sap([1, 0])],
        vec![Unit::new(Pos::new(0, 0), 100, 0)],
        vec![Unit::new(Pos::new(0, 0), 65, 0)],
        Some(0.25),
        vec![false, false, true],
    )]
    // Accounts for dead units and stacking units for energy void factor
    #[case(
        vec![Unit::new(Pos::new(0, 1), 100, 0)],
        Vec::new(),
        vec![Sap([1, 0])],
        vec![Unit::new(Pos::new(0, 0), 100, 0), Unit::new(Pos::new(0, 0), 90, 1)],
        vec![Unit::new(Pos::new(0, 0), 83, 0), Unit::new(Pos::new(0, 0), 73, 1)],
        Some(0.25),
        vec![false, true, false],
    )]
    // Counts move action cost
    #[case(
        vec![Unit::new(Pos::new(4, 4), 10, 0)],
        vec![Unit::new(Pos::new(4, 4), 10, 0)],
        vec![Sap([-3, -3])],
        vec![Unit::new(Pos::new(0, 1), 100, 0)],
        vec![Unit::new(Pos::new(0, 0), 93, 0)],
        None,
        vec![false, true, false],
    )]
    // Considers both NoOp and Sap actions for opposing units
    #[case(
        vec![
            Unit::new(Pos::new(4, 4), 10, 0),
            Unit::new(Pos::new(4, 4), 10, 1),
        ],
        vec![
            Unit::new(Pos::new(4, 4), 10, 0),
            Unit::new(Pos::new(4, 4), 10, 1),
        ],
        vec![Sap([-3, -3]), Sap([-3, -3])],
        vec![Unit::new(Pos::new(0, 0), 100, 0)],
        vec![Unit::new(Pos::new(0, 0), 80, 0)],
        None,
        vec![false, true, true],
    )]
    // Hit by direct and adjacent sap
    #[case(
        vec![
            Unit::new(Pos::new(4, 4), 10, 0),
            Unit::new(Pos::new(4, 4), 10, 1),
        ],
        vec![
            Unit::new(Pos::new(4, 4), 10, 0),
            Unit::new(Pos::new(4, 4), 10, 1),
        ],
        vec![Sap([-3, -3]), Sap([-4, -4])],
        vec![Unit::new(Pos::new(0, 0), 100, 0)],
        vec![Unit::new(Pos::new(0, 0), 85, 0)],
        None,
        vec![false, true, false],
    )]
    // Goes into negative energy after sap
    #[case(
        vec![Unit::new(Pos::new(1, 0), 100, 0)],
        vec![Unit::new(Pos::new(1, 0), 100, 0)],
        vec![Sap([0, 1])],
        vec![Unit::new(Pos::new(1, 1), 0, 0)],
        vec![Unit::new(Pos::new(1, 2), -7, 0)],
        None,
        vec![false, true, false],
    )]
    fn test_determine_unit_sap_dropoff_factor(
        #[case] my_units_last_turn: Vec<Unit>,
        #[case] my_units: Vec<Unit>,
        #[case] last_actions: Vec<Action>,
        #[case] opp_units_last_turn: Vec<Unit>,
        #[case] opp_units: Vec<Unit>,
        #[case] unit_energy_void_factor: Option<f32>,
        #[case] expected_result: Vec<bool>,
    ) {
        let mut possibilities =
            MaskedPossibilities::from_options(vec![0.25, 0.5, 1.0]);
        let obs = Observation {
            units: [my_units, opp_units],
            energy_field: arr2(&[[Some(0), Some(1), Some(2)]; 3]),
            ..Default::default()
        };
        let last_obs_data = LastObservationData {
            my_units: my_units_last_turn,
            opp_units: opp_units_last_turn,
            nebulae: vec![Pos::new(0, 2)],
            sensor_mask: Array2::from_elem(FIXED_PARAMS.map_size, true),
        };
        let params = KnownVariableParams {
            unit_move_cost: 2,
            unit_sap_cost: 10,
            ..Default::default()
        };
        let last_actions = to_vec_some(last_actions);
        let sap_count_maps = compute_sap_count_maps(
            &last_obs_data.my_units,
            &last_actions,
            FIXED_PARAMS.map_size,
        );
        let aggregate_energy_void_map = compute_aggregate_energy_void_map(
            &last_obs_data.my_units,
            &last_actions,
            FIXED_PARAMS.map_size,
            params.unit_move_cost,
        );
        determine_unit_sap_dropoff_factor(
            &mut possibilities,
            unit_energy_void_factor,
            &obs,
            &last_obs_data,
            &sap_count_maps,
            &aggregate_energy_void_map,
            &FIXED_PARAMS,
            &params,
        );
        assert_eq!(possibilities.get_mask(), expected_result);
    }

    #[rstest]
    #[case(vec![true, true, true])]
    #[should_panic(expected = "unit_sap_dropoff_factor mask is all false")]
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
            sensor_mask: Array2::from_elem(FIXED_PARAMS.map_size, true),
            ..Default::default()
        };
        let last_actions = vec![Sap([-3, -3]); FIXED_PARAMS.max_units];
        let params = KnownVariableParams {
            unit_move_cost: 2,
            unit_sap_cost: 10,
            ..Default::default()
        };
        let last_actions = to_vec_some(last_actions);
        let sap_count_maps = compute_sap_count_maps(
            &last_obs_data.my_units,
            &last_actions,
            FIXED_PARAMS.map_size,
        );
        let aggregate_energy_void_map = compute_aggregate_energy_void_map(
            &last_obs_data.my_units,
            &last_actions,
            FIXED_PARAMS.map_size,
            params.unit_move_cost,
        );
        determine_unit_sap_dropoff_factor(
            &mut possibilities,
            None,
            &obs,
            &last_obs_data,
            &sap_count_maps,
            &aggregate_energy_void_map,
            &FIXED_PARAMS,
            &params,
        );
    }

    #[rstest]
    // Not affected
    #[case(
        vec![Unit::new(Pos::new(0, 0), 10, 0)],
        vec![Unit::new(Pos::new(0, 0), 10, 0)],
        vec![NoOp],
        vec![Unit::new(Pos::new(1, 1), 100, 0)],
        vec![Unit::new(Pos::new(1, 1), 100, 0)],
        None,
        vec![true, true, true],
    )]
    // Hit by one unit's void factor
    #[case(
        vec![Unit::new(Pos::new(0, 0), 50, 0)],
        vec![Unit::new(Pos::new(0, 0), 50, 0)],
        vec![NoOp],
        vec![Unit::new(Pos::new(1, 0), 50, 0)],
        vec![Unit::new(Pos::new(1, 0), 25, 0)],
        None,
        vec![false, false, true],
    )]
    // Void factor rounds down
    #[case(
        vec![Unit::new(Pos::new(0, 0), 50, 0)],
        vec![Unit::new(Pos::new(0, 0), 50, 0)],
        vec![NoOp],
        vec![Unit::new(Pos::new(1, 0), 50, 0)],
        vec![Unit::new(Pos::new(1, 0), 38, 0)],
        None,
        vec![false, true, false],
    )]
    // Hit by multiple void factors
    #[case(
        vec![
            Unit::new(Pos::new(0, 0), 100, 0),
            Unit::new(Pos::new(0, 0), 50, 1),
        ],
        vec![
            Unit::new(Pos::new(0, 0), 100, 0),
            Unit::new(Pos::new(0, 0), 50, 1),
        ],
        vec![NoOp, NoOp],
        vec![Unit::new(Pos::new(1, 0), 100, 0)],
        vec![Unit::new(Pos::new(1, 0), 85, 0)],
        None,
        vec![true, false, false],
    )]
    // Damage is shared between units
    #[case(
        vec![Unit::new(Pos::new(0, 0), 52, 0)],
        vec![Unit::new(Pos::new(0, 0), 52, 0)],
        vec![NoOp],
        vec![
            Unit::new(Pos::new(1, 0), 50, 0),
            Unit::new(Pos::new(1, 0), 50, 1),
        ],
        vec![
            Unit::new(Pos::new(1, 0), 37, 0),
            Unit::new(Pos::new(1, 0), 37, 1),
        ],
        None,
        vec![false, false, true],
    )]
    // Includes energy field
    #[case(
        vec![Unit::new(Pos::new(0, 0), 50, 0)],
        vec![Unit::new(Pos::new(0, 0), 50, 0)],
        vec![NoOp],
        vec![Unit::new(Pos::new(0, 1), 50, 0)],
        vec![Unit::new(Pos::new(0, 1), 26, 0)],
        None,
        vec![false, false, true],
    )]
    // Goes into negative energy after damage
    #[case(
        vec![Unit::new(Pos::new(0, 0), 50, 0)],
        vec![Unit::new(Pos::new(0, 0), 50, 0)],
        vec![NoOp],
        vec![Unit::new(Pos::new(0, 1), 23, 0)],
        vec![Unit::new(Pos::new(0, 1), -2, 0)],
        None,
        vec![false, false, true],
    )]
    // Ignore units in nebulae
    #[case(
        vec![Unit::new(Pos::new(0, 1), 10, 0)],
        vec![Unit::new(Pos::new(0, 1), 10, 0)],
        vec![NoOp],
        vec![Unit::new(Pos::new(0, 2), 100, 0)],
        vec![Unit::new(Pos::new(0, 2), 75, 0)],
        None,
        vec![true, true, true],
    )]
    // Include direct sap damage
    #[case(
        vec![Unit::new(Pos::new(0, 0), 50, 0)],
        vec![Unit::new(Pos::new(0, 0), 50, 0)],
        vec![Sap([1, 0])],
        vec![Unit::new(Pos::new(1, 0), 50, 0)],
        vec![Unit::new(Pos::new(1, 0), 35, 0)],
        None,
        vec![true, false, false],
    )]
    // Ignore units hit by adjacent sap
    #[case(
        vec![Unit::new(Pos::new(0, 0), 50, 0)],
        vec![Unit::new(Pos::new(0, 0), 50, 0)],
        vec![Sap([1, 1])],
        vec![Unit::new(Pos::new(1, 0), 50, 0)],
        vec![Unit::new(Pos::new(1, 0), 35, 0)],
        None,
        vec![true, true, true],
    )]
    // Include units hit by adjacent sap if unit_sap_dropoff_factor is known
    #[case(
        vec![Unit::new(Pos::new(0, 0), 50, 0)],
        vec![Unit::new(Pos::new(0, 0), 50, 0)],
        vec![Sap([1, 1])],
        vec![Unit::new(Pos::new(1, 0), 50, 0)],
        vec![Unit::new(Pos::new(1, 0), 35, 0)],
        Some(1.),
        vec![true, false, false],
    )]
    // Counts move action cost
    #[case(
        vec![Unit::new(Pos::new(0, 0), 50, 0)],
        vec![Unit::new(Pos::new(0, 0), 50, 0)],
        vec![NoOp],
        vec![Unit::new(Pos::new(1, 1), 50, 0)],
        vec![Unit::new(Pos::new(1, 0), 23, 0)],
        None,
        vec![false, false, true],
    )]
    // Considers both NoOp and Sap actions for opposing units
    #[case(
        vec![Unit::new(Pos::new(0, 0), 25, 0)],
        vec![Unit::new(Pos::new(0, 0), 25, 0)],
        vec![NoOp],
        vec![Unit::new(Pos::new(1, 0), 50, 0)],
        vec![Unit::new(Pos::new(1, 0), 38, 0)],
        None,
        vec![true, false, true],
    )]
    fn test_determine_unit_energy_void_factor(
        #[case] my_units_last_turn: Vec<Unit>,
        #[case] my_units: Vec<Unit>,
        #[case] last_actions: Vec<Action>,
        #[case] opp_units_last_turn: Vec<Unit>,
        #[case] opp_units: Vec<Unit>,
        #[case] unit_sap_dropoff_factor: Option<f32>,
        #[case] expected_result: Vec<bool>,
    ) {
        // These aren't the real values, but are easier to write tests for
        let mut possibilities =
            MaskedPossibilities::from_options(vec![0.1, 0.25, 0.5]);
        let obs = Observation {
            units: [my_units, opp_units],
            energy_field: arr2(&[[Some(0), Some(1), Some(2)]; 3]),
            ..Default::default()
        };
        let last_obs_data = LastObservationData {
            my_units: my_units_last_turn,
            opp_units: opp_units_last_turn,
            nebulae: vec![Pos::new(0, 2)],
            sensor_mask: Array2::from_elem(FIXED_PARAMS.map_size, true),
        };
        let params = KnownVariableParams {
            unit_move_cost: 2,
            unit_sap_cost: 10,
            ..Default::default()
        };
        let last_actions = to_vec_some(last_actions);
        let sap_count_maps = compute_sap_count_maps(
            &last_obs_data.my_units,
            &last_actions,
            FIXED_PARAMS.map_size,
        );
        let aggregate_energy_void_map = compute_aggregate_energy_void_map(
            &last_obs_data.my_units,
            &last_actions,
            FIXED_PARAMS.map_size,
            params.unit_move_cost,
        );
        determine_unit_energy_void_factor(
            &mut possibilities,
            unit_sap_dropoff_factor,
            &obs,
            &last_obs_data,
            &sap_count_maps,
            &aggregate_energy_void_map,
            &FIXED_PARAMS,
            &params,
        );
        assert_eq!(possibilities.get_mask(), expected_result);
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
            Unit::with_pos(Pos::new(1, 0)),
        ];
        let expected_result = BTreeMap::from([
            (Pos::new(0, 0), 2),
            (Pos::new(0, 1), 3),
            (Pos::new(1, 0), 1),
        ]);
        let result = get_unit_counts_map(&units);
        pretty_assert_eq!(result, expected_result);
    }
}
