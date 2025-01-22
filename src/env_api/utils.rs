use crate::env_api::env_data::{ActionInfoArraysView, ObsArraysView};
use crate::feature_engineering::action_space::write_basic_action_space;
use crate::feature_engineering::memory::Memory;
use crate::feature_engineering::obs_space::basic_obs_space::write_obs_arrays;
use crate::feature_engineering::unit_features::write_unit_features;
use crate::rules_engine::action::Action;
use crate::rules_engine::param_ranges::PARAM_RANGES;
use crate::rules_engine::params::{KnownVariableParams, FIXED_PARAMS, P};
use crate::rules_engine::state::Observation;
use itertools::Itertools;
use numpy::ndarray::{ArrayView2, ArrayView3};

pub fn action_array_to_vec(actions: ArrayView3<isize>) -> [Vec<Action>; P] {
    actions
        .outer_iter()
        .map(player_action_array_to_vec)
        .collect_vec()
        .try_into()
        .unwrap()
}

pub fn player_action_array_to_vec(actions: ArrayView2<isize>) -> Vec<Action> {
    actions
        .outer_iter()
        .map(|a| {
            let a: [isize; 3] = a.as_slice().unwrap().try_into().unwrap();
            Action::from(a)
        })
        .collect()
}

/// Writes the observations into the respective arrays and updates memories
/// Must be called *after* updating state and getting latest observation
pub fn update_memories_and_write_output_arrays(
    mut obs_slice: ObsArraysView,
    mut action_info_slice: ActionInfoArraysView,
    memories: &mut [Memory],
    observations: &[Observation],
    last_actions: &[Vec<Action>],
    params: &KnownVariableParams,
) {
    memories
        .iter_mut()
        .zip_eq(observations.iter())
        .zip_eq(last_actions.iter())
        .for_each(|((mem, obs), last_actions)| {
            mem.update(obs, last_actions, &FIXED_PARAMS, params)
        });
    write_obs_arrays(
        obs_slice.temporal_spatial_obs.view_mut(),
        obs_slice.nontemporal_spatial_obs.view_mut(),
        obs_slice.temporal_global_obs.view_mut(),
        obs_slice.nontemporal_global_obs.view_mut(),
        observations,
        memories,
        params,
    );
    let known_valuable_points_maps = memories
        .iter()
        .map(|m| m.get_relic_known_to_have_points_map().view())
        .collect_vec();
    write_basic_action_space(
        action_info_slice.action_mask.view_mut(),
        action_info_slice.sap_mask.view_mut(),
        observations,
        &known_valuable_points_maps,
        params,
        &PARAM_RANGES,
    );
    write_unit_features(
        action_info_slice.unit_indices.view_mut(),
        action_info_slice.unit_energies.view_mut(),
        action_info_slice.units_mask.view_mut(),
        observations,
    );
}
