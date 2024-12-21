use crate::env_api::env_data::{
    ActionInfoArraysView, ObsArraysView, PlayerData,
};
use crate::feature_engineering::action_space::write_basic_action_space;
use crate::feature_engineering::obs_space::basic_obs_space::write_obs_arrays;
use crate::feature_engineering::unit_features::write_unit_features;
use crate::rules_engine::action::Action;
use crate::rules_engine::params::{FIXED_PARAMS, P};
use crate::rules_engine::state::Observation;
use itertools::Itertools;
use numpy::ndarray::ArrayView3;

pub fn action_array_to_vec(actions: ArrayView3<isize>) -> [Vec<Action>; P] {
    actions
        .outer_iter()
        .map(|player_actions| {
            player_actions
                .outer_iter()
                .map(|a| {
                    let a: [isize; 3] =
                        a.as_slice().unwrap().try_into().unwrap();
                    Action::from(a)
                })
                .collect_vec()
        })
        .collect_vec()
        .try_into()
        .unwrap()
}

/// Writes the observations into the respective arrays and updates memories
/// Must be called *after* updating state and getting latest observation
pub fn update_memories_and_write_output_arrays(
    mut obs_slice: ObsArraysView,
    mut action_info_slice: ActionInfoArraysView,
    player_data: &mut PlayerData,
    observations: &[Observation; P],
    last_actions: &[Vec<Action>; P],
) {
    player_data
        .memories
        .iter_mut()
        .zip_eq(observations.iter())
        .zip_eq(last_actions.iter())
        .for_each(|((mem, obs), last_actions)| {
            mem.update(
                obs,
                last_actions,
                &FIXED_PARAMS,
                &player_data.known_params,
            )
        });
    write_obs_arrays(
        obs_slice.spatial_obs.view_mut(),
        obs_slice.global_obs.view_mut(),
        observations,
        &player_data.memories,
    );
    write_basic_action_space(
        action_info_slice.action_mask.view_mut(),
        action_info_slice.sap_mask.view_mut(),
        observations,
        &player_data.known_params,
    );
    write_unit_features(
        action_info_slice.unit_indices.view_mut(),
        action_info_slice.unit_energies.view_mut(),
        action_info_slice.units_mask.view_mut(),
        observations,
    );
}
