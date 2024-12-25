use crate::env_api::env_data::{
    ActionInfoArraysView, ObsArraysView, PlayerData, SingleEnvView,
};
use crate::env_api::utils::{
    action_array_to_vec, update_memories_and_write_output_arrays,
};
use crate::feature_engineering::obs_space::basic_obs_space::{
    get_global_feature_count, get_spatial_feature_count,
};
use crate::feature_engineering::reward_space::RewardSpace;
use crate::izip_eq;
use crate::rules_engine::action::Action;
use crate::rules_engine::env::{get_energy_field, get_reset_observation, step};
use crate::rules_engine::game_stats::GameStats;
use crate::rules_engine::param_ranges::PARAM_RANGES;
use crate::rules_engine::params::{
    KnownVariableParams, VariableParams, FIXED_PARAMS, P,
};
use crate::rules_engine::state::from_array::{
    get_asteroids, get_energy_nodes, get_nebulae,
};
use crate::rules_engine::state::{Pos, State};
use itertools::Itertools;
use numpy::ndarray::{
    stack, Array1, Array2, Array3, Array4, Array5, ArrayView2, Axis,
};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyArray4, PyArray5,
    PyArrayMethods, PyReadonlyArray2, PyReadonlyArray3, PyReadonlyArray4,
};
use pyo3::prelude::*;
use rand::rngs::ThreadRng;
use rayon::prelude::*;
use std::collections::HashMap;
use strum::EnumCount;

type PyStatsOutputs<'py> = (
    HashMap<String, f32>,
    HashMap<String, Bound<'py, PyArray1<f32>>>,
);
type PyEnvOutputs<'py> = (
    (Bound<'py, PyArray5<f32>>, Bound<'py, PyArray3<f32>>),
    (
        Bound<'py, PyArray4<bool>>,
        Bound<'py, PyArray5<bool>>,
        Bound<'py, PyArray4<isize>>,
        Bound<'py, PyArray3<f32>>,
        Bound<'py, PyArray3<bool>>,
    ),
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray1<bool>>,
    Option<PyStatsOutputs<'py>>,
);

#[pyclass]
pub struct ParallelEnv {
    n_envs: usize,
    reward_space: RewardSpace,
    env_data: Vec<EnvData>,
}

#[pymethods]
impl ParallelEnv {
    #[new]
    fn new(n_envs: usize, reward_space: RewardSpace) -> Self {
        let env_data = (0..n_envs).map(|_| EnvData::new()).collect();
        Self {
            n_envs,
            reward_space,
            env_data,
        }
    }

    /// Manually terminates the specified env IDs
    fn terminate_envs(&mut self, env_ids: Vec<usize>) {
        env_ids
            .into_iter()
            .for_each(|env_id| self.env_data[env_id].terminate())
    }

    fn get_empty_outputs<'py>(&self, py: Python<'py>) -> PyEnvOutputs<'py> {
        ParallelEnvOutputs::new(self.n_envs).into_pyarray_bound(py)
    }

    /// The environments that are starting a new match (there are up to 5 matches in a game)
    fn get_new_match_envs(&self) -> Vec<usize> {
        self.env_data
            .iter()
            .enumerate()
            .filter_map(|(i, ed)| ed.is_new_match().then_some(i))
            .collect()
    }

    /// The environments that are starting a new game (each game consists of five matches)
    fn get_new_game_envs(&self) -> Vec<usize> {
        self.env_data
            .iter()
            .enumerate()
            .filter_map(|(i, ed)| ed.is_new_game().then_some(i))
            .collect()
    }

    /// Resets all environments that are done, leaving active environments as-is. \
    /// Does not update reward or done arrays.
    /// de = envs that are done
    /// P = player count
    /// - obs_arrays: output arrays from self.step()
    /// - tile_type: (de, width, height)
    /// - energy_nodes: (de, max_energy_nodes, P)
    /// - energy_node_fns: (de, max_energy_nodes, 4)
    /// - energy_nodes_mask: (de, max_energy_nodes)
    /// - relic_nodes: (de, max_relic_nodes, P)
    /// - relic_node_configs: (de, max_relic_nodes, K, K) for a KxK relic configuration
    /// - relic_nodes_mask: (de, max_relic_nodes)
    #[allow(clippy::too_many_arguments)]
    fn soft_reset<'py>(
        &mut self,
        output_arrays: PyEnvOutputs<'py>,
        tile_type: PyReadonlyArray3<'py, i32>,
        energy_nodes: PyReadonlyArray3<'py, i16>,
        energy_node_fns: PyReadonlyArray3<'py, f32>,
        energy_nodes_mask: PyReadonlyArray2<'py, bool>,
        relic_nodes: PyReadonlyArray3<'py, i16>,
        relic_node_configs: PyReadonlyArray4<'py, bool>,
        relic_nodes_mask: PyReadonlyArray2<'py, bool>,
    ) {
        let mut rng = rand::thread_rng();
        let (
            (spatial_obs, global_obs),
            (action_mask, sap_mask, unit_indices, unit_energies, units_mask),
            reward,
            done,
            _,
        ) = output_arrays;
        for (
            (mut slice, env_data),
            tile_type,
            energy_nodes,
            energy_node_fns,
            energy_nodes_mask,
            relic_nodes,
            relic_node_configs,
            relic_nodes_mask,
        ) in izip_eq!(
            izip_eq!(
                spatial_obs.readwrite().as_array_mut().outer_iter_mut(),
                global_obs.readwrite().as_array_mut().outer_iter_mut(),
                action_mask.readwrite().as_array_mut().outer_iter_mut(),
                sap_mask.readwrite().as_array_mut().outer_iter_mut(),
                unit_indices.readwrite().as_array_mut().outer_iter_mut(),
                unit_energies.readwrite().as_array_mut().outer_iter_mut(),
                units_mask.readwrite().as_array_mut().outer_iter_mut(),
                reward.readwrite().as_array_mut().outer_iter_mut(),
                done.readwrite().as_array_mut().iter_mut()
            )
            .map(
                |(
                    spatial_obs,
                    global_obs,
                    action_mask,
                    sap_mask,
                    unit_indices,
                    unit_energies,
                    units_mask,
                    reward,
                    done,
                )| {
                    let obs_arrays = ObsArraysView {
                        spatial_obs,
                        global_obs,
                    };
                    let action_info_arrays = ActionInfoArraysView {
                        action_mask,
                        sap_mask,
                        unit_indices,
                        unit_energies,
                        units_mask,
                    };
                    SingleEnvView {
                        obs_arrays,
                        action_info_arrays,
                        reward,
                        done,
                    }
                },
            )
            .zip_eq(self.env_data.iter_mut())
            .filter(|(_, ed)| ed.done()),
            tile_type.as_array().outer_iter(),
            energy_nodes.as_array().outer_iter(),
            energy_node_fns.as_array().outer_iter(),
            energy_nodes_mask.as_array().outer_iter(),
            relic_nodes.as_array().outer_iter(),
            relic_node_configs.as_array().outer_iter(),
            relic_nodes_mask.as_array().outer_iter(),
        ) {
            let mut state = State {
                asteroids: get_asteroids(tile_type),
                nebulae: get_nebulae(tile_type),
                energy_nodes: get_energy_nodes(
                    energy_nodes,
                    energy_node_fns,
                    energy_nodes_mask,
                ),
                ..Default::default()
            };
            let (locations, configs): (_, Vec<ArrayView2<bool>>) = relic_nodes
                .mapv(|x| x as usize)
                .outer_iter()
                .map(|pos| Pos::from(pos.as_slice().unwrap()))
                .zip_eq(relic_node_configs.outer_iter())
                .zip_eq(relic_nodes_mask.iter())
                .filter_map(|(data, mask)| mask.then_some(data))
                .unzip();
            state.set_relic_nodes(
                locations,
                stack(Axis(0), &configs).unwrap().view(),
                FIXED_PARAMS.map_size,
                FIXED_PARAMS.relic_config_size,
            );
            state.energy_field =
                get_energy_field(&state.energy_nodes, &FIXED_PARAMS);
            // We randomly generate params here, as they aren't needed when generating the
            // initial map state in python
            let params = PARAM_RANGES.random_params(&mut rng);
            *env_data = EnvData::from_state_params(state, params);

            let obs = get_reset_observation(&env_data.state, &env_data.params);
            slice.obs_arrays.reset();
            slice.action_info_arrays.reset();
            update_memories_and_write_output_arrays(
                slice.obs_arrays,
                slice.action_info_arrays,
                &mut env_data.player_data.memories,
                &obs,
                &[Vec::new(), Vec::new()],
                &env_data.player_data.known_params,
            );
        }
    }

    fn seq_step<'py>(
        &mut self,
        py: Python<'py>,
        actions: PyReadonlyArray4<'py, isize>,
    ) -> PyEnvOutputs<'py> {
        let actions = actions.as_array();
        assert_eq!(actions.dim(), (self.n_envs, P, FIXED_PARAMS.max_units, 3));
        let mut out = ParallelEnvOutputs::new(self.n_envs);
        let mut rng = rand::thread_rng();
        for ((env_data, slice), actions) in self
            .env_data
            .iter_mut()
            .zip_eq(out.iter_env_slices_mut())
            .zip_eq(actions.outer_iter())
        {
            let actions = action_array_to_vec(actions);
            Self::step_env(
                env_data,
                &mut rng,
                slice,
                &actions,
                self.reward_space,
            );
        }
        self.write_game_stats(&mut out.stats);
        out.into_pyarray_bound(py)
    }

    fn par_step<'py>(
        &mut self,
        py: Python<'py>,
        actions: PyReadonlyArray4<'py, isize>,
    ) -> PyEnvOutputs<'py> {
        let actions = actions.as_array();
        assert_eq!(actions.dim(), (self.n_envs, P, FIXED_PARAMS.max_units, 3));
        let mut out = ParallelEnvOutputs::new(self.n_envs);
        self.env_data
            .iter_mut()
            .zip_eq(out.iter_env_slices_mut())
            .zip_eq(actions.outer_iter())
            .par_bridge()
            .map_init(rand::thread_rng, |rng, ((env_data, slice), actions)| {
                let actions = action_array_to_vec(actions);
                Self::step_env(
                    env_data,
                    rng,
                    slice,
                    &actions,
                    self.reward_space,
                );
            })
            .for_each(|_| {});
        self.write_game_stats(&mut out.stats);
        out.into_pyarray_bound(py)
    }
}

impl ParallelEnv {
    fn step_env(
        env_data: &mut EnvData,
        rng: &mut ThreadRng,
        mut env_slice: SingleEnvView,
        actions: &[Vec<Action>; P],
        reward_space: RewardSpace,
    ) {
        let (obs, result, stats) = step(
            &mut env_data.state,
            rng,
            actions,
            &env_data.params,
            reward_space.termination_mode(),
            None,
        );
        env_data.stats.extend(stats);
        update_memories_and_write_output_arrays(
            env_slice.obs_arrays,
            env_slice.action_info_arrays,
            &mut env_data.player_data.memories,
            &obs,
            actions,
            &env_data.player_data.known_params,
        );
        env_slice
            .reward
            .iter_mut()
            .zip_eq(reward_space.get_reward(result))
            .for_each(|(slice_reward, r)| *slice_reward = r);
        *env_slice.done = result.done;
    }
    fn write_game_stats(
        &self,
        stats: &mut Option<ParallelEnvGameStatsOutputs>,
    ) {
        let all_game_stats = self
            .env_data
            .iter()
            .filter(|ed| ed.done())
            .map(|ed| &ed.stats)
            .collect_vec();
        if all_game_stats.is_empty() {
            *stats = None;
            return;
        }

        *stats =
            Some(ParallelEnvGameStatsOutputs::from_game_stats(all_game_stats));
    }
}

struct EnvData {
    state: State,
    player_data: PlayerData,
    stats: GameStats,
    params: VariableParams,
}

impl EnvData {
    fn new() -> Self {
        let state = State {
            done: true,
            ..Default::default()
        };
        let params = VariableParams::default();
        Self::from_state_params(state, params)
    }

    fn from_state_params(state: State, params: VariableParams) -> Self {
        let stats = GameStats::new();
        let known_params = KnownVariableParams::from(params.clone());
        let player_data =
            PlayerData::from_player_count_known_params(P, known_params);
        Self {
            state,
            player_data,
            stats,
            params,
        }
    }

    #[inline(always)]
    fn terminate(&mut self) {
        self.state.done = true;
    }

    #[inline(always)]
    fn done(&self) -> bool {
        self.state.done
    }

    #[inline(always)]
    fn is_new_match(&self) -> bool {
        self.state.match_steps == 0
    }

    #[inline(always)]
    fn is_new_game(&self) -> bool {
        self.state.total_steps == 0
    }
}

struct ParallelEnvOutputs {
    spatial_obs: Array5<f32>,
    global_obs: Array3<f32>,
    action_mask: Array4<bool>,
    sap_mask: Array5<bool>,
    unit_indices: Array4<isize>,
    unit_energies: Array3<f32>,
    units_mask: Array3<bool>,
    reward: Array2<f32>,
    done: Array1<bool>,
    stats: Option<ParallelEnvGameStatsOutputs>,
}

impl ParallelEnvOutputs {
    fn new(n_envs: usize) -> Self {
        let spatial_obs = Array5::zeros((
            n_envs,
            P,
            get_spatial_feature_count(),
            FIXED_PARAMS.map_width,
            FIXED_PARAMS.map_height,
        ));
        let global_obs = Array3::zeros((n_envs, P, get_global_feature_count()));
        let action_mask =
            Array4::default((n_envs, P, FIXED_PARAMS.max_units, Action::COUNT));
        let sap_mask = Array5::default((
            n_envs,
            P,
            FIXED_PARAMS.max_units,
            FIXED_PARAMS.map_width,
            FIXED_PARAMS.map_height,
        ));
        let unit_indices =
            Array4::zeros((n_envs, P, FIXED_PARAMS.max_units, 2));
        let unit_energies = Array3::zeros((n_envs, P, FIXED_PARAMS.max_units));
        let units_mask = Array3::default((n_envs, P, FIXED_PARAMS.max_units));
        let reward = Array2::zeros((n_envs, P));
        let done = Array1::default(n_envs);
        Self {
            spatial_obs,
            global_obs,
            action_mask,
            sap_mask,
            unit_indices,
            unit_energies,
            units_mask,
            reward,
            done,
            stats: None,
        }
    }

    fn into_pyarray_bound(self, py: Python) -> PyEnvOutputs {
        let obs = (
            self.spatial_obs.into_pyarray_bound(py),
            self.global_obs.into_pyarray_bound(py),
        );
        let action_info = (
            self.action_mask.into_pyarray_bound(py),
            self.sap_mask.into_pyarray_bound(py),
            self.unit_indices.into_pyarray_bound(py),
            self.unit_energies.into_pyarray_bound(py),
            self.units_mask.into_pyarray_bound(py),
        );
        let stats_dict = self.stats.map(|s| s.into_py_bound_dicts(py));
        (
            obs,
            action_info,
            self.reward.into_pyarray_bound(py),
            self.done.into_pyarray_bound(py),
            stats_dict,
        )
    }

    fn iter_env_slices_mut(&mut self) -> impl Iterator<Item = SingleEnvView> {
        izip_eq!(
            self.spatial_obs.outer_iter_mut(),
            self.global_obs.outer_iter_mut(),
            self.action_mask.outer_iter_mut(),
            self.sap_mask.outer_iter_mut(),
            self.unit_indices.outer_iter_mut(),
            self.unit_energies.outer_iter_mut(),
            self.units_mask.outer_iter_mut(),
            self.reward.outer_iter_mut(),
            self.done.iter_mut(),
        )
        .map(
            |(
                spatial_obs,
                global_obs,
                action_mask,
                sap_mask,
                unit_indices,
                unit_energies,
                units_mask,
                reward,
                done,
            )| {
                let obs_arrays = ObsArraysView {
                    spatial_obs,
                    global_obs,
                };
                let action_info_arrays = ActionInfoArraysView {
                    action_mask,
                    sap_mask,
                    unit_indices,
                    unit_energies,
                    units_mask,
                };
                SingleEnvView {
                    obs_arrays,
                    action_info_arrays,
                    reward,
                    done,
                }
            },
        )
    }
}

struct ParallelEnvGameStatsOutputs {
    terminal_points_scored: Array1<f32>,
    mean_terminal_points_scored: f32,
    normalized_terminal_points_scored: Array1<f32>,
    mean_normalized_terminal_points_scored: f32,
    energy_field_deltas: Array1<f32>,
    normalized_energy_field_deltas: Array1<f32>,
    nebula_energy_deltas: Array1<f32>,
    energy_void_field_deltas: Array1<f32>,

    mean_units_lost_to_energy: f32,
    mean_units_lost_to_collision: f32,
    noop_frequency: f32,
    move_frequency: f32,
    sap_frequency: f32,

    /// Could be >= 1.0 if most saps hit more than 1 unit
    sap_direct_hits_frequency: f32,
    /// Could be >= 1.0 if most saps hit more than 1 unit
    sap_adjacent_hits_frequency: f32,
    sap_miss_frequency: f32,
}

impl ParallelEnvGameStatsOutputs {
    fn from_game_stats(game_stats: Vec<&GameStats>) -> Self {
        let terminal_points_scored = Array1::from_vec(
            game_stats
                .iter()
                .flat_map(|gs| gs.terminal_points_scored.iter())
                .map(|&p| p as f32)
                .collect(),
        );
        let mean_terminal_points_scored =
            terminal_points_scored.mean().unwrap();
        let normalized_terminal_points_scored = Array1::from_vec(
            game_stats
                .iter()
                .flat_map(|gs| gs.normalized_terminal_points_scored.iter())
                .copied()
                .collect(),
        );
        let mean_normalized_terminal_points_scored =
            normalized_terminal_points_scored.mean().unwrap();
        let energy_field_deltas = Array1::from_vec(
            game_stats
                .iter()
                .flat_map(|gs| gs.energy_field_deltas.iter())
                .map(|&p| p as f32)
                .collect(),
        );
        let normalized_energy_field_deltas = Array1::from_vec(
            game_stats
                .iter()
                .flat_map(|gs| gs.normalized_energy_field_deltas.iter())
                .copied()
                .collect(),
        );
        let nebula_energy_deltas = Array1::from_vec(
            game_stats
                .iter()
                .flat_map(|gs| gs.nebula_energy_deltas.iter())
                .map(|&p| p as f32)
                .collect(),
        );
        let energy_void_field_deltas = Array1::from_vec(
            game_stats
                .iter()
                .flat_map(|gs| gs.energy_void_field_deltas.iter())
                .map(|&p| p as f32)
                .collect(),
        );

        let n_games = game_stats.len() as f32;
        let mean_units_lost_to_energy = game_stats
            .iter()
            .map(|gs| gs.units_lost_to_energy as f32)
            .sum::<f32>()
            / n_games;
        let mean_units_lost_to_collision = game_stats
            .iter()
            .map(|gs| gs.units_lost_to_collision as f32)
            .sum::<f32>()
            / n_games;

        let noop_frequency =
            game_stats.iter().map(|gs| gs.noop_frequency()).sum::<f32>()
                / n_games;
        let move_frequency =
            game_stats.iter().map(|gs| gs.move_frequency()).sum::<f32>()
                / n_games;
        let sap_frequency =
            game_stats.iter().map(|gs| gs.sap_frequency()).sum::<f32>()
                / n_games;
        let sap_direct_hits_frequency = game_stats
            .iter()
            .map(|gs| gs.sap_direct_hits_frequency())
            .sum::<f32>()
            / n_games;
        let sap_adjacent_hits_frequency = game_stats
            .iter()
            .map(|gs| gs.sap_adjacent_hits_frequency())
            .sum::<f32>()
            / n_games;
        let sap_miss_frequency = game_stats
            .iter()
            .map(|gs| gs.sap_miss_frequency())
            .sum::<f32>()
            / n_games;

        Self {
            terminal_points_scored,
            mean_terminal_points_scored,
            normalized_terminal_points_scored,
            mean_normalized_terminal_points_scored,
            energy_field_deltas,
            normalized_energy_field_deltas,
            nebula_energy_deltas,
            energy_void_field_deltas,
            mean_units_lost_to_energy,
            mean_units_lost_to_collision,
            noop_frequency,
            move_frequency,
            sap_frequency,
            sap_direct_hits_frequency,
            sap_adjacent_hits_frequency,
            sap_miss_frequency,
        }
    }

    fn into_py_bound_dicts(self, py: Python) -> PyStatsOutputs {
        let scalar_values = HashMap::from([
            (
                "mean_terminal_points_scored".to_string(),
                self.mean_terminal_points_scored,
            ),
            (
                "mean_normalized_terminal_points_scored".to_string(),
                self.mean_normalized_terminal_points_scored,
            ),
            (
                "mean_units_lost_to_energy".to_string(),
                self.mean_units_lost_to_energy,
            ),
            (
                "mean_units_lost_to_collision".to_string(),
                self.mean_units_lost_to_collision,
            ),
            ("noop_frequency".to_string(), self.noop_frequency),
            ("move_frequency".to_string(), self.move_frequency),
            ("sap_frequency".to_string(), self.sap_frequency),
            (
                "sap_direct_hits_frequency".to_string(),
                self.sap_direct_hits_frequency,
            ),
            (
                "sap_adjacent_hits_frequency".to_string(),
                self.sap_adjacent_hits_frequency,
            ),
            (
                "sap_miss_frequency".to_string(),
                self.sap_miss_frequency,
            ),
        ]);
        let array_values = HashMap::from([
            (
                "terminal_points_scored".to_string(),
                self.terminal_points_scored.into_pyarray_bound(py),
            ),
            (
                "normalized_terminal_points_scored".to_string(),
                self.normalized_terminal_points_scored
                    .into_pyarray_bound(py),
            ),
            (
                "energy_field_deltas".to_string(),
                self.energy_field_deltas.into_pyarray_bound(py),
            ),
            (
                "normalized_energy_field_deltas".to_string(),
                self.normalized_energy_field_deltas.into_pyarray_bound(py),
            ),
            (
                "nebula_energy_deltas".to_string(),
                self.nebula_energy_deltas.into_pyarray_bound(py),
            ),
            (
                "energy_void_field_deltas".to_string(),
                self.energy_void_field_deltas.into_pyarray_bound(py),
            ),
        ]);
        (scalar_values, array_values)
    }
}
