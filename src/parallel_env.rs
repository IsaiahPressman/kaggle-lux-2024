use crate::feature_engineering::action_space::write_basic_action_space;
use crate::feature_engineering::memory::Memory;
use crate::feature_engineering::obs_space::basic_obs_space::write_obs_arrays;
use crate::rules_engine::action::Action;
use crate::rules_engine::env::get_reset_observation;
use crate::rules_engine::param_ranges::PARAM_RANGES;
use crate::rules_engine::params::{
    KnownVariableParams, VariableParams, FIXED_PARAMS,
};
use crate::rules_engine::state::from_array::{
    get_asteroids, get_energy_nodes, get_nebulae,
};
use crate::rules_engine::state::{Observation, Pos, State};
use itertools::Itertools;
use numpy::ndarray::{
    stack, Array3, Array4, Array5, ArrayView2, ArrayViewMut2, ArrayViewMut3,
    ArrayViewMut4, Axis,
};
use numpy::{
    IntoPyArray, PyArray3, PyArray4, PyArray5, PyArrayMethods,
    PyReadonlyArray2, PyReadonlyArray3, PyReadonlyArray4,
};
use pyo3::prelude::*;
use rand::thread_rng;

type ObsArrays<'py> = (
    Bound<'py, PyArray5<f32>>,
    Bound<'py, PyArray3<f32>>,
    Bound<'py, PyArray4<bool>>,
    Bound<'py, PyArray5<bool>>,
);
type Reward = (isize, isize);
type Done = bool;

#[pyclass]
pub struct ParallelEnv {
    n_envs: usize,
    env_data: Vec<EnvData>,
}

#[pymethods]
impl ParallelEnv {
    #[new]
    fn new(n_envs: usize) -> Self {
        let env_data = (0..n_envs).map(|_| EnvData::new()).collect();
        Self { n_envs, env_data }
    }

    /// Manually terminates the specified env IDs
    fn terminate_envs(&mut self, env_ids: Vec<usize>) {
        env_ids
            .into_iter()
            .for_each(|env_id| self.env_data[env_id].terminate())
    }

    /// Resets all environments that are done, leaving active environments as-is. \
    /// values below -10_000 indicate masked value \
    /// de = envs that are done
    /// - obs_arrays: (spatial_obs, global_obs, action_mask, sap_mask) output arrays
    /// - tile_type: (de, width, height)
    /// - energy_nodes: (de, max_energy_nodes, 2)
    /// - energy_node_fns: (de, max_energy_nodes, 4)
    /// - energy_nodes_mask: (de, max_energy_nodes)
    /// - relic_nodes: (de, max_relic_nodes, 2)
    /// - relic_node_configs: (de, max_relic_nodes, K, K) for a KxK relic configuration
    /// - relic_nodes_mask: (de, max_relic_nodes)
    #[allow(clippy::too_many_arguments)]
    fn soft_reset<'py>(
        &mut self,
        obs_arrays: ObsArrays<'py>,
        tile_type: PyReadonlyArray3<'py, i32>,
        energy_nodes: PyReadonlyArray3<'py, i16>,
        energy_node_fns: PyReadonlyArray3<'py, f32>,
        energy_nodes_mask: PyReadonlyArray2<'py, bool>,
        relic_nodes: PyReadonlyArray3<'py, i16>,
        relic_node_configs: PyReadonlyArray4<'py, bool>,
        relic_nodes_mask: PyReadonlyArray2<'py, bool>,
    ) {
        let mut rng = thread_rng();
        let (spatial_obs, global_obs, action_mask, sap_mask) = obs_arrays;
        for (
            (
                (
                    (
                        env_data,
                        (
                            (mut spatial_obs, mut global_obs),
                            (mut action_mask, mut sap_mask),
                        ),
                    ),
                    tile_type,
                ),
                ((energy_nodes, energy_node_fns), energy_nodes_mask),
            ),
            ((relic_nodes, relic_node_configs), relic_nodes_mask),
        ) in self
            .env_data
            .iter_mut()
            .zip_eq(
                spatial_obs
                    .readwrite()
                    .as_array_mut()
                    .outer_iter_mut()
                    .zip_eq(
                        global_obs.readwrite().as_array_mut().outer_iter_mut(),
                    )
                    .zip_eq(
                        action_mask
                            .readwrite()
                            .as_array_mut()
                            .outer_iter_mut()
                            .zip_eq(
                                sap_mask
                                    .readwrite()
                                    .as_array_mut()
                                    .outer_iter_mut(),
                            ),
                    ),
            )
            .filter(|(ed, _)| ed.state.done)
            .zip_eq(tile_type.as_array().outer_iter())
            .zip_eq(
                energy_nodes
                    .as_array()
                    .outer_iter()
                    .zip_eq(energy_node_fns.as_array().outer_iter())
                    .zip_eq(energy_nodes_mask.as_array().outer_iter()),
            )
            .zip_eq(
                relic_nodes
                    .as_array()
                    .outer_iter()
                    .zip_eq(relic_node_configs.as_array().outer_iter())
                    .zip_eq(relic_nodes_mask.as_array().outer_iter()),
            )
        {
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
            // We randomly generate params here, as they aren't needed when generating the
            // initial map state in python
            let params = PARAM_RANGES.random_params(&mut rng);
            *env_data = EnvData::from_state_params(state, params);

            let (obs, _) =
                get_reset_observation(&env_data.state, &env_data.params);
            spatial_obs.fill(0.0);
            global_obs.fill(0.0);
            action_mask.fill(false);
            sap_mask.fill(false);
            update_memories_and_write_output_arrays(
                spatial_obs,
                global_obs,
                action_mask,
                sap_mask,
                &mut env_data.memories,
                &obs,
                &[Vec::new(), Vec::new()],
                &env_data.known_params,
            );
        }
    }

    fn seq_step<'py>(
        &mut self,
        py: Python<'py>,
        _actions: PyReadonlyArray3<'py, isize>,
    ) -> (ObsArrays<'py>, Vec<Reward>, Vec<Done>) {
        let mut obs: Vec<f32> = vec![0.; self.n_envs];
        let mut reward: Vec<Reward> = vec![(0, 0); self.n_envs];
        let mut done: Vec<Done> = vec![false; self.n_envs];
        self.env_data
            .iter_mut()
            .map(|ed| &mut ed.state)
            .zip_eq(obs.iter_mut())
            .zip_eq(reward.iter_mut())
            .zip_eq(done.iter_mut())
            .map(|(((s, o), r), d)| (s, o, r, d))
            .for_each(|(s, o, r, d)| {
                s.total_steps += 1;
                get_result_inplace(s, o, r, d)
            });

        let mut array_obs = Array5::zeros((self.n_envs, 2, 10, 24, 24));
        array_obs += obs[0];
        let o2 = Array3::zeros((1, 1, 1));
        let o3 = Array4::default((1, 1, 1, 1));
        let o4 = Array5::default((1, 1, 1, 1, 1));

        let obs_arrays = (
            array_obs.into_pyarray_bound(py),
            o2.into_pyarray_bound(py),
            o3.into_pyarray_bound(py),
            o4.into_pyarray_bound(py),
        );
        (obs_arrays, reward, done)
    }

    // TODO
    // fn par_step(&mut self) -> Bound<'_, StepResult> {}
}

/// Writes the observations into the respective arrays and updates memories
/// Must be called *after* updating state and getting latest observation
#[allow(clippy::too_many_arguments)]
fn update_memories_and_write_output_arrays(
    spatial_out: ArrayViewMut4<f32>,
    global_out: ArrayViewMut2<f32>,
    action_mask: ArrayViewMut3<bool>,
    sap_mask: ArrayViewMut4<bool>,
    memories: &mut [Memory; 2],
    observations: &[Observation; 2],
    last_actions: &[Vec<Action>; 2],
    params: &KnownVariableParams,
) {
    memories
        .iter_mut()
        .zip_eq(observations.iter())
        .zip_eq(last_actions.iter())
        .for_each(|((mem, obs), last_actions)| {
            mem.update(obs, last_actions, &FIXED_PARAMS, params)
        });
    write_obs_arrays(spatial_out, global_out, observations, memories);
    write_basic_action_space(action_mask, sap_mask, observations, params);
}

fn get_result_inplace(
    s: &State,
    obs: &mut f32,
    reward: &mut Reward,
    done: &mut Done,
) {
    *obs = s.total_steps as f32;
    *reward = (1, 2);
    *done = true;
}

struct EnvData {
    state: State,
    memories: [Memory; 2],
    params: VariableParams,
    known_params: KnownVariableParams,
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
        let known_params = KnownVariableParams::from(params.clone());
        let memories = [
            Memory::new(&PARAM_RANGES, FIXED_PARAMS.map_size),
            Memory::new(&PARAM_RANGES, FIXED_PARAMS.map_size),
        ];
        Self {
            state,
            memories,
            params,
            known_params,
        }
    }

    fn terminate(&mut self) {
        self.state.done = true;
    }
}
