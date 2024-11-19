use crate::feature_engineering::memory::Memory;
use crate::rules_engine::param_ranges::PARAM_RANGES;
use crate::rules_engine::params::{
    KnownVariableParams, VariableParams, FIXED_PARAMS,
};
use crate::rules_engine::state::from_array::{
    get_asteroids, get_energy_nodes, get_nebulae,
};
use crate::rules_engine::state::{Pos, State};
use itertools::Itertools;
use numpy::ndarray::{stack, Array5, ArrayView2, Axis};
use numpy::{
    IntoPyArray, PyArray5, PyReadonlyArray2, PyReadonlyArray3, PyReadonlyArray4,
};
use pyo3::prelude::*;
use rand::thread_rng;

type Obs = PyArray5<f32>;
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

    /// Resets all environments that are done, leaving active environments as-is. \
    /// values below -10_000 indicate masked value \
    /// de = envs that are done
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
        tile_type: PyReadonlyArray3<'py, i32>,
        energy_nodes: PyReadonlyArray3<'py, i16>,
        energy_node_fns: PyReadonlyArray3<'py, f32>,
        energy_nodes_mask: PyReadonlyArray2<'py, bool>,
        relic_nodes: PyReadonlyArray3<'py, i16>,
        relic_node_configs: PyReadonlyArray4<'py, bool>,
        relic_nodes_mask: PyReadonlyArray2<'py, bool>,
    ) {
        let mut rng = thread_rng();
        for (
            (
                (env_data, tile_type),
                ((energy_nodes, energy_node_fns), energy_nodes_mask),
            ),
            ((relic_nodes, relic_node_configs), relic_nodes_mask),
        ) in self
            .env_data
            .iter_mut()
            .filter(|ed| ed.state.done)
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
        }
        todo!("Test me from python side")
    }

    fn hard_reset(&mut self) {
        todo!()
    }

    fn seq_step<'py>(
        &mut self,
        py: Python<'py>,
        _actions: PyReadonlyArray3<'py, isize>,
    ) -> (Bound<'py, Obs>, Vec<Reward>, Vec<Done>) {
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
        (array_obs.into_pyarray_bound(py), reward, done)
    }

    // TODO
    // fn par_step(&mut self) -> Bound<'_, StepResult> {}
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
            Memory::new(&FIXED_PARAMS, &PARAM_RANGES),
            Memory::new(&FIXED_PARAMS, &PARAM_RANGES),
        ];
        Self {
            state,
            memories,
            params,
            known_params,
        }
    }
}
