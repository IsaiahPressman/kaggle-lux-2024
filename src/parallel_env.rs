use crate::feature_engineering::energy_field_frequencies::ENERGY_FIELD_PROBABILITIES;
use crate::feature_engineering::memory::Memory;
use crate::rules_engine::param_ranges::PARAM_RANGES;
use crate::rules_engine::params::{
    KnownVariableParams, VariableParams, FIXED_PARAMS,
};
use crate::rules_engine::state::State;
use itertools::Itertools;
use numpy::ndarray::Array5;
use numpy::{IntoPyArray, PyArray5, PyReadonlyArray3};
use pyo3::prelude::*;

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
        let env_data = (0..n_envs).map(|_| EnvData::default()).collect();
        Self { n_envs, env_data }
    }

    fn soft_reset(&mut self) {
        todo!()
        // for state in self.states.iter_mut().filter(|s| s.needs_reset).zip()
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

impl Default for EnvData {
    fn default() -> Self {
        let state = State::default();
        let params = VariableParams::default();
        let known_params = KnownVariableParams::from(params.clone());
        let memories = [
            Memory::new(
                ENERGY_FIELD_PROBABILITIES.clone(),
                &FIXED_PARAMS,
                &PARAM_RANGES,
            ),
            Memory::new(
                ENERGY_FIELD_PROBABILITIES.clone(),
                &FIXED_PARAMS,
                &PARAM_RANGES,
            ),
        ];
        Self {
            state,
            memories,
            params,
            known_params,
        }
    }
}
