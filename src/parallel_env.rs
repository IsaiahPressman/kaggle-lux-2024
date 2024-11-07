use crate::feature_engineering::memory::RelicNodeMemory;
use crate::rules_engine::params::{KnownVariableParams, Params, MAP_SIZE};
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
    states: Vec<State>,
    memories: Vec<[RelicNodeMemory; 2]>,
    params: Vec<Params>,
    known_params: Vec<KnownVariableParams>,
}

#[pymethods]
impl ParallelEnv {
    #[new]
    fn new(n_envs: usize) -> Self {
        let (states, (memories, (params, known_params))) = (0..n_envs)
            .map(|_| {
                let params = Params::default();
                let known_params = KnownVariableParams::from(params.clone());
                (
                    State::default(),
                    (
                        [
                            RelicNodeMemory::new(MAP_SIZE),
                            RelicNodeMemory::new(MAP_SIZE),
                        ],
                        (params, known_params),
                    ),
                )
            })
            .unzip();
        Self {
            n_envs,
            states,
            memories,
            params,
            known_params,
        }
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
        self.states
            .iter_mut()
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
