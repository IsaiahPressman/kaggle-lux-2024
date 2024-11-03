use crate::rules_engine::params::{Params, DEFAULT_MAP_SIZE};
use crate::rules_engine::state::State;
use itertools::Itertools;
use numpy::ndarray::Array4;
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArray3};
use pyo3::prelude::*;

type Obs = PyArrayDyn<f32>;
type Reward = (isize, isize);
type Done = bool;

#[pyclass]
pub struct ParallelEnv {
    n_envs: usize,
    states: Vec<State>,
    params: Vec<Params>,
}

#[pymethods]
impl ParallelEnv {
    #[new]
    fn new(n_envs: usize) -> Self {
        let states = (0..n_envs)
            .map(|_| State::empty(DEFAULT_MAP_SIZE))
            .collect();
        let params = (0..n_envs).map(|_| Params::default()).collect();
        ParallelEnv {
            n_envs,
            states,
            params,
        }
    }

    fn soft_reset(&mut self) {
        todo!()
        // for state in self.states.iter_mut().filter(|s| s.needs_reset).zip()
    }

    // TODO
    // fn hard_reset() {
    //
    // }

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

        let mut array_obs = Array4::zeros((self.n_envs, 24, 24, 10)).into_dyn();
        array_obs += obs[0];
        (array_obs.into_pyarray_bound(py), reward, done)
    }

    // fn par_step(&mut self) -> Bound<'_, StepResult> {
    //     todo!()
    // }
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
