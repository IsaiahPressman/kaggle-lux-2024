use crate::rules_engine::action::Action;
use crate::rules_engine::env::step;
use crate::rules_engine::env::TerminationMode::FinalStep;
use crate::rules_engine::params::FIXED_PARAMS;
use crate::rules_engine::replay::FullReplay;
use crate::rules_engine::state::{Observation, State};
use itertools::Itertools;
use std::fs;
use std::path::PathBuf;

pub fn load_replay(path: PathBuf) -> FullReplay {
    let json_data = fs::read_to_string(path).unwrap();
    let full_replay: FullReplay = serde_json::from_str(&json_data).unwrap();
    assert_eq!(full_replay.params.fixed, FIXED_PARAMS);
    full_replay
}

pub fn run_replay(
    full_replay: &FullReplay,
) -> impl Iterator<Item = (State, [Vec<Action>; 2], [Observation; 2], State)> + use<'_>
{
    let mut rng = rand::thread_rng();
    full_replay
        .get_states()
        .into_iter()
        .tuple_windows()
        .zip_eq(full_replay.get_actions())
        .map(move |((state, next_state), actions)| {
            let energy_node_deltas = state.get_energy_node_deltas(&next_state);
            let (obs, _, _) = step(
                &mut state.clone(),
                &mut rng,
                &actions,
                &full_replay.params.variable,
                FinalStep,
                Some(
                    energy_node_deltas[0..energy_node_deltas.len() / 2]
                        .to_vec(),
                ),
            );
            (state, actions, obs, next_state)
        })
}
