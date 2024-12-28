mod energy_field;
mod hidden_parameter;
mod masked_possibilities;
#[allow(dead_code)]
pub mod probabilities;
mod relic_node;
mod space_obstacle;

use crate::feature_engineering::memory::space_obstacle::SpaceObstacleMemory;
use crate::rules_engine::action::Action;
use crate::rules_engine::param_ranges::ParamRanges;
use crate::rules_engine::params::{FixedParams, KnownVariableParams};
use crate::rules_engine::state::{Observation, Pos};
use energy_field::EnergyFieldMemory;
use hidden_parameter::HiddenParameterMemory;
use numpy::ndarray::{Array2, Zip};
use relic_node::RelicNodeMemory;

pub struct Memory {
    // TODO: Test various memory modules against ground truth
    energy_field: EnergyFieldMemory,
    hidden_parameter: HiddenParameterMemory,
    relic_node: RelicNodeMemory,
    space_obstacle: SpaceObstacleMemory,
}

impl Memory {
    pub fn new(param_ranges: &ParamRanges, map_size: [usize; 2]) -> Self {
        let energy_field = EnergyFieldMemory::new(param_ranges, map_size);
        let hidden_parameters = HiddenParameterMemory::new(param_ranges);
        let relic_nodes = RelicNodeMemory::new(map_size);
        let space_obstacles = SpaceObstacleMemory::new(param_ranges, map_size);
        Self {
            energy_field,
            hidden_parameter: hidden_parameters,
            relic_node: relic_nodes,
            space_obstacle: space_obstacles,
        }
    }

    pub fn update(
        &mut self,
        obs: &Observation,
        last_actions: &[Action],
        fixed_params: &FixedParams,
        params: &KnownVariableParams,
    ) {
        self.energy_field.update(obs);
        self.space_obstacle.update(obs, params);
        let nebulae_could_have_moved = self
            .space_obstacle
            .space_obstacles_could_have_just_moved(obs.total_steps);
        self.hidden_parameter.update(
            obs,
            last_actions,
            fixed_params,
            params,
            nebulae_could_have_moved,
        );
        self.relic_node.update(obs);
    }

    pub fn get_energy_field(&self) -> &Array2<Option<i32>> {
        &self.energy_field.energy_field
    }

    pub fn get_energy_node_drift_speed_weights(&self) -> Vec<f32> {
        self.energy_field
            .energy_node_drift_speed
            .get_weighted_possibilities()
    }

    pub fn get_nebula_tile_vision_reduction_weights(&self) -> Vec<f32> {
        self.hidden_parameter
            .nebula_tile_vision_reduction
            .get_weighted_possibilities()
    }

    pub fn get_nebula_tile_energy_reduction_weights(&self) -> Vec<f32> {
        self.hidden_parameter
            .nebula_tile_energy_reduction
            .get_weighted_possibilities()
    }

    pub fn get_unit_sap_dropoff_factor_weights(&self) -> Vec<f32> {
        self.hidden_parameter
            .unit_sap_dropoff_factor
            .get_weighted_possibilities()
    }

    pub fn get_relic_nodes(&self) -> &[Pos] {
        &self.relic_node.relic_nodes
    }

    pub fn get_explored_relic_nodes_map(&self) -> &Array2<bool> {
        &self.relic_node.explored_nodes_map
    }

    pub fn get_relic_points_map(&self) -> &Array2<f32> {
        &self.relic_node.points_map
    }

    pub fn get_known_relic_points_map(&self) -> &Array2<bool> {
        &self.relic_node.known_points_map
    }

    pub fn get_known_valuable_relic_points_map(&self) -> Array2<bool> {
        Zip::from(&self.relic_node.points_map)
            .and(&self.relic_node.known_points_map)
            .map_collect(|&value, &known| known && value >= 0.99)
    }

    pub fn get_known_asteroids_map(&self) -> &Array2<bool> {
        &self.space_obstacle.known_asteroids
    }

    pub fn get_known_nebulae_map(&self) -> &Array2<bool> {
        &self.space_obstacle.known_nebulae
    }

    pub fn get_explored_tiles_map(&self) -> &Array2<bool> {
        &self.space_obstacle.explored_tiles
    }

    pub fn get_nebula_tile_drift_speed_weights(&self) -> Vec<f32> {
        self.space_obstacle
            .nebula_tile_drift_speed
            .get_weighted_possibilities()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::izip_eq;
    use crate::rules_engine::env::step;
    use crate::rules_engine::env::TerminationMode::FinalStep;
    use crate::rules_engine::param_ranges::PARAM_RANGES;
    use crate::rules_engine::params::FIXED_PARAMS;
    use crate::rules_engine::replay::FullReplay;
    use crate::rules_engine::state::State;
    use itertools::Itertools;
    use pretty_assertions::assert_eq;
    use rstest::rstest;
    use std::fs;
    use std::path::Path;

    fn load_replay(file_name: &str) -> FullReplay {
        let path = Path::new(file!())
            .parent()
            .unwrap()
            .join("test_data")
            .join(file_name);
        let json_data = fs::read_to_string(path).unwrap();
        let full_replay: FullReplay = serde_json::from_str(&json_data).unwrap();
        assert_eq!(full_replay.params.fixed, FIXED_PARAMS);
        full_replay
    }

    fn run_replay(
        full_replay: &FullReplay,
    ) -> impl Iterator<Item = (State, [Vec<Action>; 2], [Observation; 2], State)>
           + use<'_> {
        let mut rng = rand::thread_rng();
        full_replay
            .get_states()
            .into_iter()
            .tuple_windows()
            .zip_eq(full_replay.get_actions())
            .map(move |((mut state, next_state), actions)| {
                let energy_node_deltas =
                    state.get_energy_node_deltas(&next_state);
                // Currently in the replay file, each observed energy field is from the previous
                // step's computed energy field
                state.energy_field = next_state.energy_field.clone();
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

    #[rstest]
    #[ignore = "slow"]
    #[case("processed_replay_478448958.json")]
    fn test_energy_field_memory(#[case] file_name: &str) {
        let full_replay = load_replay(file_name);
        let variable_params = &full_replay.params.variable;
        let known_params = KnownVariableParams::from(variable_params.clone());

        let mut memories = [
            Memory::new(&PARAM_RANGES, FIXED_PARAMS.map_size),
            Memory::new(&PARAM_RANGES, FIXED_PARAMS.map_size),
        ];
        let mut known_pcts = Vec::new();
        for (state, actions, obs, _next_state) in run_replay(&full_replay) {
            let mut known_count = 0;
            let mut unknown_count = 0;
            for (mem, obs, last_actions) in
                izip_eq!(memories.iter_mut(), obs, actions)
            {
                mem.update(&obs, &last_actions, &FIXED_PARAMS, &known_params);
                for (e_mem, e_actual) in mem
                    .energy_field
                    .energy_field
                    .iter()
                    .copied()
                    .zip_eq(state.energy_field.iter().copied())
                {
                    if let Some(e) = e_mem {
                        known_count += 1;
                        assert_eq!(e, e_actual);
                    } else {
                        unknown_count += 1;
                    }
                }
                assert!(mem
                    .energy_field
                    .energy_node_drift_speed
                    .iter_unmasked_options()
                    .any(|&speed| speed
                        == variable_params.energy_node_drift_speed));
            }
            known_pcts.push(
                known_count as f32 / (known_count + unknown_count) as f32,
            );
        }
        assert!(
            known_pcts.iter().sum::<f32>() / known_pcts.len() as f32 >= 0.5
        );
        assert!(
            *known_pcts
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap()
                >= 0.7
        );
        for mem in memories.iter() {
            assert!(!mem.energy_field.energy_node_drift_speed.still_unsolved());
        }
    }

    #[rstest]
    #[ignore = "slow"]
    #[case("processed_replay_478448958.json")]
    fn test_hidden_parameter_memory(#[case] file_name: &str) {
        let full_replay = load_replay(file_name);
        let variable_params = &full_replay.params.variable;
        let known_params = KnownVariableParams::from(variable_params.clone());

        let mut memories = [
            Memory::new(&PARAM_RANGES, FIXED_PARAMS.map_size),
            Memory::new(&PARAM_RANGES, FIXED_PARAMS.map_size),
        ];
        for (_state, actions, obs, _next_state) in run_replay(&full_replay) {
            for (mem, obs, last_actions) in
                izip_eq!(memories.iter_mut(), obs, actions)
            {
                mem.update(&obs, &last_actions, &FIXED_PARAMS, &known_params);
                assert!(mem
                    .hidden_parameter
                    .nebula_tile_vision_reduction
                    .iter_unmasked_options()
                    .any(|&vr| vr
                        == variable_params.nebula_tile_vision_reduction));
                assert!(mem
                    .hidden_parameter
                    .nebula_tile_energy_reduction
                    .iter_unmasked_options()
                    .any(|&er| er
                        == variable_params.nebula_tile_energy_reduction));
                assert!(mem
                    .hidden_parameter
                    .unit_sap_dropoff_factor
                    .iter_unmasked_options()
                    .any(|&sd| sd == variable_params.unit_sap_dropoff_factor));
            }
        }

        for mem in memories.iter() {
            assert!(!mem
                .hidden_parameter
                .nebula_tile_vision_reduction
                .still_unsolved());
            assert!(!mem
                .hidden_parameter
                .nebula_tile_energy_reduction
                .still_unsolved());
            assert!(!mem
                .hidden_parameter
                .unit_sap_dropoff_factor
                .still_unsolved());
        }
    }

    #[rstest]
    #[ignore = "slow"]
    #[case("processed_replay_478448958.json")]
    fn test_relic_node_memory(#[case] file_name: &str) {
        let full_replay = load_replay(file_name);
        let variable_params = &full_replay.params.variable;
        let known_params = KnownVariableParams::from(variable_params.clone());

        let mut memories = [
            Memory::new(&PARAM_RANGES, FIXED_PARAMS.map_size),
            Memory::new(&PARAM_RANGES, FIXED_PARAMS.map_size),
        ];
        for (state, actions, obs, _next_state) in run_replay(&full_replay) {
            for (mem, obs, last_actions) in
                izip_eq!(memories.iter_mut(), obs, actions)
            {
                mem.update(&obs, &last_actions, &FIXED_PARAMS, &known_params);
                Zip::from(&state.relic_node_points_map)
                    .and(&mem.relic_node.points_map)
                    .and(&mem.relic_node.known_points_map)
                    .for_each(|&actual_point, &mem_point, &mem_point_known| {
                        if mem_point_known {
                            assert_eq!(
                                if actual_point { 1.0 } else { 0.0 },
                                mem_point
                            );
                        }
                    });

                for explored_pos in
                    mem.relic_node.explored_nodes_map.indexed_iter().filter_map(
                        |((x, y), explored)| explored.then_some(Pos::new(x, y)),
                    )
                {
                    assert_eq!(
                        state.relic_node_locations.contains(&explored_pos),
                        mem.relic_node.relic_nodes.contains(&explored_pos)
                    );
                }
            }
        }
        for mem in memories.iter() {
            let explored_pct = mem
                .relic_node
                .explored_nodes_map
                .mapv(|ex| if ex { 1.0 } else { 0.0 })
                .mean()
                .unwrap();
            assert!(explored_pct >= 0.9);
            if mem.relic_node.get_all_nodes_registered() {
                assert_eq!(explored_pct, 1.0);
                assert_eq!(
                    mem.relic_node.relic_nodes.len(),
                    FIXED_PARAMS.max_relic_nodes
                );
            }
        }
    }
}
