mod cached_energy_fields;
mod energy_field;
mod hidden_parameter;
mod masked_possibilities;
mod relic_node;
mod space_obstacle;

use crate::rules_engine::action::Action;
use crate::rules_engine::param_ranges::ParamRanges;
use crate::rules_engine::params::{FixedParams, KnownVariableParams};
use crate::rules_engine::state::{Observation, Pos};
use energy_field::EnergyFieldMemory;
use hidden_parameter::HiddenParameterMemory;
use itertools::Itertools;
use numpy::ndarray::Array2;
use relic_node::RelicNodeMemory;
use space_obstacle::SpaceObstacleMemory;

pub struct Memory {
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

    pub fn iter_unmasked_nebula_tile_vision_reduction_options(
        &self,
    ) -> impl Iterator<Item = &i32> {
        self.hidden_parameter
            .nebula_tile_vision_reduction
            .iter_unmasked_options()
    }

    pub fn iter_unmasked_nebula_tile_energy_reduction_options(
        &self,
    ) -> impl Iterator<Item = &i32> {
        self.hidden_parameter
            .nebula_tile_energy_reduction
            .iter_unmasked_options()
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
        &self.relic_node.explored_nodes
    }

    pub fn get_relic_known_to_have_points_map(&self) -> &Array2<bool> {
        &self.relic_node.known_to_have_points
    }

    pub fn get_relic_estimated_unexplored_points_map(&self) -> &Array2<f32> {
        &self.relic_node.estimated_unexplored_points
    }

    pub fn get_relic_explored_points_map(&self) -> &Array2<bool> {
        &self.relic_node.explored_points
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
        let full_weights = self
            .space_obstacle
            .nebula_tile_drift_speed
            .get_weighted_possibilities();

        let (first, second) = full_weights.split_at(full_weights.len() / 2);
        first
            .iter()
            .zip_eq(second.iter().rev())
            .map(|(a, b)| a + b)
            .collect()
    }

    pub fn get_nebula_tile_drift_direction_weights(&self) -> [f32; 2] {
        let negative_drift_possible = self
            .space_obstacle
            .nebula_tile_drift_speed
            .iter_unmasked_options()
            .any(|&s| s < 0.0);
        let positive_drift_possible = self
            .space_obstacle
            .nebula_tile_drift_speed
            .iter_unmasked_options()
            .any(|&s| s > 0.0);
        match (negative_drift_possible, positive_drift_possible) {
            (true, true) => [0.5, 0.5],
            (true, false) => [1.0, 0.0],
            (false, true) => [0.0, 1.0],
            (false, false) => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::feature_engineering::replay::{load_replay, run_replay};
    use crate::izip_eq;
    use crate::rules_engine::param_ranges::PARAM_RANGES;
    use crate::rules_engine::params::{FIXED_PARAMS, P};
    use itertools::Itertools;
    use numpy::ndarray::{ArrayView2, Zip};
    use rstest::rstest;
    use std::path::PathBuf;

    #[test]
    fn test_nebula_tile_drift_speed_weights_assumptions() {
        let drift_speeds = PARAM_RANGES.nebula_tile_drift_speed.clone();
        assert_eq!(drift_speeds.len() % 2, 0);
        let (first, second) = drift_speeds.split_at(drift_speeds.len() / 2);
        let first = first.iter().map(|s| s.abs()).collect_vec();
        let second = second.iter().rev().copied().collect_vec();
        assert_eq!(first, second);
    }

    #[rstest]
    #[case(vec![true; 6], vec![1. / 3.; 3])]
    #[case(vec![false, false, false, true, true, true], vec![1. / 3.; 3])]
    #[case(
        vec![true, false, false, false, false, false],
        vec![1., 0., 0.],
    )]
    #[case(
        vec![true, false, false, false, false, true],
        vec![1., 0., 0.],
    )]
    #[case(
        vec![true, false, false, false, true, false],
        vec![0.5, 0.5, 0.],
    )]
    #[case(
        vec![true, false, false, false, true, true],
        vec![2. / 3., 1. / 3., 0.],
    )]
    fn test_nebula_tile_drift_speed_weights(
        #[case] mask: Vec<bool>,
        #[case] expected_weights: Vec<f32>,
    ) {
        let mut mem = Memory::new(&PARAM_RANGES, FIXED_PARAMS.map_size);
        mem.space_obstacle.nebula_tile_drift_speed.mask = mask;
        assert_eq!(mem.get_nebula_tile_drift_speed_weights(), expected_weights);
    }

    #[test]
    fn test_nebula_tile_drift_direction_assumptions() {
        assert!(!PARAM_RANGES.nebula_tile_drift_speed.contains(&0.));
    }

    fn is_symmetrical<T>(arr: ArrayView2<T>) -> bool
    where
        T: Copy + PartialEq,
    {
        let (width, height) = arr.dim();
        let map_size = [width, height];
        for (pos, &v) in arr.indexed_iter().map(|(xy, v)| (Pos::from(xy), v)) {
            if arr[pos.reflect(map_size).as_index()] != v {
                return false;
            }
        }
        true
    }

    #[rstest]
    #[ignore = "slow"]
    fn test_energy_field_memory(
        #[files("src/feature_engineering/test_data/*.json")] path: PathBuf,
    ) {
        let full_replay = load_replay(path);
        let variable_params = &full_replay.params.variable;
        let known_params = KnownVariableParams::from(variable_params.clone());

        let mut memories = [
            Memory::new(&PARAM_RANGES, FIXED_PARAMS.map_size),
            Memory::new(&PARAM_RANGES, FIXED_PARAMS.map_size),
        ];
        let mut known_pcts = Vec::new();
        let mut incorrect_speed_count = 0.;
        let mut uncached_count = 0;
        for (state, actions, obs, _next_state) in run_replay(&full_replay) {
            let mut known_count = 0;
            let mut unknown_count = 0;
            for (mem, obs, last_actions) in
                izip_eq!(memories.iter_mut(), obs, actions)
            {
                mem.update(&obs, &last_actions, &FIXED_PARAMS, &known_params);
                for ((loc, e_mem), e_actual) in mem
                    .energy_field
                    .energy_field
                    .indexed_iter()
                    .zip_eq(state.energy_field.iter().copied())
                {
                    if let &Some(e) = e_mem {
                        known_count += 1;
                        assert_eq!(e, e_actual);
                    } else {
                        unknown_count += 1;
                        assert!(!obs.sensor_mask[loc]);
                    }
                }
                if !mem
                    .energy_field
                    .energy_node_drift_speed
                    .iter_unmasked_options()
                    .any(|&speed| {
                        speed == variable_params.energy_node_drift_speed
                    })
                {
                    incorrect_speed_count += 1.;
                }
                assert!(is_symmetrical(mem.energy_field.energy_field.view()));
                if mem.energy_field.energy_field_uncached() {
                    uncached_count += 1;
                }
            }
            known_pcts.push(
                known_count as f32 / (known_count + unknown_count) as f32,
            );
        }

        let mean_known_pct =
            known_pcts.iter().sum::<f32>() / known_pcts.len() as f32;
        let max_known_pct = *known_pcts
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        let uncached_pct = uncached_count as f32
            / P as f32
            / FIXED_PARAMS.get_max_steps_in_game() as f32;
        assert!(mean_known_pct >= 0.99);
        assert_eq!(max_known_pct, 1.);
        assert!(uncached_pct <= 1. / FIXED_PARAMS.max_steps_in_match as f32);
        assert!(
            incorrect_speed_count
                / (FIXED_PARAMS.get_max_steps_in_game() * P as u32) as f32
                <= 0.1
        );
        for mem in memories.iter() {
            assert!(
                mem.energy_field.energy_node_drift_speed.solved(),
                "{:?}",
                mem.energy_field.energy_node_drift_speed
            );
        }
    }

    #[rstest]
    #[ignore = "slow"]
    fn test_hidden_parameter_memory(
        #[files("src/feature_engineering/test_data/*.json")] path: PathBuf,
    ) {
        let full_replay = load_replay(path);
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
                assert!(mem
                    .hidden_parameter
                    .unit_energy_void_factor
                    .iter_unmasked_options()
                    .any(|&sd| sd == variable_params.unit_energy_void_factor));
            }
        }

        for mem in memories.iter() {
            if full_replay.params.variable.unit_sensor_range <= 1
                && full_replay.params.variable.nebula_tile_vision_reduction >= 3
            {
                assert!(mem
                    .hidden_parameter
                    .nebula_tile_vision_reduction
                    .iter_unmasked_options()
                    .all(|&vr| vr >= 3));
            } else {
                assert!(
                    mem.hidden_parameter.nebula_tile_vision_reduction.solved(),
                    "{:?}",
                    mem.hidden_parameter.nebula_tile_vision_reduction
                );
            }

            assert!(
                mem.hidden_parameter.nebula_tile_energy_reduction.solved(),
                "{:?}",
                mem.hidden_parameter.nebula_tile_energy_reduction
            );
            assert!(
                mem.hidden_parameter.unit_sap_dropoff_factor.solved(),
                "{:?}",
                mem.hidden_parameter.unit_sap_dropoff_factor
            );
            assert!(
                mem.hidden_parameter.unit_energy_void_factor.solved(),
                "{:?}",
                mem.hidden_parameter.unit_energy_void_factor
            );
        }
    }

    #[rstest]
    #[ignore = "slow"]
    fn test_relic_node_memory(
        #[files("src/feature_engineering/test_data/*.json")] path: PathBuf,
    ) {
        let full_replay = load_replay(path);
        let variable_params = &full_replay.params.variable;
        let known_params = KnownVariableParams::from(variable_params.clone());

        let mut memories = [
            Memory::new(&PARAM_RANGES, FIXED_PARAMS.map_size),
            Memory::new(&PARAM_RANGES, FIXED_PARAMS.map_size),
        ];
        for (_state, actions, obs, next_state) in run_replay(&full_replay) {
            for (mem, obs, last_actions) in
                izip_eq!(memories.iter_mut(), obs, actions)
            {
                mem.update(&obs, &last_actions, &FIXED_PARAMS, &known_params);
                Zip::from(&next_state.relic_node_points_map)
                    .and(&mem.relic_node.known_to_have_points)
                    .and(&mem.relic_node.explored_points)
                    .for_each(
                        |&actual_point, &known_to_have_point, &explored| {
                            if explored {
                                assert_eq!(known_to_have_point, actual_point,);
                            }
                        },
                    );

                for (loc, &explored) in
                    mem.relic_node.explored_nodes.indexed_iter()
                {
                    if explored {
                        let pos = loc.into();
                        assert_eq!(
                            next_state.relic_node_locations.contains(&pos),
                            mem.relic_node.relic_nodes.contains(&pos),
                        );
                    } else if obs.match_steps
                        >= FIXED_PARAMS.max_steps_in_match / 2
                    {
                        assert!(!obs.sensor_mask[loc]);
                    }
                }

                for rn in &mem.relic_node.relic_nodes {
                    assert!(mem
                        .relic_node
                        .relic_nodes
                        .contains(&rn.reflect(FIXED_PARAMS.map_size)));
                }
                assert!(is_symmetrical(mem.relic_node.explored_nodes.view()));
                assert!(is_symmetrical(
                    mem.relic_node.known_to_have_points.view()
                ));
                assert!(is_symmetrical(
                    mem.relic_node.estimated_unexplored_points.view()
                ));
                assert!(is_symmetrical(mem.relic_node.explored_points.view()));
            }
        }
        for mem in memories.iter() {
            let explored_pct = mem
                .relic_node
                .explored_nodes
                .mapv(|ex| if ex { 1.0 } else { 0.0 })
                .mean()
                .unwrap();
            match variable_params.unit_sensor_range {
                1 => assert!(explored_pct >= 0.5),
                2 => assert!(explored_pct >= 0.6),
                3 => assert!(explored_pct >= 0.75),
                4 => assert!(explored_pct >= 0.85),
                n => panic!("Unrecognized unit_sensor_range {}", n),
            }
            if mem.relic_node.get_all_nodes_registered() {
                assert_eq!(explored_pct, 1.0);
            }

            let point_explored_pct = mem
                .relic_node
                .explored_points
                .mapv(|ex| if ex { 1.0 } else { 0.0 })
                .mean()
                .unwrap();
            if full_replay.get_relic_nodes().len()
                == FIXED_PARAMS.max_relic_nodes
            {
                assert!(mem.relic_node.get_all_nodes_registered());
                assert!(point_explored_pct >= 0.98);
            } else {
                // TODO: Re-add this check once there's a strong agent
                //  assert!(point_explored_pct >= 0.7);
            }
        }
    }

    #[rstest]
    #[ignore = "slow"]
    fn test_space_obstacle_memory(
        #[files("src/feature_engineering/test_data/*.json")] path: PathBuf,
    ) {
        let full_replay = load_replay(path);
        let variable_params = &full_replay.params.variable;
        let known_params = KnownVariableParams::from(variable_params.clone());

        let mut memories = [
            Memory::new(&PARAM_RANGES, FIXED_PARAMS.map_size),
            Memory::new(&PARAM_RANGES, FIXED_PARAMS.map_size),
        ];
        let mut nebula_tile_drift_speed_solve_step = [u32::MAX, u32::MAX];
        for (_state, actions, obs, next_state) in run_replay(&full_replay) {
            for (mem, obs, last_actions, solve_step) in izip_eq!(
                memories.iter_mut(),
                obs,
                actions,
                nebula_tile_drift_speed_solve_step.iter_mut()
            ) {
                mem.update(&obs, &last_actions, &FIXED_PARAMS, &known_params);
                for (pos, explored) in mem
                    .space_obstacle
                    .explored_tiles
                    .indexed_iter()
                    .map(|(xy, explored)| (Pos::from(xy), *explored))
                {
                    if explored {
                        assert_eq!(
                            mem.space_obstacle.known_asteroids[pos.as_index()],
                            next_state.asteroids.contains(&pos)
                        );
                        assert_eq!(
                            mem.space_obstacle.known_nebulae[pos.as_index()],
                            next_state.nebulae.contains(&pos)
                        );
                    } else {
                        assert!(!obs.sensor_mask[pos.as_index()]);
                    }
                }
                assert!(mem
                    .space_obstacle
                    .nebula_tile_drift_speed
                    .iter_unmasked_options()
                    .any(|&speed| speed
                        == variable_params.nebula_tile_drift_speed));

                assert!(is_symmetrical(
                    mem.space_obstacle.known_asteroids.view()
                ));
                assert!(is_symmetrical(
                    mem.space_obstacle.known_nebulae.view()
                ));
                assert!(is_symmetrical(
                    mem.space_obstacle.explored_tiles.view()
                ));
                if mem.space_obstacle.nebula_tile_drift_speed.solved() {
                    *solve_step = (*solve_step).min(obs.total_steps);
                }
            }
        }
        for mem in memories.iter() {
            let explored_pct = mem
                .space_obstacle
                .explored_tiles
                .mapv(|ex| if ex { 1.0 } else { 0.0 })
                .mean()
                .unwrap();
            assert!(explored_pct >= 0.98);
            assert!(
                mem.space_obstacle.nebula_tile_drift_speed.solved(),
                "{:?}",
                mem.space_obstacle.nebula_tile_drift_speed
            );
        }
        // Nebula / asteroid drift should be solved by the first time the tiles move
        let expected_solved_by = (1.0
            / full_replay.params.variable.nebula_tile_drift_speed)
            .abs()
            .ceil() as u32
            + 1;
        assert!(
            nebula_tile_drift_speed_solve_step
                .into_iter()
                .all(|step| step <= expected_solved_by),
            "{:?}",
            nebula_tile_drift_speed_solve_step
        );
    }
}
