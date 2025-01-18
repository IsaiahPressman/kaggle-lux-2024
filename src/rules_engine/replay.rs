use crate::rules_engine::action::Action;
use crate::rules_engine::params::{FixedParams, VariableParams, P};
use crate::rules_engine::state::from_array::{
    get_asteroids, get_energy_nodes, get_nebulae,
};
use crate::rules_engine::state::{
    EnergyNode, LuxMapFeatures, LuxPlayerObservation, Observation, Pos, State,
    Unit,
};
use itertools::Itertools;
use numpy::ndarray::{arr2, arr3, Array1, Array2, Array3};
use serde::Deserialize;

#[derive(Deserialize)]
pub struct FullReplay {
    pub params: CombinedParams,
    actions: Vec<ReplayPlayerActions>,
    observations: Vec<ReplayObservation>,
    energy_node_fns: Vec<[f32; 4]>,
    #[serde(default)]
    player_observations: Option<[Vec<LuxPlayerObservation>; P]>,
}

impl FullReplay {
    fn get_map_size(&self) -> [usize; 2] {
        self.params.fixed.map_size
    }

    fn get_relic_config_size(&self) -> usize {
        self.params.fixed.relic_config_size
    }

    pub fn get_states(&self) -> Vec<State> {
        let mut result: Vec<State> =
            Vec::with_capacity(self.observations.len());
        for obs in &self.observations {
            let game_over = obs.team_wins.iter().sum::<u32>()
                >= self.params.fixed.match_count_per_episode;
            let mut state = State {
                units: obs.get_units(),
                asteroids: obs.get_asteroids(self.get_map_size()),
                nebulae: obs.get_nebulae(self.get_map_size()),
                energy_nodes: self.get_energy_nodes(&obs.energy_nodes),
                energy_field: Array2::from_shape_vec(
                    self.get_map_size(),
                    obs.map_features.energy.iter().flatten().copied().collect(),
                )
                .unwrap(),
                team_points: obs.team_points,
                team_wins: obs.team_wins,
                total_steps: obs.steps,
                match_steps: obs.match_steps,
                done: game_over,
                ..Default::default()
            };
            state.set_relic_nodes(
                obs.relic_nodes
                    .iter()
                    .copied()
                    .map(|pos| Pos::try_from(pos).unwrap())
                    .collect(),
                arr3(&obs.relic_node_configs).view(),
                self.get_map_size(),
                self.get_relic_config_size(),
            );
            state.sort();
            // In the replay file, each observed energy field is from the previous
            // step's computed energy field
            if let Some(last_state) = result.last_mut() {
                last_state.energy_field = state.energy_field.clone();
            }
            result.push(state);
        }
        result
    }

    pub fn get_player_observations(&self) -> Vec<[Observation; P]> {
        let [p1_obs, p2_obs] = self.player_observations.clone().unwrap();
        p1_obs
            .iter()
            .zip_eq(p2_obs)
            .map(|(p1_obs, p2_obs)| {
                [
                    p1_obs.get_observation(0, self.get_map_size()),
                    p2_obs.get_observation(1, self.get_map_size()),
                ]
            })
            .collect()
    }

    pub fn get_energy_fields(&self) -> Vec<Array2<i32>> {
        self.observations
            .iter()
            .map(|obs| {
                Array2::from_shape_vec(
                    self.get_map_size(),
                    obs.map_features.energy.iter().flatten().copied().collect(),
                )
                .unwrap()
            })
            .collect()
    }

    fn get_energy_nodes(&self, locations: &[[i16; 2]]) -> Vec<EnergyNode> {
        let mask = Array1::from_elem(locations.len(), true);
        get_energy_nodes(
            arr2(locations).view(),
            arr2(&self.energy_node_fns).view(),
            mask.view(),
        )
    }

    pub fn get_vision_power_maps(&self) -> Vec<Array3<i32>> {
        let [width, height] = self.get_map_size();
        self.observations
            .iter()
            .map(|obs| {
                Array3::from_shape_vec(
                    (P, width, height),
                    obs.vision_power_map
                        .iter()
                        .flatten()
                        .flatten()
                        .copied()
                        .collect(),
                )
                .unwrap()
            })
            .collect()
    }

    pub fn get_actions(&self) -> Vec<[Vec<Action>; P]> {
        self.actions
            .iter()
            .map(|acts| acts.clone().into_actions())
            .collect()
    }

    pub fn get_relic_nodes(&self) -> Vec<Pos> {
        let obs = self.observations.first().unwrap();
        obs.relic_nodes
            .iter()
            .copied()
            .map(|pos| Pos::try_from(pos).unwrap())
            .collect()
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct CombinedParams {
    #[serde(flatten)]
    pub fixed: FixedParams,
    #[serde(flatten)]
    pub variable: VariableParams,
}

#[derive(Clone, Deserialize)]
struct ReplayPlayerActions {
    player_0: Vec<[isize; 3]>,
    player_1: Vec<[isize; 3]>,
}

impl ReplayPlayerActions {
    fn into_actions(self) -> [Vec<Action>; P] {
        [
            self.player_0.into_iter().map(Action::from).collect(),
            self.player_1.into_iter().map(Action::from).collect(),
        ]
    }
}

#[derive(Deserialize)]
struct ReplayObservation {
    units: ReplayUnits,
    units_mask: [Vec<bool>; P],
    energy_nodes: Vec<[i16; 2]>,
    relic_nodes: Vec<[isize; 2]>,
    relic_node_configs: Vec<[[bool; 5]; 5]>,
    map_features: LuxMapFeatures,
    vision_power_map: [Vec<Vec<i32>>; P],
    team_points: [u32; P],
    team_wins: [u32; P],
    steps: u32,
    match_steps: u32,
}

impl ReplayObservation {
    fn get_units(&self) -> [Vec<Unit>; P] {
        let mut result = [Vec::new(), Vec::new()];
        for team in [0, 1] {
            result[team] = self.units.position[team]
                .iter()
                .copied()
                .zip_eq(self.units.energy[team].iter().copied())
                .enumerate()
                .zip_eq(self.units_mask[team].iter().copied())
                .filter_map(|((id, (pos, [e])), alive)| {
                    alive.then_some(Unit::new(pos.into(), e, id))
                })
                .collect();
        }
        result
    }

    fn get_asteroids(&self, map_size: [usize; 2]) -> Vec<Pos> {
        let tile_type = Array2::from_shape_vec(
            map_size,
            self.map_features
                .tile_type
                .iter()
                .flatten()
                .copied()
                .collect(),
        )
        .unwrap();
        get_asteroids(tile_type.view())
    }

    fn get_nebulae(&self, map_size: [usize; 2]) -> Vec<Pos> {
        let tile_type = Array2::from_shape_vec(
            map_size,
            self.map_features
                .tile_type
                .iter()
                .flatten()
                .copied()
                .collect(),
        )
        .unwrap();
        get_nebulae(tile_type.view())
    }
}

#[derive(Deserialize)]
struct ReplayUnits {
    position: [Vec<[usize; 2]>; P],
    energy: [Vec<[i32; 1]>; P],
}
