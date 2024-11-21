use crate::rules_engine::action::Action;
use crate::rules_engine::params::{FixedParams, VariableParams};
use crate::rules_engine::state::from_array::{
    get_asteroids, get_energy_nodes, get_nebulae,
};
use crate::rules_engine::state::{EnergyNode, Observation, Pos, State, Unit};
use itertools::Itertools;
use numpy::ndarray::{arr2, arr3, Array1, Array2, Array3, Zip};
use serde::Deserialize;

#[derive(Deserialize)]
pub struct FullReplay {
    pub params: CombinedParams,
    actions: Vec<ReplayPlayerActions>,
    observations: Vec<ReplayObservation>,
    energy_node_fns: Vec<[f32; 4]>,
    player_observations: [Vec<ReplayPlayerObservation>; 2],
}

impl FullReplay {
    fn get_map_size(&self) -> [usize; 2] {
        self.params.fixed.map_size
    }

    fn get_relic_config_size(&self) -> usize {
        self.params.fixed.relic_config_size
    }

    pub fn get_states(&self) -> Vec<State> {
        let mut result = Vec::with_capacity(self.observations.len());
        for obs in &self.observations {
            let game_over = obs.team_wins.iter().sum::<u32>()
                >= self.params.fixed.match_count_per_episode;
            let mut state = State {
                units: obs.get_units(),
                asteroids: obs.get_asteroids(self.get_map_size()),
                nebulae: obs.get_nebulae(self.get_map_size()),
                energy_nodes: self.get_energy_nodes(&obs.energy_nodes),
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
            result.push(state);
        }
        result
    }

    pub fn get_player_observations(&self) -> Vec<[Observation; 2]> {
        let [p1_obs, p2_obs] = &self.player_observations;
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
                    (2, width, height),
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

    pub fn get_actions(&self) -> Vec<[Vec<Action>; 2]> {
        self.actions
            .iter()
            .map(|acts| acts.clone().into_actions())
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
    fn into_actions(self) -> [Vec<Action>; 2] {
        [
            self.player_0.into_iter().map(Action::from).collect(),
            self.player_1.into_iter().map(Action::from).collect(),
        ]
    }
}

#[derive(Deserialize)]
struct ReplayObservation {
    units: ReplayUnits,
    units_mask: [Vec<bool>; 2],
    energy_nodes: Vec<[i16; 2]>,
    relic_nodes: Vec<[isize; 2]>,
    relic_node_configs: Vec<[[bool; 5]; 5]>,
    map_features: ReplayMapFeatures,
    vision_power_map: [Vec<Vec<i32>>; 2],
    team_points: [u32; 2],
    team_wins: [u32; 2],
    steps: u32,
    match_steps: u32,
}

impl ReplayObservation {
    fn get_units(&self) -> [Vec<Unit>; 2] {
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
    position: [Vec<[usize; 2]>; 2],
    energy: [Vec<[i32; 1]>; 2],
}

#[derive(Deserialize)]
struct ReplayMapFeatures {
    energy: Vec<Vec<i32>>,
    tile_type: Vec<Vec<i32>>,
}

#[derive(Deserialize)]
struct ReplayPlayerObservation {
    units: ReplayPlayerObservationUnits,
    units_mask: [Vec<bool>; 2],
    sensor_mask: Vec<Vec<bool>>,
    map_features: ReplayMapFeatures,
    relic_nodes: Vec<[isize; 2]>,
    relic_nodes_mask: Vec<bool>,
    team_points: [u32; 2],
    team_wins: [u32; 2],
    steps: u32,
    match_steps: u32,
}

impl ReplayPlayerObservation {
    fn get_observation(
        &self,
        team_id: usize,
        map_size: [usize; 2],
    ) -> Observation {
        let units = self.get_units();
        let sensor_mask = Array2::from_shape_vec(
            map_size,
            self.sensor_mask.iter().flatten().copied().collect(),
        )
        .unwrap();
        let energy_field = Array2::from_shape_vec(
            map_size,
            self.map_features.energy.iter().flatten().copied().collect(),
        )
        .unwrap();
        let energy_field = Zip::from(&energy_field)
            .and(&sensor_mask)
            .map_collect(|&e, visible| visible.then_some(e));
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
        let asteroids = get_asteroids(tile_type.view());
        let nebulae = get_nebulae(tile_type.view());
        let relic_node_locations = self
            .relic_nodes
            .iter()
            .zip_eq(self.relic_nodes_mask.iter())
            .filter(|(_, &mask)| mask)
            .map(|(&[x, y], _)| {
                Pos::new(x.try_into().unwrap(), y.try_into().unwrap())
            })
            .collect();
        Observation {
            team_id,
            units,
            sensor_mask,
            energy_field,
            asteroids,
            nebulae,
            relic_node_locations,
            team_points: self.team_points,
            team_wins: self.team_wins,
            total_steps: self.steps,
            match_steps: self.match_steps,
        }
    }

    fn get_units(&self) -> [Vec<Unit>; 2] {
        let mut result = [Vec::new(), Vec::new()];
        for team in [0, 1] {
            result[team] = self.units.position[team]
                .iter()
                .copied()
                .zip_eq(self.units.energy[team].iter().copied())
                .enumerate()
                .zip_eq(self.units_mask[team].iter().copied())
                .filter(|&(_, alive)| alive)
                .map(|((id, (pos, e)), _)| {
                    Unit::new(pos.try_into().unwrap(), e, id)
                })
                .collect();
        }
        result
    }
}

#[derive(Deserialize)]
struct ReplayPlayerObservationUnits {
    position: [Vec<[isize; 2]>; 2],
    energy: [Vec<i32>; 2],
}
