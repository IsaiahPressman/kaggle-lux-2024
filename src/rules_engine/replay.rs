use crate::rules_engine::action::Action;
use crate::rules_engine::params::{FixedParams, VariableParams};
use crate::rules_engine::state::{EnergyNode, Pos, State, Unit};
use itertools::Itertools;
use numpy::ndarray::{arr2, Array2, Array3};
use serde::Deserialize;

const EMPTY_TILE: u8 = 0;
const NEBULA_TILE: u8 = 1;
const ASTEROID_TILE: u8 = 2;

#[derive(Deserialize)]
pub struct FullReplay {
    pub params: CombinedParams,
    actions: Vec<ReplayPlayerActions>,
    observations: Vec<ReplayObservation>,
    energy_node_fns: Vec<[f32; 4]>,
}

impl FullReplay {
    pub fn get_states(&self) -> Vec<State> {
        let mut result = Vec::with_capacity(self.observations.len());
        for obs in &self.observations {
            let game_over = obs.team_wins.iter().sum::<u32>()
                >= self.params.fixed.match_count_per_episode;
            let mut state = State {
                units: obs.get_units(),
                asteroids: obs.get_asteroids(),
                nebulae: obs.get_nebulae(),
                energy_nodes: Self::get_energy_nodes(
                    &obs.energy_nodes,
                    &self.energy_node_fns,
                ),
                relic_node_locations: obs.get_relic_node_locations(),
                relic_node_points_map: obs
                    .get_relic_node_points_map(self.params.fixed.map_size),
                team_points: obs.team_points,
                team_wins: obs.team_wins,
                total_steps: obs.steps,
                match_steps: obs.match_steps,
                done: game_over,
            };
            state.sort();
            result.push(state);
        }
        result
    }

    fn get_energy_nodes(
        locations: &[[usize; 2]],
        node_fns: &[[f32; 4]],
    ) -> Vec<EnergyNode> {
        locations
            .iter()
            .copied()
            .zip_eq(node_fns.iter().copied())
            .map(|(pos, [f_id, xyz @ ..])| {
                EnergyNode::new(pos.into(), f_id as u8, xyz)
            })
            .collect()
    }

    pub fn get_vision_power_maps(&self) -> Vec<Array3<i32>> {
        self.observations
            .iter()
            .map(|obs| {
                Array3::from_shape_vec(
                    (
                        2,
                        self.params.fixed.map_width,
                        self.params.fixed.map_height,
                    ),
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
                    self.params.fixed.map_size,
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
    energy_nodes: Vec<[usize; 2]>,
    relic_nodes: Vec<[usize; 2]>,
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
                    if alive {
                        Some(Unit::new(pos.into(), e, id))
                    } else {
                        None
                    }
                })
                .collect();
        }
        result
    }

    fn get_asteroids(&self) -> Vec<Pos> {
        self.map_features
            .tile_type
            .iter()
            .enumerate()
            .flat_map(|(x, tiles)| {
                tiles
                    .iter()
                    .copied()
                    .enumerate()
                    .map(move |(y, t)| (x, y, t))
            })
            .filter_map(|(x, y, t)| {
                Self::filter_map_tile(x, y, t, ASTEROID_TILE)
            })
            .sorted()
            .collect()
    }

    fn get_nebulae(&self) -> Vec<Pos> {
        self.map_features
            .tile_type
            .iter()
            .enumerate()
            .flat_map(|(x, tiles)| {
                tiles
                    .iter()
                    .copied()
                    .enumerate()
                    .map(move |(y, t)| (x, y, t))
            })
            .filter_map(|(x, y, t)| Self::filter_map_tile(x, y, t, NEBULA_TILE))
            .sorted()
            .collect()
    }

    fn filter_map_tile(
        x: usize,
        y: usize,
        tile_type: u8,
        target: u8,
    ) -> Option<Pos> {
        if tile_type == target {
            Some(Pos::new(x, y))
        } else if tile_type == EMPTY_TILE
            || tile_type == ASTEROID_TILE
            || tile_type == NEBULA_TILE
        {
            None
        } else {
            panic!("Unrecognized tile type: {}", tile_type)
        }
    }

    fn get_relic_node_locations(&self) -> Vec<Pos> {
        self.relic_nodes.iter().copied().map(Pos::from).collect()
    }

    fn get_relic_node_points_map(&self, map_size: [usize; 2]) -> Array2<bool> {
        let mut result = Array2::default(map_size);
        for (pos, points_mask) in self
            .relic_nodes
            .iter()
            .copied()
            .map(Pos::from)
            .zip_eq(self.relic_node_configs.iter().map(|cfg| arr2(cfg)))
        {
            for point_pos in points_mask
                .indexed_iter()
                .filter_map(|((x, y), &p)| {
                    if p {
                        Some([x as isize - 2, y as isize - 2])
                    } else {
                        None
                    }
                })
                .filter_map(|deltas| pos.maybe_translate(deltas, map_size))
            {
                result[point_pos.as_index()] = true;
            }
        }
        result
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
    tile_type: Vec<Vec<u8>>,
}
