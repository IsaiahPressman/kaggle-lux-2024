use crate::rules_engine::params::P;
use crate::rules_engine::state::from_array::{get_asteroids, get_nebulae};
use crate::rules_engine::state::{Observation, Pos, Unit};
use itertools::Itertools;
use numpy::ndarray::{Array2, Zip};
use serde::Deserialize;

#[derive(Deserialize)]
pub struct LuxMapFeatures {
    pub energy: Vec<Vec<i32>>,
    pub tile_type: Vec<Vec<i32>>,
}

#[derive(Deserialize)]
pub struct LuxPlayerObservation {
    units: LuxPlayerObservationUnits,
    units_mask: [Vec<bool>; P],
    sensor_mask: Vec<Vec<bool>>,
    map_features: LuxMapFeatures,
    relic_nodes: Vec<[isize; 2]>,
    relic_nodes_mask: Vec<bool>,
    team_points: [u32; P],
    team_wins: [u32; P],
    steps: u32,
    match_steps: u32,
}

impl LuxPlayerObservation {
    pub fn get_observation(
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

    fn get_units(&self) -> [Vec<Unit>; P] {
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
pub struct LuxPlayerObservationUnits {
    position: [Vec<[isize; 2]>; 2],
    energy: [Vec<i32>; 2],
}
