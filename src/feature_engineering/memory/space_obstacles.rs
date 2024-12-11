use crate::rules_engine::params::FIXED_PARAMS;
use crate::rules_engine::state::{Observation, Pos};
use itertools::Itertools;
use numpy::ndarray::{Array2, Zip};

/// Tracks everything known by a player about space obstacle (asteroid and nebulae) locations

#[derive(Debug)]
pub struct SpaceObstacleMemory {
    pub known_asteroids: Vec<Pos>,
    pub known_nebulae: Vec<Pos>,
    pub asteroid_map: Array2<bool>,
    pub nebula_map: Array2<f32>,
    pub unknown_map: Array2<bool>,
    all_nodes_registered: bool,
    map_size: [usize; 2],
}

impl SpaceObstacleMemory {
    pub fn new(map_size: [usize; 2]) -> Self {
        SpaceObstacleMemory{
            known_asteroids: Vec::new(),
            known_nebulae: Vec::new(),
            asteroid_map: Array2::default(map_size),
            nebula_map: Array2::default(map_size),
            unknown_map: Array2::default(map_size),
            all_nodes_registered: false,
            map_size,
        }
    }
}

fn update_explored_nodes(&mut self, obs: &Observation) {
    if self.all_nodes_registered {
        return;
    }
    // TODO Can I do this for both asteroids and nebula in one iteration?
    for pos in obs.asteroids.iter() {
        if self.known_asteroids.contains(pos) {
            continue;
        }
        self.known_asteroids.push(*pos);
        self.known_asteroids.push(pos.reflect(self.map_size));
    }

    for pos in obs.nebulae.iter() {
        if self.known_nebulae.contains(pos) {
            continue;
        }
        self.known_nebulae.push(*pos);
        self.known_nebulae.push(pos.reflect(self.map_size));
    }

    // TODO - is there an analogous check for ast/neb?
    // if self.check_if_all_relic_nodes_found() {
    //     self.register_all_relic_nodes_found()
    // }
}