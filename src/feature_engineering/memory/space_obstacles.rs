use crate::rules_engine::env::estimate_vision_power_map;
use crate::rules_engine::params::KnownVariableParams;
use crate::rules_engine::state::{Observation, Pos};
use numpy::ndarray::{Array2, Zip};

/// Tracks everything known by a player about space obstacle (asteroid and nebulae) locations

#[derive(Debug, Clone)]
pub struct SpaceObstacleMemory {
    pub known_asteroids: Array2<bool>,
    pub known_nebulae: Array2<bool>,
    pub explored_tiles_map: Array2<bool>,
    map_size: [usize; 2],
}

#[derive(Debug, PartialEq, Eq)]
enum TileType {
    Empty,
    Nebula,
    Asteroid,
}

impl SpaceObstacleMemory {
    pub fn new(map_size: [usize; 2]) -> Self {
        SpaceObstacleMemory {
            known_asteroids: Array2::default(map_size),
            known_nebulae: Array2::default(map_size),
            explored_tiles_map: Array2::default(map_size),
            map_size,
        }
    }

    pub fn update(&mut self, obs: &Observation, params: &KnownVariableParams) {
        if (obs.total_steps - 1) % 20 == 0 {
            let mut fresh_memory = Self::new(self.map_size);
            fresh_memory.update_explored_obstacles(obs, params);
            self.handle_space_object_movement(fresh_memory);
        }

        self.update_explored_obstacles(obs, params);
    }

    fn update_explored_obstacles(
        &mut self,
        obs: &Observation,
        params: &KnownVariableParams,
    ) {
        self.update_explored_asteroids(&obs.asteroids);
        self.update_explored_nebulae(&obs.nebulae);

        let expected_vision_power_map = estimate_vision_power_map(
            obs.get_my_units(),
            self.map_size,
            params.unit_sensor_range,
        )
        .mapv(|vision| vision > 0);
        Zip::indexed(&obs.sensor_mask)
            .and(&expected_vision_power_map)
            .for_each(|(x, y), &sensed, &should_see| {
                let pos = Pos::new(x, y);
                if sensed {
                    self.explored_tiles_map[pos.as_index()] = true;
                    self.explored_tiles_map
                        [pos.reflect(self.map_size).as_index()] = true;
                } else if should_see && !self.known_nebulae[pos.as_index()] {
                    // TODO: Handle nebula movement after vision map is computed
                    self.explored_tiles_map[pos.as_index()] = true;
                    self.explored_tiles_map
                        [pos.reflect(self.map_size).as_index()] = true;
                    self.known_nebulae[pos.as_index()] = true;
                    self.known_nebulae[pos.reflect(self.map_size).as_index()] =
                        true;
                }
            });
    }

    fn update_explored_asteroids(&mut self, asteroids: &[Pos]) {
        for pos in asteroids.iter() {
            self.known_asteroids[pos.as_index()] = true;
            self.known_asteroids[pos.reflect(self.map_size).as_index()] = true;
        }
    }

    fn update_explored_nebulae(&mut self, nebulae: &[Pos]) {
        for pos in nebulae.iter() {
            self.known_nebulae[pos.as_index()] = true;
            self.known_nebulae[pos.reflect(self.map_size).as_index()] = true;
        }
    }

    fn handle_space_object_movement(&mut self, observed: SpaceObstacleMemory) {
        let mut not_moved_possible = true;
        let mut negative_drift_possible = true;
        let mut positive_drift_possible = true;
        let negative_drift = [-1, 1];
        let positive_drift = [1, -1];
        for (pos, observed_tile) in observed
            .explored_tiles_map
            .indexed_iter()
            .filter(|(_, &explored)| explored)
            .map(|((x, y), _)| {
                let pos = Pos::new(x, y);
                (pos, observed.get_tile_type_at(pos.as_index()).unwrap())
            })
        {
            if self
                .get_tile_type_at(pos.as_index())
                .is_some_and(|tt| tt != observed_tile)
            {
                not_moved_possible = false;
            }

            if self
                .get_tile_type_at(
                    pos.inverted_wrapped_translate(
                        negative_drift,
                        self.map_size,
                    )
                    .as_index(),
                )
                .is_some_and(|tt| tt != observed_tile)
            {
                negative_drift_possible = false;
            }

            if self
                .get_tile_type_at(
                    pos.inverted_wrapped_translate(
                        positive_drift,
                        self.map_size,
                    )
                    .as_index(),
                )
                .is_some_and(|tt| tt != observed_tile)
            {
                positive_drift_possible = false;
            }
        }

        // TODO: Left off here
        todo!()
    }

    fn get_tile_type_at(&self, index: [usize; 2]) -> Option<TileType> {
        if !self.explored_tiles_map[index] {
            None
        } else if self.known_nebulae[index] {
            Some(TileType::Nebula)
        } else if self.known_asteroids[index] {
            Some(TileType::Asteroid)
        } else {
            Some(TileType::Empty)
        }
    }
}
