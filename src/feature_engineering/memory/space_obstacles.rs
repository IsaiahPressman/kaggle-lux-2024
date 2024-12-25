use crate::feature_engineering::memory::masked_possibilities::MaskedPossibilities;
use crate::rules_engine::env::estimate_vision_power_map;
use crate::rules_engine::param_ranges::ParamRanges;
use crate::rules_engine::params::KnownVariableParams;
use crate::rules_engine::state::{Observation, Pos};
use itertools::Itertools;
use numpy::ndarray::{Array2, Zip};

/// Tracks everything known by a player about space obstacle (asteroid and nebulae) locations

#[derive(Debug, Clone)]
pub struct SpaceObstacleMemory {
    pub known_asteroids: Array2<bool>,
    pub known_nebulae: Array2<bool>,
    pub explored_tiles_map: Array2<bool>,
    pub nebula_tile_drift_speed: MaskedPossibilities<f32>,
    map_size: [usize; 2],
}

#[derive(Debug, PartialEq, Eq)]
enum TileType {
    Empty,
    Nebula,
    Asteroid,
}

impl SpaceObstacleMemory {
    pub fn new(param_ranges: &ParamRanges, map_size: [usize; 2]) -> Self {
        let nebula_tile_drift_speed = MaskedPossibilities::from_options(
            param_ranges
                .nebula_tile_drift_speed
                .iter()
                .copied()
                .sorted_by(|a, b| a.partial_cmp(b).unwrap())
                .dedup()
                .collect_vec(),
        );
        Self {
            known_asteroids: Array2::default(map_size),
            known_nebulae: Array2::default(map_size),
            explored_tiles_map: Array2::default(map_size),
            nebula_tile_drift_speed,
            map_size,
        }
    }

    fn new_empty_space_obstacles(
        nebula_tile_drift_speed: MaskedPossibilities<f32>,
        map_size: [usize; 2],
    ) -> Self {
        Self {
            known_asteroids: Array2::default(map_size),
            known_nebulae: Array2::default(map_size),
            explored_tiles_map: Array2::default(map_size),
            nebula_tile_drift_speed,
            map_size,
        }
    }

    pub fn update(&mut self, obs: &Observation, params: &KnownVariableParams) {
        let update_step = obs.total_steps - 1;
        if self.space_obstacles_could_move(update_step) {
            let mut fresh_memory = Self::new_empty_space_obstacles(
                self.nebula_tile_drift_speed.clone(),
                self.map_size,
            );
            fresh_memory.update_explored_obstacles(obs, params);
            self.handle_space_object_movement(fresh_memory, update_step);
        }

        self.update_explored_obstacles(obs, params);
    }

    fn space_obstacles_could_move(&self, step: u32) -> bool {
        self.nebula_tile_drift_speed
            .iter_unmasked_options()
            .any(|&speed| step as f32 * speed % 1.0 == 0.0)
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

    fn handle_space_object_movement(
        &mut self,
        observed: SpaceObstacleMemory,
        step: u32,
    ) {
        let mut not_drifting_possible = self.not_drifting_possible(step);
        let mut negative_drift_possible = self.negative_drift_possible(step);
        let mut positive_drift_possible = self.positive_drift_possible(step);
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
                not_drifting_possible = false;
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

        if !not_drifting_possible {
            self.nebula_tile_drift_speed
                .iter_unmasked_options_mut_mask()
                .for_each(|(&speed, mask)| {
                    if !should_drift(step, speed) {
                        *mask = false;
                    }
                })
        }
        if !negative_drift_possible {
            self.nebula_tile_drift_speed
                .iter_unmasked_options_mut_mask()
                .for_each(|(&speed, mask)| {
                    if should_negative_drift(step, speed) {
                        *mask = false;
                    }
                })
        }
        if !positive_drift_possible {
            self.nebula_tile_drift_speed
                .iter_unmasked_options_mut_mask()
                .for_each(|(&speed, mask)| {
                    if should_positive_drift(step, speed) {
                        *mask = false;
                    }
                })
        }

        match u8::from(not_drifting_possible)
            + u8::from(negative_drift_possible)
            + u8::from(positive_drift_possible)
        {
            0 => panic!(
                "No possible space object movement matches the observation"
            ),
            1 => {},
            2..4 => {
                // This isn't ideal, but can happen whenever there are multiple
                // possibilities for how the map moved.
                // TODO: Maintain multiple "candidate" interpretations of the
                //  world to handle this case better?
                *self = observed;
                return;
            },
            4.. => unreachable!(),
        }

        // TODO: Left off here - move space objects according to
        //  specified drift
        todo!()
    }

    fn not_drifting_possible(&self, step: u32) -> bool {
        self.nebula_tile_drift_speed
            .iter_unmasked_options()
            .any(|&speed| !should_drift(step, speed))
    }

    fn negative_drift_possible(&self, step: u32) -> bool {
        self.nebula_tile_drift_speed
            .iter_unmasked_options()
            .any(|&speed| should_negative_drift(step, speed))
    }

    fn positive_drift_possible(&self, step: u32) -> bool {
        self.nebula_tile_drift_speed
            .iter_unmasked_options()
            .any(|&speed| should_positive_drift(step, speed))
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

#[inline(always)]
fn should_drift(step: u32, speed: f32) -> bool {
    step as f32 * speed % 1.0 == 0.0
}

#[inline(always)]
fn should_negative_drift(step: u32, speed: f32) -> bool {
    speed < 0.0 && should_drift(step, speed)
}

#[inline(always)]
fn should_positive_drift(step: u32, speed: f32) -> bool {
    speed > 0.0 && should_drift(step, speed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules_engine::param_ranges::PARAM_RANGES;

    #[test]
    fn test_update_total_steps_assumption() {
        assert!(PARAM_RANGES
            .nebula_tile_drift_speed
            .iter()
            .all(|&s| (1. / s) % 20. == 0.))
    }
}
