use crate::feature_engineering::memory::masked_possibilities::MaskedPossibilities;
use crate::rules_engine::env::estimate_vision_power_map;
use crate::rules_engine::param_ranges::ParamRanges;
use crate::rules_engine::params::KnownVariableParams;
use crate::rules_engine::state::{Observation, Pos};
use itertools::Itertools;
use numpy::ndarray::{s, Array2, Zip};

/// Tracks everything known by a player about space obstacle (asteroid and nebulae) locations

#[derive(Debug, Clone)]
pub struct SpaceObstacleMemory {
    pub known_asteroids: Array2<bool>,
    pub known_nebulae: Array2<bool>,
    pub explored_tiles: Array2<bool>,
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
            explored_tiles: Array2::default(map_size),
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
            explored_tiles: Array2::default(map_size),
            nebula_tile_drift_speed,
            map_size,
        }
    }

    pub fn update(&mut self, obs: &Observation, params: &KnownVariableParams) {
        if obs.total_steps > 0
            && self.space_obstacles_could_move(obs.total_steps - 1)
        {
            let mut fresh_memory = Self::new_empty_space_obstacles(
                self.nebula_tile_drift_speed.clone(),
                self.map_size,
            );
            fresh_memory.update_explored_obstacles(obs, params);
            self.handle_space_object_movement(
                fresh_memory,
                obs.total_steps - 1,
            );
        }

        self.update_explored_obstacles(obs, params);
    }

    fn space_obstacles_could_move(&self, step: u32) -> bool {
        self.nebula_tile_drift_speed
            .iter_unmasked_options()
            .any(|&speed| step as f32 * speed % 1.0 == 0.0)
    }

    pub fn space_obstacles_could_have_just_moved(&self, step: u32) -> bool {
        step > 0 && self.space_obstacles_could_move(step - 1)
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
            .for_each(|xy, &sensed, &should_see| {
                let pos = Pos::from(xy);
                if sensed {
                    self.explored_tiles[pos.as_index()] = true;
                    self.explored_tiles
                        [pos.reflect(self.map_size).as_index()] = true;
                } else if should_see
                    && !self.known_nebulae[pos.as_index()]
                    && !self.space_obstacles_could_move(obs.total_steps - 1)
                {
                    self.explored_tiles[pos.as_index()] = true;
                    self.explored_tiles
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
            .explored_tiles
            .indexed_iter()
            .filter(|(_, &explored)| explored)
            .map(|(xy, _)| {
                let pos = Pos::from(xy);
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

        update_nebula_tile_drift_speed(
            &mut self.nebula_tile_drift_speed,
            !not_drifting_possible,
            !negative_drift_possible,
            !positive_drift_possible,
            step,
        );
        match u8::from(not_drifting_possible)
            + u8::from(negative_drift_possible)
            + u8::from(positive_drift_possible)
        {
            0 => panic!(
                "No possible space object movement matches the observation"
            ),
            1 => {
                if negative_drift_possible {
                    self.apply_drift(negative_drift);
                } else if positive_drift_possible {
                    self.apply_drift(positive_drift);
                }
            },
            2..4 => {
                // This isn't ideal, but can happen whenever there are multiple
                // possibilities for how the map moved.
                // TODO: Maintain multiple "candidate" interpretations of the
                //  world to handle this case better?
                *self = observed;
            },
            4.. => unreachable!(),
        }
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
        if !self.explored_tiles[index] {
            None
        } else if self.known_nebulae[index] {
            Some(TileType::Nebula)
        } else if self.known_asteroids[index] {
            Some(TileType::Asteroid)
        } else {
            Some(TileType::Empty)
        }
    }

    fn apply_drift(&mut self, drift: [isize; 2]) {
        apply_drift(&mut self.known_asteroids, drift);
        apply_drift(&mut self.known_nebulae, drift);
        apply_drift(&mut self.explored_tiles, drift);
    }
}

fn should_drift(step: u32, speed: f32) -> bool {
    step as f32 * speed % 1.0 == 0.0
}

fn should_negative_drift(step: u32, speed: f32) -> bool {
    speed < 0.0 && should_drift(step, speed)
}

fn should_positive_drift(step: u32, speed: f32) -> bool {
    speed > 0.0 && should_drift(step, speed)
}

fn update_nebula_tile_drift_speed(
    nebula_tile_drift_speed: &mut MaskedPossibilities<f32>,
    not_drifting_impossible: bool,
    negative_drift_impossible: bool,
    positive_drift_impossible: bool,
    step: u32,
) {
    // not_drifting_impossible => objects have definitely drifted
    if not_drifting_impossible {
        nebula_tile_drift_speed
            .iter_unmasked_options_mut_mask()
            .for_each(|(&speed, mask)| {
                if !should_drift(step, speed) {
                    *mask = false;
                }
            })
    }
    // negative_drift_impossible => drifted positively or not moved
    if negative_drift_impossible {
        nebula_tile_drift_speed
            .iter_unmasked_options_mut_mask()
            .for_each(|(&speed, mask)| {
                if should_negative_drift(step, speed) {
                    *mask = false;
                }
            })
    }
    // positive_drift_impossible => drifted negatively or not moved
    if positive_drift_impossible {
        nebula_tile_drift_speed
            .iter_unmasked_options_mut_mask()
            .for_each(|(&speed, mask)| {
                if should_positive_drift(step, speed) {
                    *mask = false;
                }
            })
    }

    if nebula_tile_drift_speed.all_masked() {
        // TODO: For game-time build, don't panic and instead just fail to update mask
        panic!("nebula_tile_drift_speed mask is all false")
    }
}

fn apply_drift<T>(arr: &mut Array2<T>, drift: [isize; 2])
where
    T: Clone,
{
    let mut shifted = Array2::uninit(arr.dim());
    let [dx, dy] = drift;
    arr.slice(s![-dx.., -dy..])
        .assign_to(shifted.slice_mut(s![..dx, ..dy]));
    arr.slice(s![-dx.., ..-dy])
        .assign_to(shifted.slice_mut(s![..dx, dy..]));
    arr.slice(s![..-dx, -dy..])
        .assign_to(shifted.slice_mut(s![dx.., ..dy]));
    arr.slice(s![..-dx, ..-dy])
        .assign_to(shifted.slice_mut(s![dx.., dy..]));
    *arr = unsafe { shifted.assume_init() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules_engine::param_ranges::PARAM_RANGES;
    use rstest::rstest;

    #[test]
    fn test_update_total_steps_assumption() {
        assert!(PARAM_RANGES
            .nebula_tile_drift_speed
            .iter()
            .all(|&s| (1. / s) % 20. == 0.))
    }

    #[rstest]
    // Basic single impossible option cases
    #[case(false, false, false, 20, [true, true, true, true])]
    #[case(true, false, false, 20, [true, false, false, true])]
    #[case(true, false, false, 40, [true, true, true, true])]
    #[case(false, true, false, 20, [false, true, true, true])]
    #[case(false, true, false, 40, [false, false, true, true])]
    #[case(false, false, true, 20, [true, true, true, false])]
    #[case(false, false, true, 40, [true, true, false, false])]
    // Multiple impossible options
    #[case(true, true, false, 20, [false, false, false, true])]
    #[case(true, false, true, 20, [true, false, false, false])]
    #[case(false, true, true, 20, [false, true, true, false])]
    fn test_update_nebula_tile_drift_speed(
        #[case] not_drifting_impossible: bool,
        #[case] negative_drift_impossible: bool,
        #[case] positive_drift_impossible: bool,
        #[case] step: u32,
        #[case] expected_mask: [bool; 4],
    ) {
        let mut possibilities =
            MaskedPossibilities::from_options(vec![-0.05, -0.025, 0.025, 0.05]);
        update_nebula_tile_drift_speed(
            &mut possibilities,
            not_drifting_impossible,
            negative_drift_impossible,
            positive_drift_impossible,
            step,
        );
        assert_eq!(possibilities.mask, expected_mask);
    }

    #[rstest]
    #[case([true, true, true, true])]
    #[should_panic(expected = "nebula_tile_drift_speed mask is all false")]
    #[case([true, true, true, false])]
    fn test_update_nebula_tile_drift_speed_panics(#[case] mask: [bool; 4]) {
        let mut possibilities = MaskedPossibilities::new(
            vec![-0.05, -0.025, 0.025, 0.05],
            mask.to_vec(),
        );
        let not_drifting_impossible = true;
        let negative_drift_impossible = true;
        let positive_drift_impossible = false;
        let step = 20;
        update_nebula_tile_drift_speed(
            &mut possibilities,
            not_drifting_impossible,
            negative_drift_impossible,
            positive_drift_impossible,
            step,
        );
    }

    #[rstest]
    #[case(Pos::new(0, 0))]
    #[case(Pos::new(0, 8))]
    #[case(Pos::new(1, 9))]
    #[case(Pos::new(3, 5))]
    #[case(Pos::new(7, 4))]
    #[case(Pos::new(9, 9))]
    fn test_apply_drift(#[case] pos: Pos) {
        let base = 2;
        let special = 5;
        let map_size = [10, 10];
        for drift in [[-1, 1], [1, -1]] {
            let mut arr = Array2::from_elem(map_size, base);
            arr[pos.as_index()] = special;
            apply_drift(&mut arr, drift);
            let pos_drifted = pos.wrapped_translate(drift, map_size);
            for (xy, &val) in arr.indexed_iter() {
                if Pos::from(xy) == pos_drifted {
                    assert_eq!(val, special);
                } else {
                    assert_eq!(val, base);
                }
            }
        }
    }
}
