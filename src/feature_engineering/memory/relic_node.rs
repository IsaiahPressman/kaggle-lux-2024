use crate::rules_engine::params::FIXED_PARAMS;
use crate::rules_engine::state::{Observation, Pos};
use itertools::Itertools;
use numpy::ndarray::{Array2, Zip};
use std::collections::BTreeSet;

const RELIC_WINDOW: isize = FIXED_PARAMS.relic_config_size as isize / 2;

/// Tracks everything known by a player so far about relic nodes
#[derive(Debug)]
pub struct RelicNodeMemory {
    // TODO: Overhaul this with new ruleset
    pub relic_nodes: Vec<Pos>,
    pub explored_nodes_map: Array2<bool>,
    pub known_and_explored_points_map: Array2<bool>,
    pub estimated_unexplored_points_map: Array2<f32>,
    pub explored_points_map: Array2<bool>,
    points_sum_map: Array2<f32>,
    points_count_map: Array2<u32>,
    points_last_turn: u32,
    all_nodes_registered: bool,
    map_size: [usize; 2],
}

impl RelicNodeMemory {
    pub fn new(map_size: [usize; 2]) -> Self {
        RelicNodeMemory {
            relic_nodes: Vec::with_capacity(FIXED_PARAMS.max_relic_nodes),
            explored_nodes_map: Array2::default(map_size),
            known_and_explored_points_map: Array2::default(map_size),
            estimated_unexplored_points_map: Array2::zeros(map_size),
            explored_points_map: Array2::default(map_size),
            points_sum_map: Array2::zeros(map_size),
            points_count_map: Array2::zeros(map_size),
            points_last_turn: 0,
            all_nodes_registered: false,
            map_size,
        }
    }

    fn check_if_all_relic_nodes_found(&self) -> bool {
        self.relic_nodes.len() >= FIXED_PARAMS.max_relic_nodes
            || self.explored_nodes_map.iter().all(|explored| *explored)
    }

    pub fn update(&mut self, obs: &Observation) {
        self.update_explored_nodes(obs);
        self.update_points_map(obs);
    }

    fn update_explored_nodes(&mut self, obs: &Observation) {
        if self.all_nodes_registered {
            return;
        }

        for pos in obs.relic_node_locations.iter() {
            if self.relic_nodes.contains(pos) {
                continue;
            }
            self.relic_nodes.push(*pos);
            self.relic_nodes.push(pos.reflect(self.map_size));
        }

        for pos in obs
            .sensor_mask
            .indexed_iter()
            .filter_map(|(xy, sensed)| sensed.then_some(Pos::from(xy)))
        {
            if self.explored_nodes_map[pos.as_index()] {
                continue;
            }

            let reflected = pos.reflect(self.map_size);
            self.explored_nodes_map[pos.as_index()] = true;
            self.explored_nodes_map[reflected.as_index()] = true;
            self.update_known_pointless_map(pos);
            self.update_known_pointless_map(reflected);
        }

        if self.check_if_all_relic_nodes_found() {
            self.register_all_relic_nodes_found()
        }
    }

    fn update_known_pointless_map(&mut self, seen_pos: Pos) {
        let map_size = self.map_size;
        for pos in (-RELIC_WINDOW..=RELIC_WINDOW)
            .cartesian_product(-RELIC_WINDOW..=RELIC_WINDOW)
            .filter_map(|(dx, dy)| seen_pos.maybe_translate([dx, dy], map_size))
        {
            if !self.explored_points_map[pos.as_index()]
                && self.known_pointless(pos)
            {
                self.set_known_points(pos, false);
            }
        }
    }

    fn known_pointless(&self, base_pos: Pos) -> bool {
        (-RELIC_WINDOW..=RELIC_WINDOW)
            .cartesian_product(-RELIC_WINDOW..=RELIC_WINDOW)
            .filter_map(|(dx, dy)| {
                base_pos.maybe_translate([dx, dy], self.map_size)
            })
            .all(|pos| {
                self.explored_nodes_map[pos.as_index()]
                    && !self.relic_nodes.contains(&pos)
            })
    }

    fn set_known_points(&mut self, pos: Pos, worth_points: bool) {
        if self.explored_points_map[pos.as_index()] {
            assert_eq!(
                worth_points,
                self.known_and_explored_points_map[pos.as_index()]
            );
        }

        self.explored_points_map[pos.as_index()] = true;
        self.known_and_explored_points_map[pos.as_index()] = worth_points;
        self.estimated_unexplored_points_map[pos.as_index()] = 0.0;

        let reflected = pos.reflect(self.map_size);
        self.explored_points_map[reflected.as_index()] = true;
        self.known_and_explored_points_map[reflected.as_index()] = worth_points;
        self.estimated_unexplored_points_map[reflected.as_index()] = 0.0;
    }

    fn update_points_map(&mut self, obs: &Observation) {
        if obs.is_new_match() {
            assert_eq!(obs.team_points, [0, 0]);
            self.points_last_turn = 0;
            return;
        }

        let new_points: usize =
            (obs.get_my_points() - self.points_last_turn) as usize;
        self.points_last_turn = obs.get_my_points();
        let (known_locations, frontier_locations): (
            BTreeSet<Pos>,
            BTreeSet<Pos>,
        ) = obs.units[obs.team_id]
            .iter()
            .filter(|u| u.alive())
            .map(|u| u.pos)
            .partition(|pos| self.explored_points_map[pos.as_index()]);
        let unaccounted_new_points = new_points
            - known_locations
                .into_iter()
                .filter(|pos| {
                    self.known_and_explored_points_map[pos.as_index()]
                })
                .count();
        if unaccounted_new_points == 0 {
            frontier_locations.into_iter().for_each(|pos| {
                self.set_known_points(pos, false);
            });
        } else if unaccounted_new_points == frontier_locations.len() {
            frontier_locations.into_iter().for_each(|pos| {
                self.set_known_points(pos, true);
            });
        } else if unaccounted_new_points > frontier_locations.len() {
            panic!(
                "unaccounted_new_points {} > frontier_locations.len() {}",
                unaccounted_new_points,
                frontier_locations.len()
            );
        } else {
            let mean_points =
                unaccounted_new_points as f32 / frontier_locations.len() as f32;
            frontier_locations.into_iter().for_each(|pos| {
                let reflected = pos.reflect(self.map_size);
                self.points_sum_map[pos.as_index()] += mean_points;
                self.points_sum_map[reflected.as_index()] += mean_points;
                self.points_count_map[pos.as_index()] += 1;
                self.points_count_map[reflected.as_index()] += 1;
                self.estimated_unexplored_points_map[pos.as_index()] = self
                    .points_sum_map[pos.as_index()]
                    / self.points_count_map[pos.as_index()] as f32;
                self.estimated_unexplored_points_map[reflected.as_index()] =
                    self.points_sum_map[reflected.as_index()]
                        / self.points_count_map[reflected.as_index()] as f32;
            });
        }
    }

    fn register_all_relic_nodes_found(&mut self) {
        self.all_nodes_registered = true;
        self.explored_nodes_map.fill(true);
        let mut potential_points_mask = Array2::default(self.map_size);
        for pos in self
            .relic_nodes
            .iter()
            .cartesian_product(-RELIC_WINDOW..=RELIC_WINDOW)
            .cartesian_product(-RELIC_WINDOW..=RELIC_WINDOW)
            .filter_map(|((rn, dx), dy)| {
                rn.maybe_translate([dx, dy], self.map_size)
            })
        {
            potential_points_mask[pos.as_index()] = true;
        }
        Zip::from(&mut self.known_and_explored_points_map)
            .and(&mut self.estimated_unexplored_points_map)
            .and(&mut self.explored_points_map)
            .and(&potential_points_mask)
            .for_each(
                |known_points,
                 estimated_points,
                 explored_points,
                 potential_points| {
                    if !potential_points {
                        *known_points = false;
                        *estimated_points = 0.0;
                        *explored_points = true;
                    }
                },
            )
    }
}

#[cfg(test)]
impl RelicNodeMemory {
    pub fn get_all_nodes_registered(&self) -> bool {
        self.all_nodes_registered
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules_engine::state::Unit;
    use numpy::ndarray::arr2;

    #[test]
    fn test_relic_window() {
        assert_eq!(
            RELIC_WINDOW * 2 + 1,
            FIXED_PARAMS.relic_config_size as isize
        );
    }

    #[test]
    fn test_update_explored_nodes() {
        let map_size = [4, 4];
        let mut memory = RelicNodeMemory::new(map_size);
        let obs = Observation {
            relic_node_locations: vec![Pos::new(0, 0), Pos::new(0, 1)],
            sensor_mask: arr2(&[
                [true, true, false, false],
                [false, true, false, false],
                [false, false, false, false],
                [false, false, false, false],
            ]),
            ..Default::default()
        };
        memory.update_explored_nodes(&obs);
        assert_eq!(
            memory.relic_nodes.iter().copied().sorted().collect_vec(),
            vec![
                Pos::new(0, 0),
                Pos::new(0, 1),
                Pos::new(2, 3),
                Pos::new(3, 3)
            ]
        );
        assert_eq!(
            memory.explored_nodes_map,
            arr2(&[
                [true, true, false, false],
                [false, true, false, false],
                [false, false, true, true],
                [false, false, false, true],
            ])
        );
        assert!(!memory.check_if_all_relic_nodes_found());
        assert!(!memory.all_nodes_registered);

        let obs = Observation {
            sensor_mask: arr2(&[
                [false, true, false, true],
                [false, true, true, true],
                [false, false, false, false],
                [false, false, false, false],
            ]),
            ..Default::default()
        };
        memory.update_explored_nodes(&obs);
        assert_eq!(
            memory.explored_nodes_map,
            arr2(&[
                [true, true, true, true],
                [false, true, true, true],
                [false, false, true, true],
                [false, false, false, true],
            ])
        );

        let obs = Observation {
            relic_node_locations: vec![Pos::new(1, 1)],
            sensor_mask: Array2::default(map_size),
            ..Default::default()
        };
        memory.update_explored_nodes(&obs);
        assert_eq!(
            memory.relic_nodes.iter().copied().sorted().collect_vec(),
            vec![
                Pos::new(0, 0),
                Pos::new(0, 1),
                Pos::new(1, 1),
                Pos::new(2, 2),
                Pos::new(2, 3),
                Pos::new(3, 3)
            ]
        );
        assert!(memory.explored_nodes_map.iter().all(|explored| *explored));
        assert!(memory.check_if_all_relic_nodes_found());
        assert!(memory.all_nodes_registered);
    }

    #[test]
    fn test_update_points_map() {
        let map_size = [4, 4];
        let mut memory = RelicNodeMemory::new(map_size);
        let obs = Observation {
            match_steps: 1,
            units: [
                vec![
                    Unit::with_pos(Pos::new(0, 0)),
                    Unit::with_pos(Pos::new(0, 1)),
                ],
                Vec::new(),
            ],
            team_points: [1, 0],
            ..Default::default()
        };
        let pos = Pos::new(0, 0);
        memory.explored_points_map[pos.as_index()] = true;
        memory.explored_points_map[pos.reflect(map_size).as_index()] = true;
        memory.update_points_map(&obs);
        assert_eq!(memory.points_last_turn, 1);
        assert_eq!(
            memory.explored_points_map,
            arr2(&[
                [true, true, false, false],
                [false, false, false, false],
                [false, false, false, true],
                [false, false, false, true]
            ])
        );
        assert_eq!(
            memory.known_and_explored_points_map,
            arr2(&[
                [false, true, false, false],
                [false, false, false, false],
                [false, false, false, true],
                [false, false, false, false]
            ])
        );

        let obs = Observation {
            match_steps: 1,
            units: [
                vec![
                    Unit::with_pos(Pos::new(1, 0)),
                    Unit::with_pos(Pos::new(1, 1)),
                ],
                Vec::new(),
            ],
            team_points: [3, 0],
            ..Default::default()
        };
        memory.update_points_map(&obs);
        assert_eq!(memory.points_last_turn, 3);
        assert_eq!(
            memory.explored_points_map,
            arr2(&[
                [true, true, false, false],
                [true, true, false, false],
                [false, false, true, true],
                [false, false, true, true]
            ])
        );
        assert_eq!(
            memory.known_and_explored_points_map,
            arr2(&[
                [false, true, false, false],
                [true, true, false, false],
                [false, false, true, true],
                [false, false, true, false]
            ])
        );

        let obs = Observation {
            match_steps: 1,
            units: [
                vec![
                    Unit::with_pos(Pos::new(0, 0)),
                    Unit::with_pos(Pos::new(2, 0)),
                    Unit::with_pos(Pos::new(2, 1)),
                ],
                Vec::new(),
            ],
            team_points: [4, 0],
            ..Default::default()
        };
        memory.update_points_map(&obs);
        assert_eq!(memory.points_last_turn, 4);
        assert_eq!(
            memory.explored_points_map,
            arr2(&[
                [true, true, false, false],
                [true, true, false, false],
                [false, false, true, true],
                [false, false, true, true]
            ])
        );
        assert_eq!(
            memory.known_and_explored_points_map,
            arr2(&[
                [false, true, false, false],
                [true, true, false, false],
                [false, false, true, true],
                [false, false, true, false]
            ])
        );
        assert_eq!(
            memory.estimated_unexplored_points_map,
            arr2(&[
                [0., 0., 0., 0.],
                [0., 0., 0., 0.],
                [0.5, 0.5, 0., 0.],
                [0., 0.5, 0., 0.]
            ])
        )
    }

    #[test]
    #[should_panic(expected = "unaccounted_new_points")]
    fn test_update_points_map_panics() {
        let map_size = [4, 4];
        let mut memory = RelicNodeMemory::new(map_size);
        let obs = Observation {
            match_steps: 1,
            units: [vec![Unit::with_pos(Pos::new(0, 0))], Vec::new()],
            team_points: [1, 0],
            ..Default::default()
        };
        memory.explored_points_map[[0, 0]] = true;
        memory.update_points_map(&obs);
    }

    #[test]
    fn test_register_all_relic_nodes_found() {
        let map_size = [5, 5];
        let mut memory = RelicNodeMemory::new(map_size);
        memory.relic_nodes = vec![Pos::new(0, 3), Pos::new(1, 4)];
        memory.known_and_explored_points_map.fill(false);
        memory.estimated_unexplored_points_map.fill(0.5);

        memory.set_known_points(Pos::new(0, 4), true);
        memory.register_all_relic_nodes_found();
        assert!(memory.all_nodes_registered);
        assert_eq!(
            memory.explored_nodes_map,
            Array2::from_elem(map_size, true)
        );
        assert_eq!(
            memory.known_and_explored_points_map,
            arr2(&[
                [false, false, false, false, true],
                [false; 5],
                [false; 5],
                [false; 5],
                [false; 5],
            ])
        );
        assert_eq!(
            memory.estimated_unexplored_points_map,
            arr2(&[
                [0.0, 0.5, 0.5, 0.5, 0.0],
                [0.0, 0.5, 0.5, 0.5, 0.5],
                [0.0, 0.5, 0.5, 0.5, 0.5],
                [0.0, 0.0, 0.5, 0.5, 0.5],
                [0.0, 0.0, 0.0, 0.0, 0.0]
            ])
        );
        assert_eq!(
            memory.explored_points_map,
            arr2(&[
                [true, false, false, false, true],
                [true, false, false, false, false],
                [true, false, false, false, false],
                [true, true, false, false, false],
                [true, true, true, true, true],
            ])
        );
    }
}
