use crate::rules_engine::params::FIXED_PARAMS;
use crate::rules_engine::state::{Observation, Pos};
use itertools::Itertools;
use numpy::ndarray::{Array2, Zip};
use std::collections::BTreeSet;

const RELIC_WINDOW: isize = FIXED_PARAMS.relic_config_size as isize / 2;

/// Tracks everything known by a player so far about relic nodes
#[derive(Debug)]
pub struct RelicNodeMemory {
    pub relic_nodes: Vec<Pos>,
    /// True if this tile has been explored for relic nodes
    pub explored_nodes: Array2<bool>,
    /// True if this tile is known to have points
    pub known_to_have_points: Array2<bool>,
    /// An estimate of the point value of this tile, if unknown
    pub estimated_unexplored_points: Array2<f32>,
    /// True if this tile is known to have (or to not have) points
    pub explored_points: Array2<bool>,
    points_sum: Array2<f32>,
    points_count: Array2<u32>,
    points_last_turn: u32,
    match_nodes_registered: bool,
    map_size: [usize; 2],
}

impl RelicNodeMemory {
    pub fn new(map_size: [usize; 2]) -> Self {
        RelicNodeMemory {
            relic_nodes: Vec::with_capacity(FIXED_PARAMS.max_relic_nodes),
            explored_nodes: Array2::default(map_size),
            known_to_have_points: Array2::default(map_size),
            estimated_unexplored_points: Array2::zeros(map_size),
            explored_points: Array2::default(map_size),
            points_sum: Array2::zeros(map_size),
            points_count: Array2::zeros(map_size),
            points_last_turn: 0,
            match_nodes_registered: false,
            map_size,
        }
    }

    fn check_if_all_match_relic_nodes_found(&self, match_num: u32) -> bool {
        self.relic_nodes.len() >= get_max_nodes_spawned_so_far(match_num)
            || self.explored_nodes.iter().all(|explored| *explored)
    }

    fn all_match_nodes_have_spawned(&self, obs: &Observation) -> bool {
        let match_num = obs.get_match();
        obs.match_steps >= FIXED_PARAMS.max_steps_in_match / 2
            || !node_could_spawn_this_match(match_num)
            || self.relic_nodes.len() >= get_max_nodes_spawned_so_far(match_num)
    }

    pub fn update(&mut self, obs: &Observation) {
        if obs.is_new_match() && node_could_spawn_this_match(obs.get_match()) {
            self.prepare_for_new_relic_nodes_to_spawn();
        }
        self.update_explored_nodes(obs);
        self.update_points_map(obs);
    }

    fn prepare_for_new_relic_nodes_to_spawn(&mut self) {
        self.match_nodes_registered = false;
        self.explored_nodes.fill(false);
        for node in &self.relic_nodes {
            self.explored_nodes[node.as_index()] = true
        }
        self.explored_points.assign(&self.known_to_have_points);
    }

    fn update_explored_nodes(&mut self, obs: &Observation) {
        if self.match_nodes_registered {
            return;
        }

        for pos in obs.relic_node_locations.iter() {
            if self.relic_nodes.contains(pos) {
                continue;
            }

            self.relic_nodes.push(*pos);
            self.explored_nodes[pos.as_index()] = true;

            let reflected = pos.reflect(self.map_size);
            self.relic_nodes.push(reflected);
            self.explored_nodes[reflected.as_index()] = true;
        }

        if self.all_match_nodes_have_spawned(obs) {
            for pos in obs
                .sensor_mask
                .indexed_iter()
                .filter_map(|(xy, sensed)| sensed.then_some(Pos::from(xy)))
            {
                if self.explored_nodes[pos.as_index()] {
                    continue;
                }

                self.explored_nodes[pos.as_index()] = true;
                self.update_known_pointless_map(pos);

                let reflected = pos.reflect(self.map_size);
                self.explored_nodes[reflected.as_index()] = true;
                self.update_known_pointless_map(reflected);
            }
        }

        if self.check_if_all_match_relic_nodes_found(obs.get_match()) {
            self.register_all_relic_nodes_found()
        }
    }

    fn update_known_pointless_map(&mut self, seen_pos: Pos) {
        let map_size = self.map_size;
        for pos in (-RELIC_WINDOW..=RELIC_WINDOW)
            .cartesian_product(-RELIC_WINDOW..=RELIC_WINDOW)
            .filter_map(|(dx, dy)| seen_pos.maybe_translate([dx, dy], map_size))
        {
            if !self.explored_points[pos.as_index()]
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
                self.explored_nodes[pos.as_index()]
                    && !self.relic_nodes.contains(&pos)
            })
    }

    fn set_known_points(&mut self, pos: Pos, worth_points: bool) {
        if self.explored_points[pos.as_index()] {
            assert_eq!(worth_points, self.known_to_have_points[pos.as_index()]);
        }

        self.explored_points[pos.as_index()] = true;
        self.known_to_have_points[pos.as_index()] = worth_points;
        self.estimated_unexplored_points[pos.as_index()] = 0.0;
        self.points_sum[pos.as_index()] = 0.0;
        self.points_count[pos.as_index()] = 0;

        let reflected = pos.reflect(self.map_size);
        self.explored_points[reflected.as_index()] = true;
        self.known_to_have_points[reflected.as_index()] = worth_points;
        self.estimated_unexplored_points[reflected.as_index()] = 0.0;
        self.points_sum[reflected.as_index()] = 0.0;
        self.points_count[reflected.as_index()] = 0;
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
            .partition(|pos| self.explored_points[pos.as_index()]);
        let unaccounted_new_points = new_points
            - known_locations
                .into_iter()
                .filter(|pos| self.known_to_have_points[pos.as_index()])
                .count();
        if unaccounted_new_points == 0 {
            if self.all_match_nodes_have_spawned(obs) {
                for pos in frontier_locations {
                    self.set_known_points(pos, false);
                }
            }
        } else if unaccounted_new_points == frontier_locations.len() {
            for pos in frontier_locations {
                self.set_known_points(pos, true);
            }
        } else if unaccounted_new_points > frontier_locations.len() {
            panic!(
                "unaccounted_new_points {} > frontier_locations.len() {}",
                unaccounted_new_points,
                frontier_locations.len()
            );
        } else {
            let mean_points =
                unaccounted_new_points as f32 / frontier_locations.len() as f32;
            for pos in frontier_locations {
                self.points_sum[pos.as_index()] += mean_points;
                self.points_count[pos.as_index()] += 1;
                self.estimated_unexplored_points[pos.as_index()] = self
                    .points_sum[pos.as_index()]
                    / self.points_count[pos.as_index()] as f32;

                let reflected = pos.reflect(self.map_size);
                self.points_sum[reflected.as_index()] += mean_points;
                self.points_count[reflected.as_index()] += 1;
                self.estimated_unexplored_points[reflected.as_index()] = self
                    .points_sum[reflected.as_index()]
                    / self.points_count[reflected.as_index()] as f32;
            }
        }
    }

    fn register_all_relic_nodes_found(&mut self) {
        self.match_nodes_registered = true;
        self.explored_nodes.fill(true);

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
        Zip::from(&mut self.known_to_have_points)
            .and(&mut self.estimated_unexplored_points)
            .and(&mut self.explored_points)
            .and(&mut self.points_sum)
            .and(&mut self.points_count)
            .and(&potential_points_mask)
            .for_each(
                |known_points,
                 estimated_points,
                 explored_points,
                 points_sum,
                 points_count,
                 potential_points| {
                    if !potential_points {
                        *known_points = false;
                        *estimated_points = 0.0;
                        *explored_points = true;
                        *points_sum = 0.0;
                        *points_count = 0;
                    }
                },
            )
    }
}

fn get_max_nodes_spawned_so_far(match_num: u32) -> usize {
    FIXED_PARAMS
        .max_relic_nodes
        .min(2 * (match_num as usize + 1))
}

fn node_could_spawn_this_match(match_num: u32) -> bool {
    2 * (match_num as usize + 1) <= FIXED_PARAMS.max_relic_nodes
}

#[cfg(test)]
impl RelicNodeMemory {
    pub fn get_all_nodes_registered(&self) -> bool {
        self.match_nodes_registered
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
            team_wins: [2, 2],
            match_steps: FIXED_PARAMS.max_steps_in_match / 2,
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
            memory.explored_nodes,
            arr2(&[
                [true, true, false, false],
                [false, true, false, false],
                [false, false, true, true],
                [false, false, false, true],
            ])
        );
        assert!(!memory.check_if_all_match_relic_nodes_found(4));
        assert!(!memory.match_nodes_registered);

        let obs = Observation {
            sensor_mask: arr2(&[
                [false, true, false, true],
                [false, true, true, true],
                [false, false, false, false],
                [false, false, false, false],
            ]),
            team_wins: [2, 2],
            match_steps: FIXED_PARAMS.max_steps_in_match / 2,
            ..Default::default()
        };
        memory.update_explored_nodes(&obs);
        assert_eq!(
            memory.explored_nodes,
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
            team_wins: [2, 2],
            match_steps: FIXED_PARAMS.max_steps_in_match / 2,
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
        assert!(memory.explored_nodes.iter().all(|explored| *explored));
        assert!(memory.check_if_all_match_relic_nodes_found(4));
        assert!(memory.match_nodes_registered);
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
        memory.explored_points[pos.as_index()] = true;
        memory.explored_points[pos.reflect(map_size).as_index()] = true;
        memory.update_points_map(&obs);
        assert_eq!(memory.points_last_turn, 1);
        assert_eq!(
            memory.explored_points,
            arr2(&[
                [true, true, false, false],
                [false, false, false, false],
                [false, false, false, true],
                [false, false, false, true]
            ])
        );
        assert_eq!(
            memory.known_to_have_points,
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
            memory.explored_points,
            arr2(&[
                [true, true, false, false],
                [true, true, false, false],
                [false, false, true, true],
                [false, false, true, true]
            ])
        );
        assert_eq!(
            memory.known_to_have_points,
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
            memory.explored_points,
            arr2(&[
                [true, true, false, false],
                [true, true, false, false],
                [false, false, true, true],
                [false, false, true, true]
            ])
        );
        assert_eq!(
            memory.known_to_have_points,
            arr2(&[
                [false, true, false, false],
                [true, true, false, false],
                [false, false, true, true],
                [false, false, true, false]
            ])
        );
        assert_eq!(
            memory.estimated_unexplored_points,
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
        memory.explored_points[[0, 0]] = true;
        memory.update_points_map(&obs);
    }

    #[test]
    fn test_register_all_relic_nodes_found() {
        let map_size = [5, 5];
        let mut memory = RelicNodeMemory::new(map_size);
        memory.relic_nodes = vec![Pos::new(0, 3), Pos::new(1, 4)];
        memory.known_to_have_points.fill(false);
        memory.estimated_unexplored_points.fill(0.5);

        memory.set_known_points(Pos::new(0, 4), true);
        memory.register_all_relic_nodes_found();
        assert!(memory.match_nodes_registered);
        assert_eq!(memory.explored_nodes, Array2::from_elem(map_size, true));
        assert_eq!(
            memory.known_to_have_points,
            arr2(&[
                [false, false, false, false, true],
                [false; 5],
                [false; 5],
                [false; 5],
                [false; 5],
            ])
        );
        assert_eq!(
            memory.estimated_unexplored_points,
            arr2(&[
                [0.0, 0.5, 0.5, 0.5, 0.0],
                [0.0, 0.5, 0.5, 0.5, 0.5],
                [0.0, 0.5, 0.5, 0.5, 0.5],
                [0.0, 0.0, 0.5, 0.5, 0.5],
                [0.0, 0.0, 0.0, 0.0, 0.0]
            ])
        );
        assert_eq!(
            memory.explored_points,
            arr2(&[
                [true, false, false, false, true],
                [true, false, false, false, false],
                [true, false, false, false, false],
                [true, true, false, false, false],
                [true, true, true, true, true],
            ])
        );
    }

    #[test]
    fn test_get_max_nodes_spawned_so_far() {
        assert_eq!(get_max_nodes_spawned_so_far(0), 2);
        assert_eq!(get_max_nodes_spawned_so_far(1), 4);
        assert_eq!(
            get_max_nodes_spawned_so_far(2),
            FIXED_PARAMS.max_relic_nodes
        );
        assert_eq!(
            get_max_nodes_spawned_so_far(3),
            FIXED_PARAMS.max_relic_nodes
        );
        assert_eq!(
            get_max_nodes_spawned_so_far(4),
            FIXED_PARAMS.max_relic_nodes
        );
    }

    #[test]
    fn test_node_could_spawn_this_match() {
        assert!(node_could_spawn_this_match(0));
        assert!(node_could_spawn_this_match(1));
        assert!(node_could_spawn_this_match(2));
        assert!(!node_could_spawn_this_match(3));
        assert!(!node_could_spawn_this_match(4));
    }
}
