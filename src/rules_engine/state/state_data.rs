use crate::rules_engine::params::FIXED_PARAMS;
use itertools::Itertools;
use numpy::ndarray::{Array2, ArrayView3};
use std::cmp::{max, min};

fn sin_energy_fn(d: f32, x: f32, y: f32, z: f32) -> f32 {
    (d * x + y).sin() * z
}

fn div_energy_fn(d: f32, x: f32, y: f32, z: f32) -> f32 {
    (x / (d + 1.) + y) * z
}

#[derive(Debug, Copy, Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct Pos {
    pub x: usize,
    pub y: usize,
}

impl Pos {
    pub fn new(x: usize, y: usize) -> Self {
        Pos { x, y }
    }

    /// Translates self \
    /// Stops at map boundaries given by [0, width) / [0, height)
    pub fn bounded_translate(
        self,
        deltas: [isize; 2],
        map_size: [usize; 2],
    ) -> Self {
        let [dx, dy] = deltas;
        let [width, height] = map_size;
        let x = self.x as isize + dx;
        let y = self.y as isize + dy;
        let x = min(max(x, 0) as usize, width - 1);
        let y = min(max(y, 0) as usize, height - 1);
        Pos { x, y }
    }

    /// Translates self \
    /// If result is in-bounds, returns result \
    /// If result is out-of-bounds, returns None
    pub fn maybe_translate(
        self,
        deltas: [isize; 2],
        map_size: [usize; 2],
    ) -> Option<Self> {
        let [dx, dy] = deltas;
        let [width, height] = map_size;
        let x = self.x as isize + dx;
        let y = self.y as isize + dy;
        if x < 0 || x >= width as isize || y < 0 || y >= height as isize {
            None
        } else {
            Some(Pos {
                x: x as usize,
                y: y as usize,
            })
        }
    }

    pub fn wrapped_translate(
        self,
        deltas: [isize; 2],
        map_size: [usize; 2],
    ) -> Self {
        let [dx, dy] = deltas;
        let [width, height] = map_size;
        let (width, height) = (width as isize, height as isize);
        let x = (self.x as isize + dx).rem_euclid(width) as usize;
        let y = (self.y as isize + dy).rem_euclid(height) as usize;
        Pos { x, y }
    }

    pub fn subtract(self, target: Self) -> [isize; 2] {
        [
            self.x as isize - target.x as isize,
            self.y as isize - target.y as isize,
        ]
    }

    pub fn reflect(self, map_size: [usize; 2]) -> Self {
        let [width, height] = map_size;
        Pos {
            x: height - 1 - self.y,
            y: width - 1 - self.x,
        }
    }

    #[inline(always)]
    pub fn as_index(&self) -> [usize; 2] {
        [self.x, self.y]
    }
}

impl From<[usize; 2]> for Pos {
    fn from(value: [usize; 2]) -> Self {
        let [x, y] = value;
        Self { x, y }
    }
}

impl From<Pos> for [usize; 2] {
    fn from(value: Pos) -> Self {
        [value.x, value.y]
    }
}

impl From<&[usize]> for Pos {
    fn from(value: &[usize]) -> Self {
        match value {
            &[x, y] => Self { x, y },
            invalid => {
                panic!("Invalid pos: {:?}", invalid)
            },
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Unit {
    pub pos: Pos,
    pub energy: i32,
    pub id: usize,
}

impl Unit {
    pub fn new(pos: Pos, energy: i32, id: usize) -> Self {
        Unit { pos, energy, id }
    }

    pub fn with_pos(pos: Pos) -> Self {
        Unit {
            pos,
            energy: 0,
            id: 0,
        }
    }

    pub fn with_energy(energy: i32) -> Self {
        Unit {
            pos: Pos::default(),
            energy,
            id: 0,
        }
    }

    pub fn with_id(id: usize) -> Self {
        Unit {
            pos: Pos::default(),
            energy: 0,
            id,
        }
    }

    pub fn with_pos_and_energy(pos: Pos, energy: i32) -> Self {
        Unit { pos, energy, id: 0 }
    }
}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct EnergyNode {
    pub pos: Pos,
    func_id: u8,
    x: f32,
    y: f32,
    z: f32,
}

impl EnergyNode {
    pub fn new(pos: Pos, func_id: u8, xyz: [f32; 3]) -> Self {
        let [x, y, z] = xyz;
        EnergyNode {
            pos,
            func_id,
            x,
            y,
            z,
        }
    }

    pub fn new_at(pos: Pos) -> Self {
        EnergyNode {
            pos,
            func_id: 0,
            x: 0.,
            y: 0.,
            z: 0.,
        }
    }

    pub fn apply_energy_fn(&self, d: f32) -> f32 {
        match self.func_id {
            0 => sin_energy_fn(d, self.x, self.y, self.z),
            1 => div_energy_fn(d, self.x, self.y, self.z),
            _ => panic!("Invalid energy_fn id {}", self.func_id),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct State {
    pub units: [Vec<Unit>; 2],
    pub asteroids: Vec<Pos>,
    pub nebulae: Vec<Pos>,
    pub energy_nodes: Vec<EnergyNode>,
    pub relic_node_locations: Vec<Pos>,
    pub relic_node_points_map: Array2<bool>,
    pub team_points: [u32; 2],
    pub team_wins: [u32; 2],
    pub total_steps: u32,
    pub match_steps: u32,
    pub done: bool,
}

impl State {
    pub fn get_energy_node_deltas(&self, next_state: &Self) -> Vec<[isize; 2]> {
        self.energy_nodes
            .iter()
            .zip_eq(&next_state.energy_nodes)
            .map(|(node, next_node)| next_node.pos.subtract(node.pos))
            .collect()
    }

    /// Sorts the various elements of the State. Unnecessary during simulation, but useful when
    /// testing to ensure the various Vecs of state components match up.
    pub fn sort(&mut self) {
        for team in [0, 1] {
            self.units[team].sort_by(|u1, u2| u1.id.cmp(&u2.id))
        }
        self.asteroids.sort();
        self.nebulae.sort();
        self.energy_nodes.sort_by(|en1, en2| en1.pos.cmp(&en2.pos));
        self.relic_node_locations.sort();
    }

    pub fn set_relic_nodes(
        &mut self,
        locations: Vec<Pos>,
        configs: ArrayView3<bool>,
        map_size: [usize; 2],
        config_size: usize,
    ) {
        assert_eq!(config_size % 2, 1);

        self.relic_node_locations = locations;
        self.relic_node_points_map = Array2::default(map_size);

        let offset = (config_size / 2) as isize;
        for (pos, points_mask) in self
            .relic_node_locations
            .iter()
            .copied()
            .zip_eq(configs.outer_iter())
        {
            for point_pos in points_mask
                .indexed_iter()
                .filter_map(|((x, y), &p)| {
                    p.then_some([x as isize - offset, y as isize - offset])
                })
                .filter_map(|deltas| pos.maybe_translate(deltas, map_size))
            {
                self.relic_node_points_map[point_pos.as_index()] = true;
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct GameResult {
    pub match_winner: Option<u8>,
    pub final_winner: Option<u8>,
}

impl GameResult {
    pub fn new(match_winner: Option<u8>, final_winner: Option<u8>) -> Self {
        GameResult {
            match_winner,
            final_winner,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Observation {
    pub team_id: usize,
    pub units: [Vec<Unit>; 2],
    pub sensor_mask: Array2<bool>,
    pub energy_field: Array2<Option<i32>>,
    pub asteroids: Vec<Pos>,
    pub nebulae: Vec<Pos>,
    pub relic_node_locations: Vec<Pos>,
    pub team_points: [u32; 2],
    pub team_wins: [u32; 2],
    pub total_steps: u32,
    pub match_steps: u32,
}

impl Observation {
    pub fn new(
        team_id: usize,
        sensor_mask: Array2<bool>,
        energy_field: Array2<Option<i32>>,
        team_points: [u32; 2],
        team_wins: [u32; 2],
        total_steps: u32,
        match_steps: u32,
    ) -> Self {
        Observation {
            team_id,
            units: [
                Vec::with_capacity(FIXED_PARAMS.max_units),
                Vec::with_capacity(FIXED_PARAMS.max_units),
            ],
            sensor_mask,
            energy_field,
            asteroids: Vec::new(),
            nebulae: Vec::new(),
            relic_node_locations: Vec::with_capacity(
                FIXED_PARAMS.max_relic_nodes,
            ),
            team_points,
            team_wins,
            total_steps,
            match_steps,
        }
    }

    #[inline(always)]
    pub fn get_my_units(&self) -> &[Unit] {
        &self.units[self.team_id]
    }

    #[inline(always)]
    pub fn get_opp_units(&self) -> &[Unit] {
        &self.units[1 - self.team_id]
    }

    #[inline(always)]
    pub fn get_my_points(&self) -> u32 {
        self.team_points[self.team_id]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules_engine::params::FIXED_PARAMS;
    use numpy::ndarray::{arr2, arr3};

    #[test]
    fn test_pos_wrapped_translate() {
        let map_size = [6, 8];
        assert_eq!(
            Pos::new(0, 0).wrapped_translate([5, 7], map_size),
            Pos::new(5, 7)
        );
        assert_eq!(
            Pos::new(0, 0).wrapped_translate([6, 8], map_size),
            Pos::new(0, 0)
        );
        assert_eq!(
            Pos::new(0, 0)
                .wrapped_translate([6 * 10 + 1, 8 * 15 + 1], map_size),
            Pos::new(1, 1)
        );
        assert_eq!(
            Pos::new(5, 7).wrapped_translate([-5, -7], map_size),
            Pos::new(0, 0)
        );
        assert_eq!(
            Pos::new(0, 0).wrapped_translate([-1, -1], map_size),
            Pos::new(5, 7)
        );
        assert_eq!(
            Pos::new(0, 0)
                .wrapped_translate([-6 * 20 - 2, -8 * 25 - 2], map_size),
            Pos::new(4, 6)
        );
    }

    #[test]
    fn test_pos_reflect() {
        for (x, y) in (0..FIXED_PARAMS.map_width)
            .cartesian_product(0..FIXED_PARAMS.map_height)
        {
            let pos = Pos::new(x, y);
            assert_eq!(
                pos.reflect(FIXED_PARAMS.map_size)
                    .reflect(FIXED_PARAMS.map_size),
                pos
            );
        }

        let map_size = [24, 24];
        assert_eq!(Pos::new(0, 0).reflect(map_size), Pos::new(23, 23));
        assert_eq!(Pos::new(1, 1).reflect(map_size), Pos::new(22, 22));
        assert_eq!(Pos::new(2, 0).reflect(map_size), Pos::new(23, 21));
        assert_eq!(Pos::new(3, 22).reflect(map_size), Pos::new(1, 20));
    }

    #[test]
    fn test_set_relic_nodes() {
        let map_size = [5, 5];
        let relic_config_size = 3;
        let mut state = State {
            relic_node_locations: vec![Pos::new(1, 1)],
            relic_node_points_map: arr2(&[
                [true, true, true, false, false],
                [true, true, true, false, false],
                [true, true, true, false, false],
                [false; 5],
                [false; 5],
            ]),
            ..Default::default()
        };

        let new_locations = vec![Pos::new(0, 0), Pos::new(3, 3)];
        let configs = arr3(&[
            [[true; 3], [true, true, false], [false, false, true]],
            [[false; 3], [true, true, false], [true, true, false]],
        ]);
        state.set_relic_nodes(
            new_locations.clone(),
            configs.view(),
            map_size,
            relic_config_size,
        );
        assert_eq!(state.relic_node_locations, new_locations);
        let expected_points_map = arr2(&[
            [true, false, false, false, false],
            [false, true, false, false, false],
            [false, false, false, false, false],
            [false, false, true, true, false],
            [false, false, true, true, false],
        ]);
        assert_eq!(state.relic_node_points_map, expected_points_map);
    }
}
