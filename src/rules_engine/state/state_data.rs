use crate::rules_engine::params::{FIXED_PARAMS, P};
#[cfg(test)]
use itertools::Itertools;
use numpy::ndarray::Array2;
use std::array::TryFromSliceError;
use std::cmp::{max, min};
use std::num::TryFromIntError;

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

    pub fn inverted_wrapped_translate(
        self,
        deltas: [isize; 2],
        map_size: [usize; 2],
    ) -> Self {
        let [dx, dy] = deltas;
        self.wrapped_translate([-dx, -dy], map_size)
    }

    #[inline]
    pub fn subtract(self, target: Self) -> [isize; 2] {
        [
            self.x as isize - target.x as isize,
            self.y as isize - target.y as isize,
        ]
    }

    #[inline]
    pub fn reflect(self, [map_width, map_height]: [usize; 2]) -> Self {
        Pos {
            x: map_height - 1 - self.y,
            y: map_width - 1 - self.x,
        }
    }

    #[inline]
    pub fn as_index(&self) -> [usize; 2] {
        [self.x, self.y]
    }

    #[inline]
    pub fn manhattan_distance(self, target: Self) -> usize {
        self.x.abs_diff(target.x) + self.y.abs_diff(target.y)
    }
}

impl From<[usize; 2]> for Pos {
    fn from([x, y]: [usize; 2]) -> Self {
        Self { x, y }
    }
}

impl From<Pos> for [usize; 2] {
    fn from(value: Pos) -> Self {
        [value.x, value.y]
    }
}

impl From<(usize, usize)> for Pos {
    fn from((x, y): (usize, usize)) -> Self {
        Self { x, y }
    }
}

impl TryFrom<[isize; 2]> for Pos {
    type Error = TryFromIntError;

    fn try_from(value: [isize; 2]) -> Result<Self, Self::Error> {
        let [x, y] = value;
        Ok(Self {
            x: x.try_into()?,
            y: y.try_into()?,
        })
    }
}

impl TryFrom<&[usize]> for Pos {
    type Error = TryFromSliceError;

    fn try_from(value: &[usize]) -> Result<Self, Self::Error> {
        let array: [usize; 2] = value.try_into()?;
        Ok(Self::from(array))
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

    #[inline]
    pub fn alive(&self) -> bool {
        self.energy >= 0
    }
}

#[cfg(test)]
impl Unit {
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
    pub fn new(pos: Pos, func_id: u8, [x, y, z]: [f32; 3]) -> Self {
        EnergyNode {
            pos,
            func_id,
            x,
            y,
            z,
        }
    }

    pub fn from_pos_and_energy_fn(
        pos: Pos,
        (func_id, x, y, z): (u8, f32, f32, f32),
    ) -> Self {
        EnergyNode {
            pos,
            func_id,
            x,
            y,
            z,
        }
    }

    pub fn apply_energy_fn(&self, d: f32) -> f32 {
        match self.func_id {
            0 => sin_energy_fn(d, self.x, self.y, self.z),
            1 => div_energy_fn(d, self.x, self.y, self.z),
            _ => panic!("Invalid energy_fn id {}", self.func_id),
        }
    }

    #[cfg(test)]
    pub fn new_at(pos: Pos) -> Self {
        EnergyNode {
            pos,
            func_id: 0,
            x: 0.,
            y: 0.,
            z: 0.,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct RelicSpawn {
    pub spawn_step: u32,
    pub pos: Pos,
    pub config: Array2<bool>,
}

impl RelicSpawn {
    pub fn new(spawn_step: u32, pos: Pos, config: Array2<bool>) -> Self {
        Self {
            spawn_step,
            pos,
            config,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct State {
    pub units: [Vec<Unit>; P],
    pub asteroids: Vec<Pos>,
    pub nebulae: Vec<Pos>,
    pub energy_nodes: Vec<EnergyNode>,
    pub energy_field: Array2<i32>,
    pub relic_node_locations: Vec<Pos>,
    pub relic_node_points_map: Array2<bool>,
    pub relic_node_spawn_schedule: Vec<RelicSpawn>,
    pub team_points: [u32; P],
    pub team_wins: [u32; P],
    pub total_steps: u32,
    pub match_steps: u32,
    pub done: bool,
}

impl State {
    pub fn initialize_relic_nodes(
        &mut self,
        relic_node_spawn_schedule: Vec<RelicSpawn>,
        map_size: [usize; 2],
    ) {
        self.relic_node_locations =
            Vec::with_capacity(relic_node_spawn_schedule.len());
        self.relic_node_points_map = Array2::default(map_size);
        self.relic_node_spawn_schedule = relic_node_spawn_schedule;
    }

    pub fn add_relic_node(
        &mut self,
        pos: Pos,
        config: &Array2<bool>,
        relic_config_size: usize,
        map_size: [usize; 2],
    ) {
        self.relic_node_locations.push(pos);
        let offset = (relic_config_size / 2) as isize;
        for point_pos in config
            .indexed_iter()
            .filter_map(|((x, y), p)| {
                p.then_some([x as isize - offset, y as isize - offset])
            })
            .filter_map(|deltas| pos.maybe_translate(deltas, map_size))
        {
            self.relic_node_points_map[point_pos.as_index()] = true;
        }
    }
}

#[cfg(test)]
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
        self.relic_node_locations.sort();
        self.relic_node_spawn_schedule
            .sort_by_key(|rs| (rs.spawn_step, rs.pos));
    }
}

#[derive(Debug, Clone, Copy)]
pub struct GameResult {
    pub points_scored: [u32; P],
    pub match_winner: Option<u8>,
    pub final_winner: Option<u8>,
    pub done: bool,
}

impl GameResult {
    pub fn new(
        points_scored: [u32; P],
        match_winner: Option<u8>,
        final_winner: Option<u8>,
        done: bool,
    ) -> Self {
        Self {
            points_scored,
            match_winner,
            final_winner,
            done,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Observation {
    pub team_id: usize,
    pub units: [Vec<Unit>; P],
    pub sensor_mask: Array2<bool>,
    pub energy_field: Array2<Option<i32>>,
    pub asteroids: Vec<Pos>,
    pub nebulae: Vec<Pos>,
    pub relic_node_locations: Vec<Pos>,
    pub team_points: [u32; P],
    pub team_wins: [u32; P],
    pub total_steps: u32,
    pub match_steps: u32,
}

impl Observation {
    pub fn new(
        team_id: usize,
        sensor_mask: Array2<bool>,
        energy_field: Array2<Option<i32>>,
        team_points: [u32; P],
        team_wins: [u32; P],
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

    #[inline]
    pub fn opp_team_id(&self) -> usize {
        1 - self.team_id
    }

    #[inline]
    pub fn is_new_match(&self) -> bool {
        self.match_steps == 0
    }

    #[inline]
    pub fn get_my_units(&self) -> &[Unit] {
        &self.units[self.team_id]
    }

    #[inline]
    pub fn get_opp_units(&self) -> &[Unit] {
        &self.units[self.opp_team_id()]
    }

    #[inline]
    pub fn get_my_points(&self) -> u32 {
        self.team_points[self.team_id]
    }

    #[inline]
    pub fn get_match(&self) -> u32 {
        self.team_wins.iter().sum()
    }

    /// Sorts the various elements of the Observation. Unnecessary during simulation,
    /// but useful when testing to ensure the various Vecs of components match up.
    #[cfg(test)]
    pub fn sort(&mut self) {
        for team in [0, 1] {
            self.units[team].sort_by(|u1, u2| u1.id.cmp(&u2.id))
        }
        self.asteroids.sort();
        self.nebulae.sort();
        self.relic_node_locations.sort();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules_engine::params::FIXED_PARAMS;

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
    fn test_inverted_wrapped_translate() {
        let map_size = [8, 8];
        let [width, height] = map_size;
        for ((pos, dx), dy) in (0..width)
            .cartesian_product(0..height)
            .map(Pos::from)
            .cartesian_product(-3..=3)
            .cartesian_product(-3..=3)
        {
            let deltas = [dx, dy];
            assert_eq!(
                pos.wrapped_translate(deltas, map_size)
                    .inverted_wrapped_translate(deltas, map_size),
                pos
            );
        }
    }

    #[test]
    fn test_pos_reflect() {
        for pos in (0..FIXED_PARAMS.map_width)
            .cartesian_product(0..FIXED_PARAMS.map_height)
            .map(Pos::from)
        {
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
    fn test_manhattan_distance() {
        let p1 = Pos::new(10, 10);
        assert_eq!(p1.manhattan_distance(Pos::new(10, 10)), 0);
        assert_eq!(p1.manhattan_distance(Pos::new(8, 10)), 2);
        assert_eq!(p1.manhattan_distance(Pos::new(10, 12)), 2);
        assert_eq!(p1.manhattan_distance(Pos::new(7, 15)), 8);
    }
}
