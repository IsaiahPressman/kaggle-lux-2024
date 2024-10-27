use numpy::ndarray::Array2;
use std::cmp::{max, min};

fn sin_energy_fn(d: f32, x: f32, y: f32, z: f32) -> f32 {
    (d * x + y).sin() * z
}

fn div_energy_fn(d: f32, x: f32, y: f32, z: f32) -> f32 {
    (x / (d + 1.) + y) * z
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
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
        &self,
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
        &self,
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
        &self,
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

    pub fn subtract(&self, target: Self) -> [isize; 2] {
        [
            self.x as isize - target.x as isize,
            self.y as isize - target.y as isize,
        ]
    }

    pub fn as_index(&self) -> [usize; 2] {
        [self.x, self.y]
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Unit {
    pub pos: Pos,
    pub energy: i32,
    pub id: usize,
}

impl Unit {
    pub fn new(pos: Pos, energy: i32) -> Self {
        Unit { pos, energy, id: 0 }
    }

    pub fn new_at(pos: Pos) -> Self {
        Unit {
            pos,
            energy: 0,
            id: 0,
        }
    }

    pub fn new_with_id(pos: Pos, energy: i32, id: usize) -> Self {
        Unit { pos, energy, id }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct EnergyNode {
    pub pos: Pos,
    func_id: u8,
    x: f32,
    y: f32,
    z: f32,
}

impl EnergyNode {
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelicNode {
    pub pos: Pos,
    // pub config: None, TODO: relic node config
}

#[derive(Debug, Clone, PartialEq)]
pub struct State {
    pub units: [Vec<Unit>; 2],
    pub asteroids: Vec<Pos>,
    pub nebulae: Vec<Pos>,
    pub energy_nodes: Vec<EnergyNode>,
    pub relic_nodes: Vec<RelicNode>,
    pub team_points: [u32; 2],
    pub team_wins: [u32; 2],
    pub total_steps: u32,
    pub match_steps: u32,
}

impl State {
    pub fn empty() -> Self {
        State {
            units: [Vec::new(), Vec::new()],
            asteroids: Vec::new(),
            nebulae: Vec::new(),
            energy_nodes: Vec::new(),
            relic_nodes: Vec::new(),
            team_points: [0, 0],
            team_wins: [0, 0],
            total_steps: 0,
            match_steps: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Observation {
    pub units: Vec<Unit>,
    pub sensor_mask: Array2<bool>,
    pub asteroids: Vec<Pos>,
    pub relic_node_locations: Vec<Pos>,
    pub team_points: [u32; 2],
    pub team_wins: [u32; 2],
    pub total_steps: u32,
    pub match_steps: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
