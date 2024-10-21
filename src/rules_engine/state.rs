use numpy::ndarray::Array2;

type EnergyFunc = fn(f32, f32, f32, f32) -> f32;
const ENERGY_NODE_FNS: [EnergyFunc; 2] = [sin_energy_fn, div_energy_fn];

fn sin_energy_fn(d: f32, x: f32, y: f32, z: f32) -> f32 {
    (d * x + y).sin() * z
}

fn div_energy_fn(d: f32, x: f32, y: f32, z: f32) -> f32 {
    (x / (d + 1.) + y) * z
}

#[derive(Debug, Copy, Clone)]
pub struct Pos {
    pub x: usize,
    pub y: usize,
}

impl Pos {
    pub fn as_index(&self) -> [usize; 2] {
        [self.x, self.y]
    }
}

#[derive(Debug, Clone)]
pub struct Unit {
    pub pos: Pos,
    pub energy: i32,
    pub id: u8,
}

#[derive(Debug, Clone)]
pub struct EnergyNode {
    pub pos: Pos,
    pub func: EnergyFunc,
}

// #[derive(Debug, Clone)]
// pub struct MapTile {
//     pub energy: u32,
//     pub tile_type: None, // TODO: tile_type enum
// }

#[derive(Debug, Clone)]
pub struct RelicNode {
    pub pos: Pos,
    // pub config: None, TODO: relic node config
}

#[derive(Debug, Clone)]
pub struct State {
    pub units: [Vec<Unit>; 2],
    pub asteroids: Vec<Pos>,
    pub energy_nodes: Vec<EnergyNode>,
    pub relic_nodes: Vec<RelicNode>,
    pub team_points: [u32; 2],
    pub team_wins: [u32; 2],
    pub total_steps: u32,
    pub match_steps: u32,
}

fn build_asteroid_mask(asteroids: &Vec<Pos>, h: usize, w: usize) -> Array2<bool> {
    let mut result = Array2::default((h, w));
    for a in asteroids.iter() {
        result[[a.x, a.y]] = true;
    }
    result
}

impl State {
    pub fn get_asteroid_mask(&self, h: usize, w: usize) -> Array2<bool> {
        build_asteroid_mask(&self.asteroids, h, w)
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
    use numpy::ndarray::arr2;

    #[test]
    fn test_build_asteroid_mask() {
        let asteroids = Vec::new();
        let expected_result = arr2(&[[false; 3]; 3]);
        assert_eq!(build_asteroid_mask(&asteroids, 3, 3), expected_result);

        let asteroids = vec![Pos { x: 0, y: 0 }, Pos { x: 0, y: 1 }, Pos { x: 2, y: 1 }];
        let expected_result = arr2(&[
            [true, true, false],
            [false, false, false],
            [false, true, false],
        ]);
        assert_eq!(build_asteroid_mask(&asteroids, 3, 3), expected_result);
    }
}
