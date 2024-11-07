use serde::Deserialize;

pub const TEAMS: usize = 2;
pub const MAP_WIDTH: usize = 24;
pub const MAP_HEIGHT: usize = 24;
pub const MAP_SIZE: [usize; 2] = [MAP_WIDTH, MAP_HEIGHT];
pub const MAX_RELIC_NODES: usize = 6;

#[derive(Debug, Clone, Deserialize)]
pub struct Params {
    pub max_steps_in_match: u32,
    pub map_width: usize,
    pub map_height: usize,
    pub match_count_per_episode: u32,

    // configs for units
    pub max_units: usize,
    pub init_unit_energy: i32,
    pub min_unit_energy: i32,
    pub max_unit_energy: i32,
    pub unit_move_cost: i32,
    pub spawn_rate: u32,
    // The unit sap cost is the amount of energy a unit uses when it saps another unit.
    // Can change between games.
    pub unit_sap_cost: i32,
    // The unit sap range is the range of the unit's sap action.
    pub unit_sap_range: isize,
    // The unit sap dropoff factor multiplied by unit_sap_drain
    pub unit_sap_dropoff_factor: f32,
    // The unit energy void factor multiplied by unit_energy
    pub unit_energy_void_factor: f32,

    // configs for energy nodes
    pub max_energy_nodes: usize,
    pub max_energy_per_tile: i32,
    pub min_energy_per_tile: i32,

    // configs for relic nodes
    pub max_relic_nodes: usize,
    pub relic_config_size: usize,
    // The unit sensor range is the range of the unit's sensor.
    // Units provide "vision power" over tiles in range, equal to manhattan distance
    // to the unit.
    // vision power > 0 that team can see the tiles properties
    pub unit_sensor_range: usize,
    // nebula tile params
    // The nebula tile vision reduction is the amount of vision reduction a nebula
    // tile provides. A tile can be seen if the vision power over it is > 0.
    pub nebula_tile_vision_reduction: i32,
    // amount of energy nebula tiles reduce from a unit
    pub nebula_tile_energy_reduction: i32,
    // how fast nebula tiles drift in one of the diagonal directions over time.
    // If positive, flows to the top/right, negative flows to bottom/left
    pub nebula_tile_drift_speed: f32,
    // how fast energy nodes will move around over time
    pub energy_node_drift_speed: f32,
    pub energy_node_drift_magnitude: f32,
}

impl Params {
    #[inline(always)]
    pub fn get_map_size(&self) -> [usize; 2] {
        [self.map_width, self.map_height]
    }
}

impl Default for Params {
    fn default() -> Self {
        Self {
            max_steps_in_match: 100,
            map_width: MAP_WIDTH,
            map_height: MAP_HEIGHT,
            match_count_per_episode: 5,
            max_units: 16,
            init_unit_energy: 100,
            min_unit_energy: 0,
            max_unit_energy: 400,
            unit_move_cost: 2,
            spawn_rate: 3,
            unit_sap_cost: 10,
            unit_sap_range: 4,
            unit_sap_dropoff_factor: 0.5,
            unit_energy_void_factor: 0.125,
            max_energy_nodes: 6,
            max_energy_per_tile: 20,
            min_energy_per_tile: -20,
            max_relic_nodes: MAX_RELIC_NODES,
            relic_config_size: 5,
            unit_sensor_range: 2,
            nebula_tile_vision_reduction: 1,
            nebula_tile_energy_reduction: 0,
            nebula_tile_drift_speed: -0.05,
            energy_node_drift_speed: 0.02,
            energy_node_drift_magnitude: 5.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct KnownVariableParams {
    pub unit_move_cost: i32,
    pub unit_sap_cost: i32,
    pub unit_sap_range: isize,
    pub unit_sensor_range: usize,
}

impl From<Params> for KnownVariableParams {
    fn from(params: Params) -> Self {
        Self {
            unit_move_cost: params.unit_move_cost,
            unit_sap_cost: params.unit_sap_cost,
            unit_sap_range: params.unit_sap_range,
            unit_sensor_range: params.unit_sensor_range,
        }
    }
}

impl Default for KnownVariableParams {
    fn default() -> Self {
        Self::from(Params::default())
    }
}
