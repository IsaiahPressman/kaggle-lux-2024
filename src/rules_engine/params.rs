use serde::Deserialize;

pub const FIXED_PARAMS: FixedParams = FixedParams {
    max_steps_in_match: 100,
    map_width: DEFAULT_MAP_WIDTH,
    map_height: DEFAULT_MAP_HEIGHT,
    map_size: get_default_map_size(),
    match_count_per_episode: 5,
    max_units: 16,
    init_unit_energy: 100,
    min_unit_energy: 0,
    max_unit_energy: 400,
    spawn_rate: 3,
    max_energy_nodes: 6,
    min_energy_per_tile: -20,
    max_energy_per_tile: 20,
    max_relic_nodes: 6,
    relic_config_size: 5,
};

const DEFAULT_MAP_WIDTH: usize = 24;
const DEFAULT_MAP_HEIGHT: usize = 24;
const fn get_default_map_size() -> [usize; 2] {
    [DEFAULT_MAP_WIDTH, DEFAULT_MAP_HEIGHT]
}

#[derive(Debug, PartialEq, Clone, Deserialize)]
pub struct FixedParams {
    pub max_steps_in_match: u32,
    pub map_width: usize,
    pub map_height: usize,
    #[serde(default = "get_default_map_size")]
    pub map_size: [usize; 2],
    pub match_count_per_episode: u32,

    // Configs for units
    pub max_units: usize,
    pub init_unit_energy: i32,
    pub min_unit_energy: i32,
    pub max_unit_energy: i32,
    pub spawn_rate: u32,

    // Configs for energy nodes
    pub max_energy_nodes: usize,
    pub min_energy_per_tile: i32,
    pub max_energy_per_tile: i32,

    // Configs for relic nodes
    pub max_relic_nodes: usize,
    pub relic_config_size: usize,
}

#[cfg(test)]
impl FixedParams {
    /// Sets map_width and map_height along with map_size
    pub fn set_map_size(&mut self, map_size: [usize; 2]) {
        let [width, height] = map_size;
        self.map_width = width;
        self.map_height = height;
        self.map_size = map_size;
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct VariableParams {
    pub unit_move_cost: i32,
    pub unit_sap_cost: i32,
    pub unit_sap_range: isize,
    pub unit_sap_dropoff_factor: f32,
    pub unit_energy_void_factor: f32,
    pub unit_sensor_range: usize,

    pub nebula_tile_vision_reduction: i32,
    pub nebula_tile_energy_reduction: i32,
    pub nebula_tile_drift_speed: f32,
    pub energy_node_drift_speed: f32,
    pub energy_node_drift_magnitude: f32,
}

impl Default for VariableParams {
    fn default() -> Self {
        Self {
            unit_move_cost: 2,
            unit_sap_cost: 10,
            unit_sap_range: 4,
            unit_sap_dropoff_factor: 0.5,
            unit_energy_void_factor: 0.125,
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

impl From<VariableParams> for KnownVariableParams {
    fn from(params: VariableParams) -> Self {
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
        Self::from(VariableParams::default())
    }
}
