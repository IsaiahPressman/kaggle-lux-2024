pub struct Params {
    pub max_steps_in_match: u32,
    pub map_size: [usize; 2],
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
    pub fn default() -> Self {
        Params {
            max_steps_in_match: 100,
            map_size: [24, 24],
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
            max_relic_nodes: 6,
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
