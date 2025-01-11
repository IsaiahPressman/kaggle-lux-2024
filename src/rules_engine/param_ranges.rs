use crate::rules_engine::params::VariableParams;
use rand::rngs::ThreadRng;
use rand::seq::SliceRandom;
use serde::Deserialize;
use std::sync::LazyLock;

pub static PARAM_RANGES: LazyLock<ParamRanges> =
    LazyLock::new(load_param_ranges);
// Speed 0.03 is a special case that works out the same as speed 0.01
pub const IRRELEVANT_ENERGY_NODE_DRIFT_SPEED: f32 = 0.03;

fn load_param_ranges() -> ParamRanges {
    let json_data = include_str!("../data/env_params_ranges.json");
    serde_json::from_str(json_data).unwrap()
}

#[derive(Debug, Clone, Deserialize)]
pub struct ParamRanges {
    pub unit_move_cost: Vec<i32>,
    pub unit_sap_cost: Vec<i32>,
    pub unit_sap_range: Vec<isize>,
    pub unit_sap_dropoff_factor: Vec<f32>,
    pub unit_energy_void_factor: Vec<f32>,
    pub unit_sensor_range: Vec<usize>,

    pub nebula_tile_vision_reduction: Vec<i32>,
    pub nebula_tile_energy_reduction: Vec<i32>,
    pub nebula_tile_drift_speed: Vec<f32>,
    pub energy_node_drift_speed: Vec<f32>,
    pub energy_node_drift_magnitude: Vec<f32>,
}

impl ParamRanges {
    pub fn random_params(&self, rng: &mut ThreadRng) -> VariableParams {
        VariableParams {
            unit_move_cost: *self.unit_move_cost.choose(rng).unwrap(),
            unit_sap_cost: *self.unit_sap_cost.choose(rng).unwrap(),
            unit_sap_range: *self.unit_sap_range.choose(rng).unwrap(),
            unit_sap_dropoff_factor: *self
                .unit_sap_dropoff_factor
                .choose(rng)
                .unwrap(),
            unit_energy_void_factor: *self
                .unit_energy_void_factor
                .choose(rng)
                .unwrap(),
            unit_sensor_range: *self.unit_sensor_range.choose(rng).unwrap(),

            nebula_tile_vision_reduction: *self
                .nebula_tile_vision_reduction
                .choose(rng)
                .unwrap(),
            nebula_tile_energy_reduction: *self
                .nebula_tile_energy_reduction
                .choose(rng)
                .unwrap(),
            nebula_tile_drift_speed: *self
                .nebula_tile_drift_speed
                .choose(rng)
                .unwrap(),
            energy_node_drift_speed: *self
                .energy_node_drift_speed
                .choose(rng)
                .unwrap(),
            energy_node_drift_magnitude: *self
                .energy_node_drift_magnitude
                .choose(rng)
                .unwrap(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_param_ranges() {
        _ = PARAM_RANGES.clone();
    }

    #[test]
    fn test_irrelevant_energy_drift_speed() {
        assert!(PARAM_RANGES
            .energy_node_drift_speed
            .contains(&IRRELEVANT_ENERGY_NODE_DRIFT_SPEED));
        assert!(PARAM_RANGES.energy_node_drift_speed.contains(&0.01));
    }
}
