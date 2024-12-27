use crate::rules_engine::params::VariableParams;
use rand::rngs::ThreadRng;
use rand::seq::SliceRandom;
use serde::Deserialize;
use std::iter::Iterator;
use std::sync::LazyLock;

pub static PARAM_RANGES: LazyLock<ParamRanges> =
    LazyLock::new(load_param_ranges);
pub static UNIT_SAP_COST_MIN: LazyLock<i32> = LazyLock::new(get_min_sap_cost);
pub static UNIT_SAP_COST_MAX: LazyLock<i32> = LazyLock::new(get_max_sap_cost);

fn load_param_ranges() -> ParamRanges {
    let json_data = include_str!("../data/env_params_ranges.json");
    serde_json::from_str(json_data).unwrap()
}

fn get_min_sap_cost() -> i32 {
    *PARAM_RANGES.unit_sap_cost.iter().min().unwrap()
}

fn get_max_sap_cost() -> i32 {
    *PARAM_RANGES.unit_sap_cost.iter().max().unwrap()
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
}
