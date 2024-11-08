use serde::Deserialize;
use std::fs;
use std::path::Path;
use std::sync::LazyLock;

pub static PARAM_RANGES: LazyLock<ParamRanges> =
    LazyLock::new(load_param_ranges);

fn load_param_ranges() -> ParamRanges {
    let path = Path::new(file!())
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("data")
        .join("env_params_ranges.json");
    let json_data = fs::read_to_string(path).unwrap();
    serde_json::from_str(&json_data).unwrap()
}

#[derive(Debug, Clone, Deserialize)]
pub struct ParamRanges {
    pub nebula_tile_vision_reduction: Vec<i32>,
    pub nebula_tile_energy_reduction: Vec<i32>,
    pub unit_sap_dropoff_factor: Vec<f32>,
    pub unit_energy_void_factor: Vec<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_param_ranges() {
        _ = PARAM_RANGES.clone();
    }
}
